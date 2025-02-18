import C_ccv
import Diffusion
import Foundation
import NNC
import PNG

typealias FloatType = Float16

func OpenCLIPTextModel(
  vocabularySize: Int, maxLength: Int, embeddingSize: Int, numLayers: Int, numHeads: Int,
  batchSize: Int, intermediateSize: Int
) -> Model {
  let tokens = Input()
  let positions = Input()
  let casualAttentionMask = Input()
  let embedding = CLIPTextEmbedding(
    FloatType.self, batchSize: batchSize,
    vocabularySize: vocabularySize, maxLength: maxLength, embeddingSize: embeddingSize)
  var out = embedding(tokens, positions)
  let k = embeddingSize / numHeads
  var penultimate: Model.IO? = nil
  for i in 0..<numLayers {
    if i == numLayers - 1 {
      penultimate = out
    }
    let encoderLayer =
      OpenCLIPEncoderLayer(
        k: k, h: numHeads, b: batchSize, t: maxLength, intermediateSize: intermediateSize)
    out = encoderLayer(out, casualAttentionMask)
  }
  let finalLayerNorm = LayerNorm(epsilon: 1e-5, axis: [1])
  out = finalLayerNorm(out)
  return Model([tokens, positions, casualAttentionMask], [penultimate!, out])
}

func LabelEmbed(modelChannels: Int) -> Model {
  let x = Input()
  let fc0 = Dense(count: modelChannels * 4)
  var out = fc0(x).swish()
  let fc2 = Dense(count: modelChannels * 4)
  out = fc2(out)
  return Model([x], [out])
}

func CrossAttentionKeysAndValues(k: Int, h: Int, b: Int, hw: Int, t: Int) -> Model {
  let x = Input()
  let keys = Input()
  let values = Input()
  let toqueries = Dense(count: k * h, noBias: true)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, hw, h, k])
    .permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * hw, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, hw, t])
  var out = dot * values
  out = out.reshaped([b, h, hw, k]).transposed(1, 2).reshaped([b, hw, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return Model([x, keys, values], [out])
}

private func BasicTransformerBlock(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> Model {
  let x = Input()
  let keys = Input()
  let values = Input()
  let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  var out = layerNorm1(x)
  let attn1 = SelfAttention(k: k, h: h, b: b, hw: hw)
  out = attn1(out) + x
  var residual = out
  let layerNorm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm2(out)
  let attn2 = CrossAttentionKeysAndValues(
    k: k, h: h, b: b, hw: hw, t: t)
  out = attn2(out, keys, values) + residual
  residual = out
  let layerNorm3 = LayerNorm(epsilon: 1e-5, axis: [2])
  out = layerNorm3(out)
  let ff = FeedForward(hiddenSize: k * h, intermediateSize: intermediateSize)
  out = ff(out) + residual
  return Model([x, keys, values], [out])
}

private func SpatialTransformer(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> Model {
  let x = Input()
  var kvs = [Model.IO]()
  let norm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-6, reduce: [2, 3])
  var out = norm(x)
  let projIn = Convolution(groups: 1, filters: k * h, filterSize: [1, 1])
  let hw = height * width
  out = projIn(out).reshaped([b, k * h, hw]).permuted(0, 2, 1)
  for i in 0..<depth {
    let keys = Input()
    kvs.append(keys)
    let values = Input()
    kvs.append(values)
    let block = BasicTransformerBlock(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    out = block(out, keys, values)
  }
  out = out.reshaped([b, height, width, k * h]).permuted(0, 3, 1, 2)
  let projOut = Convolution(groups: 1, filters: ch, filterSize: [1, 1])
  out = projOut(out) + x
  return Model([x] + kvs, [out])
}

func BlockLayer(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> Model {
  let x = Input()
  let emb = Input()
  var kvs = [Model.IO]()
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let resBlock = ResBlock(b: batchSize, outChannels: channels, skipConnection: skipConnection)
  var out = resBlock(x, emb)
  if attentionBlock > 0 {
    let c = (0..<(attentionBlock * 2)).map { _ in Input() }
    let transformer = SpatialTransformer(
      prefix: "\(prefix).\(layerStart).1",
      ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
      depth: attentionBlock, t: embeddingSize,
      intermediateSize: channels * 4)
    out = transformer([out] + c)
    kvs.append(contentsOf: c)
  }
  return Model([x, emb] + kvs, [out])
}

func MiddleBlock(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  attentionBlock: Int, x: Model.IO, emb: Model.IO
) -> (Model.IO, [Input]) {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let resBlock1 =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  var out = resBlock1(x, emb)
  let kvs = (0..<(attentionBlock * 2)).map { _ in Input() }
  let transformer = SpatialTransformer(
    prefix: "middle_block.1", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
    width: width, depth: attentionBlock, t: embeddingSize, intermediateSize: channels * 4)
  out = transformer([out] + kvs)
  let resBlock2 =
    ResBlock(b: batchSize, outChannels: channels, skipConnection: false)
  out = resBlock2(out, emb)
  return (out, kvs)
}

func InputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int, embeddingSize: Int, attentionRes: [Int: Int], x: Model.IO, emb: Model.IO
) -> ([Model.IO], Model.IO, [Input]) {
  let conv2d = Convolution(
    groups: 1, filters: channels[0], filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  var out = conv2d(x)
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var passLayers = [out]
  var kvs = [Input]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<numRepeat {
      let inputLayer = BlockLayer(
        prefix: "input_blocks",
        layerStart: layerStart, skipConnection: previousChannel != channel,
        attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      previousChannel = channel
      let c = (0..<(attentionBlock * 2)).map { _ in Input() }
      out = inputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      passLayers.append(out)
      layerStart += 1
    }
    if i != channels.count - 1 {
      let downsample = Convolution(
        groups: 1, filters: channel, filterSize: [3, 3],
        hint: Hint(stride: [2, 2], border: Hint.Border(begin: [1, 1], end: [0, 0])))
      out = downsample(out)
      passLayers.append(out)
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  return (passLayers, out, kvs)
}

func OutputBlocks(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], x: Model.IO, emb: Model.IO,
  inputs: [Model.IO]
) -> (Model.IO, [Input]) {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var out = x
  var kvs = [Input]()
  var inputIdx = inputs.count - 1
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: 0]
    for j in 0..<(numRepeat + 1) {
      out = Concat(axis: 1)(out, inputs[inputIdx])
      inputIdx -= 1
      let outputLayer = BlockLayer(
        prefix: "output_blocks",
        layerStart: layerStart, skipConnection: true,
        attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
        batchSize: batchSize,
        height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
      let c = (0..<(attentionBlock * 2)).map { _ in Input() }
      out = outputLayer([out, emb] + c)
      kvs.append(contentsOf: c)
      if i > 0 && j == numRepeat {
        out = Upsample(.nearest, widthScale: 2, heightScale: 2)(out)
        let conv2d = Convolution(
          groups: 1, filters: channel, filterSize: [3, 3],
          hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
        out = conv2d(out)
      }
      layerStart += 1
    }
  }
  return (out, kvs)
}

func UNetXL(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  attentionRes: KeyValuePairs<Int, Int>
) -> Model {
  let x = Input()
  let t_emb = Input()
  let y = Input()
  let middleBlockAttentionBlock = attentionRes.last!.value
  let attentionRes = [Int: Int](uniqueKeysWithValues: attentionRes.map { ($0.key, $0.value) })
  let timeEmbed = TimeEmbed(modelChannels: channels[0])
  let labelEmbed = LabelEmbed(modelChannels: channels[0])
  let emb = timeEmbed(t_emb) + labelEmbed(y)
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let (inputs, inputBlocks, inputKVs) = InputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes,
    x: x, emb: emb)
  var out = inputBlocks
  let (middleBlock, middleKVs) = MiddleBlock(
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingSize: 77, attentionBlock: middleBlockAttentionBlock, x: out, emb: emb)
  out = middleBlock
  let (outputBlocks, outputKVs) = OutputBlocks(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes,
    x: out, emb: emb, inputs: inputs)
  out = outputBlocks
  let outNorm = GroupNorm(axis: 1, groups: 32, epsilon: 1e-5, reduce: [2, 3])
  out = outNorm(out)
  out = Swish()(out)
  let outConv2d = Convolution(
    groups: 1, filters: 4, filterSize: [3, 3],
    hint: Hint(stride: [1, 1], border: Hint.Border(begin: [1, 1], end: [1, 1])))
  out = outConv2d(out)
  return Model([x, t_emb, y] + inputKVs + middleKVs + outputKVs, [out])
}

func CrossAttentionFixed(k: Int, h: Int, b: Int, hw: Int, t: Int) -> Model {
  let c = Input()
  let tokeys = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(c).reshaped([b, t, h, k]).transposed(1, 2)
  let values = tovalues(c).reshaped([b, t, h, k]).transposed(1, 2)
  return Model([c], [keys, values])
}

func BasicTransformerBlockFixed(
  prefix: String, k: Int, h: Int, b: Int, hw: Int, t: Int, intermediateSize: Int
) -> Model {
  let attn2 = CrossAttentionFixed(k: k, h: h, b: b, hw: hw, t: t)
  return attn2
}

func SpatialTransformerFixed(
  prefix: String,
  ch: Int, k: Int, h: Int, b: Int, height: Int, width: Int, depth: Int, t: Int,
  intermediateSize: Int
) -> Model {
  let c = Input()
  var outs = [Model.IO]()
  let hw = height * width
  for i in 0..<depth {
    let block = BasicTransformerBlockFixed(
      prefix: "\(prefix).transformer_blocks.\(i)", k: k, h: h, b: b, hw: hw, t: t,
      intermediateSize: intermediateSize)
    outs.append(block(c))
  }
  return Model([c], outs)
}

func BlockLayerFixed(
  prefix: String,
  layerStart: Int, skipConnection: Bool, attentionBlock: Int, channels: Int, numHeadChannels: Int,
  batchSize: Int, height: Int, width: Int, embeddingSize: Int, intermediateSize: Int
) -> Model {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let transformer = SpatialTransformerFixed(
    prefix: "\(prefix).\(layerStart).1",
    ch: channels, k: k, h: numHeads, b: batchSize, height: height, width: width,
    depth: attentionBlock, t: embeddingSize,
    intermediateSize: channels * 4)
  return transformer
}

func MiddleBlockFixed(
  channels: Int, numHeadChannels: Int, batchSize: Int, height: Int, width: Int, embeddingSize: Int,
  attentionBlock: Int, c: Model.IO
) -> Model.IO {
  precondition(channels % numHeadChannels == 0)
  let numHeads = channels / numHeadChannels
  let k = numHeadChannels
  let transformer = SpatialTransformerFixed(
    prefix: "middle_block.1", ch: channels, k: k, h: numHeads, b: batchSize, height: height,
    width: width, depth: attentionBlock, t: embeddingSize, intermediateSize: channels * 4)
  let out = transformer(c)
  return out
}

func InputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], c: Model.IO
) -> [Model.IO] {
  var layerStart = 1
  var height = startHeight
  var width = startWidth
  var previousChannel = channels[0]
  var ds = 1
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated() {
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<numRepeat {
      if attentionBlock > 0 {
        let inputLayer = BlockLayerFixed(
          prefix: "input_blocks",
          layerStart: layerStart, skipConnection: previousChannel != channel,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        previousChannel = channel
        outs.append(inputLayer(c))
      }
      layerStart += 1
    }
    if i != channels.count - 1 {
      height = height / 2
      width = width / 2
      layerStart += 1
      ds *= 2
    }
  }
  return outs
}

func OutputBlocksFixed(
  channels: [Int], numRepeat: Int, numHeadChannels: Int, batchSize: Int, startHeight: Int,
  startWidth: Int,
  embeddingSize: Int, attentionRes: [Int: Int], c: Model.IO
) -> [Model.IO] {
  var layerStart = 0
  var height = startHeight
  var width = startWidth
  var ds = 1
  var heights = [height]
  var widths = [width]
  var dss = [ds]
  for _ in 0..<channels.count - 1 {
    height = height / 2
    width = width / 2
    ds *= 2
    heights.append(height)
    widths.append(width)
    dss.append(ds)
  }
  var outs = [Model.IO]()
  for (i, channel) in channels.enumerated().reversed() {
    let height = heights[i]
    let width = widths[i]
    let ds = dss[i]
    let attentionBlock = attentionRes[ds, default: 0]
    for _ in 0..<(numRepeat + 1) {
      if attentionBlock > 0 {
        let outputLayer = BlockLayerFixed(
          prefix: "output_blocks",
          layerStart: layerStart, skipConnection: true,
          attentionBlock: attentionBlock, channels: channel, numHeadChannels: numHeadChannels,
          batchSize: batchSize,
          height: height, width: width, embeddingSize: embeddingSize, intermediateSize: channel * 4)
        outs.append(outputLayer(c))
      }
      layerStart += 1
    }
  }
  return outs
}

func UNetXLFixed(
  batchSize: Int, startHeight: Int, startWidth: Int, channels: [Int],
  attentionRes: KeyValuePairs<Int, Int>
) -> Model {
  let c = Input()
  let middleBlockAttentionBlock = attentionRes.last!.value
  let attentionRes = [Int: Int](uniqueKeysWithValues: attentionRes.map { ($0.key, $0.value) })
  let inputBlocks = InputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes,
    c: c)
  var out = inputBlocks
  let middleBlockSizeMult = 1 << (channels.count - 1)
  let middleBlock = MiddleBlockFixed(
    channels: channels.last!, numHeadChannels: 64, batchSize: batchSize,
    height: startHeight / middleBlockSizeMult, width: startWidth / middleBlockSizeMult,
    embeddingSize: 77, attentionBlock: middleBlockAttentionBlock, c: c)
  out.append(middleBlock)
  let outputBlocks = OutputBlocksFixed(
    channels: channels, numRepeat: 2, numHeadChannels: 64, batchSize: batchSize,
    startHeight: startHeight, startWidth: startWidth, embeddingSize: 77, attentionRes: attentionRes,
    c: c)
  out.append(contentsOf: outputBlocks)
  return Model([c], out)
}

let tokenizer0 = CLIPTokenizer(
  vocabulary: "examples/clip/vocab.json", merges: "examples/clip/merges.txt")

let tokenizer1 = CLIPTokenizer(
  vocabulary: "examples/open_clip/vocab_16e6.json",
  merges: "examples/open_clip/bpe_simple_vocab_16e6.txt")

/*
precondition(tokenizer0.vocabulary.count == tokenizer1.vocabulary.count)
for (key, value) in tokenizer0.vocabulary {
  precondition(value == tokenizer1.vocabulary[key])
}

precondition(tokenizer0.bpeRanks.count == tokenizer1.bpeRanks.count)
for (key, value) in tokenizer0.bpeRanks {
  precondition(value == tokenizer1.bpeRanks[key])
}
*/

let prompt =
  //  "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
  //  "a professional photograph of an astronaut riding a horse, detailed, 8k"
  "a smiling indian man with a google t-shirt next to a frowning asian man with a shirt saying nexus at a meeting table facing each other, photograph, detailed, 8k"
let negativePrompt = ""

let tokens0 = tokenizer0.tokenize(text: prompt, truncation: true, maxLength: 77)
let tokens1 = tokenizer1.tokenize(text: prompt, truncation: true, maxLength: 77, paddingToken: 0)
let unconditionalTokens0 = tokenizer0.tokenize(
  text: negativePrompt, truncation: true, maxLength: 77)
let unconditionalTokens1 = tokenizer1.tokenize(
  text: negativePrompt, truncation: true, maxLength: 77, paddingToken: 0)

let graph = DynamicGraph()
graph.maxConcurrency = .limit(1)

/*
graph.withNoGrad {
  var df = DataFrame(fromCSV: "/home/liu/workspace/swift-diffusion/files.txt", automaticUseHeader: false)!
  df["image"] = df["0"].toLoadImage()
  df["resize"] = df["image"]!.toImageJitter(Float.self, size: ImageJitter.Size(rows: 768, cols: 768), resize: ImageJitter.Resize(min: 768, max: 768), centerCrop: true, normalize: ImageJitter.Normalize(mean: [127.5, 127.5, 127.5], std: [127.5, 127.5, 127.5]))
  DynamicGraph.setSeed(0)
  df.shuffle()
  let encoder = Encoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 96, startHeight: 96)
  let decoder = Decoder(
    channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: 96, startHeight: 96)
  var loadedEncoder = false
  var loadedDecoder = false
  for (i, batch) in df["resize", Tensor<Float>.self].enumerated() {
    let initImage = graph.variable(Tensor<FloatType>(from: batch.reshaped(.NCHW(1, 768, 768, 3))).toGPU(0)).permuted(0, 3, 1, 2).copied()
    if !loadedEncoder {
      encoder.compile(inputs: initImage)
      graph.openStore("/home/liu/workspace/swift-diffusion/sdxl_vae_f32_v1.0.ckpt") {
        $0.read("encoder", model: encoder)
      }
      loadedEncoder = true
    }
    let encodedImage = encoder(inputs: initImage)[0].as(of: FloatType.self)[0..<1, 0..<4, 0..<96, 0..<96].copied()
    if !loadedDecoder {
      decoder.compile(inputs: encodedImage)
      graph.openStore("/home/liu/workspace/swift-diffusion/sdxl_vae_f32_v1.0.ckpt") {
        $0.read("decoder", model: decoder)
      }
      loadedDecoder = true
    }
    var result = decoder(inputs: encodedImage)[0].as(of: FloatType.self)
    result = result.toCPU()
    let u8Img = ccv_dense_matrix_new(768, 768, Int32(CCV_8U | CCV_C3), nil, 0)!
    for y in 0..<768 {
      for x in 0..<768 {
        let (r, g, b) = (result[0, 0, y, x], result[0, 1, y, x], result[0, 2, y, x])
        u8Img.pointee.data.u8[y * 768 * 3 + x * 3] = UInt8(
          min(max(Int(Float((r + 1) / 2) * 255), 0), 255))
        u8Img.pointee.data.u8[y * 768 * 3 + x * 3 + 1] = UInt8(
          min(max(Int(Float((g + 1) / 2) * 255), 0), 255))
        u8Img.pointee.data.u8[y * 768 * 3 + x * 3 + 2] = UInt8(
          min(max(Int(Float((b + 1) / 2) * 255), 0), 255))
      }
    }
    let encodedImageCPU = encodedImage.toCPU()
    var smallerImg: UnsafeMutablePointer<ccv_dense_matrix_t>? = nil
    ccv_resample(u8Img, &smallerImg, 0, 0.125, 0.125, Int32(CCV_INTER_AREA))
    for y in 0..<96 {
      for x in 0..<96 {
        print("\(encodedImageCPU[0, 0, y, x]),\(encodedImageCPU[0, 1, y, x]),\(encodedImageCPU[0, 2, y, x]),\(encodedImageCPU[0, 3, y, x]),\(smallerImg!.pointee.data.u8[y * 96 * 3 + x * 3]),\(smallerImg!.pointee.data.u8[y * 96 * 3 + x * 3 + 1]),\(smallerImg!.pointee.data.u8[y * 96 * 3 + x * 3 + 2])")
      }
    }
  }
}
exit(0)
*/

let tokensTensor0 = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
let tokensTensor1 = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
let positionTensor = graph.variable(.CPU, .C(2 * 77), of: Int32.self)
for i in 0..<77 {
  tokensTensor0[i] = unconditionalTokens0[i]
  tokensTensor0[i + 77] = tokens0[i]
  tokensTensor1[i] = unconditionalTokens1[i]
  tokensTensor1[i + 77] = tokens1[i]
  positionTensor[i] = Int32(i)
  positionTensor[i + 77] = Int32(i)
}

let casualAttentionMask = graph.variable(Tensor<FloatType>(.CPU, .NHWC(1, 1, 77, 77)))
casualAttentionMask.full(0)
for i in 0..<76 {
  for j in (i + 1)..<77 {
    casualAttentionMask[0, 0, i, j] = -FloatType.greatestFiniteMagnitude
  }
}

let c0 = graph.withNoGrad {
  let tokensTensorGPU = tokensTensor0.toGPU(0)
  let positionTensorGPU = positionTensor.toGPU(0)
  let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)
  let textModel0 = CLIPTextModel(
    FloatType.self,
    vocabularySize: 49408, maxLength: 77, embeddingSize: 768, numLayers: 11, numHeads: 12,
    batchSize: 2, intermediateSize: 3072, noFinalLayerNorm: true)
  textModel0.compile(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/clip_vit_l14_f32.ckpt") {
    $0.read("text_model", model: textModel0)
  }
  return textModel0(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)[0].as(
    of: FloatType.self
  ).reshaped(.CHW(2, 77, 768))
}

let (c1, pooled) = graph.withNoGrad {
  let tokensTensorGPU = tokensTensor1.toGPU(0)
  let positionTensorGPU = positionTensor.toGPU(0)
  let casualAttentionMaskGPU = casualAttentionMask.toGPU(0)
  let textModel1 = OpenCLIPTextModel(
    vocabularySize: 49408, maxLength: 77, embeddingSize: 1280, numLayers: 32, numHeads: 20,
    batchSize: 2, intermediateSize: 5120)
  let textProjection = graph.variable(.GPU(0), .NC(1280, 1280), of: FloatType.self)
  textModel1.compile(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU)
  graph.openStore("/home/liu/workspace/swift-diffusion/open_clip_vit_bigg14_f16.ckpt") {
    $0.read("text_model", model: textModel1)
    $0.read("text_projection", variable: textProjection)
  }
  let c = textModel1(inputs: tokensTensorGPU, positionTensorGPU, casualAttentionMaskGPU).map {
    $0.as(of: FloatType.self)
  }
  var pooled = graph.variable(.GPU(0), .NC(2, 1280), of: FloatType.self)
  let c1 = c[0].reshaped(.CHW(2, 77, 1280))
  for (i, token) in tokens1.enumerated() {
    if token == tokenizer1.endToken {
      pooled[1..<2, 0..<1280] = c[1][(77 + i)..<(77 + i + 1), 0..<1280] * textProjection
      break
    }
  }
  for (i, token) in unconditionalTokens1.enumerated() {
    if token == tokenizer1.endToken {
      pooled[0..<1, 0..<1280] = c[1][i..<(i + 1), 0..<1280] * textProjection
      break
    }
  }
  return (c1, pooled)
}

let originalWidth = Tensor<FloatType>(
  from: timeEmbedding(
    timestep: 1024, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
let originalHeight = Tensor<FloatType>(
  from: timeEmbedding(
    timestep: 1024, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
var originalSize = Tensor<FloatType>(.CPU, .C(512))
originalSize[0..<256] = originalHeight
originalSize[256..<512] = originalWidth
let cropX = Tensor<FloatType>(
  from: timeEmbedding(timestep: 0, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
let cropY = Tensor<FloatType>(
  from: timeEmbedding(timestep: 0, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
var cropCoord = Tensor<FloatType>(.CPU, .C(512))
cropCoord[0..<256] = cropY
cropCoord[256..<512] = cropX
let targetWidth = Tensor<FloatType>(
  from: timeEmbedding(timestep: 1024, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
let targetHeight = Tensor<FloatType>(
  from: timeEmbedding(
    timestep: 1024, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
var targetSize = Tensor<FloatType>(.CPU, .C(512))
targetSize[0..<256] = targetHeight
targetSize[256..<512] = targetWidth
let aestheticScore = Tensor<FloatType>(
  from: timeEmbedding(timestep: 6.0, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))
let negativeAestheticScore = Tensor<FloatType>(
  from: timeEmbedding(
    timestep: 2.5, batchSize: 1, embeddingSize: 256, maxPeriod: 10_000))

// let random = Python.import("random")
// let numpy = Python.import("numpy")
// let torch = Python.import("torch")

let kvs0 = graph.withNoGrad {
  var crossattn = graph.variable(.GPU(0), .CHW(2, 77, 2048), of: FloatType.self)
  crossattn[0..<2, 0..<77, 0..<768] = c0
  crossattn[0..<2, 0..<77, 768..<2048] = c1
  crossattn[0..<1, 0..<77, 0..<2048].full(0)
  /*
  let crossattnNumpy = numpy.load("/home/liu/workspace/swift-diffusion/context.np.npy")
  let crossattnTensor = try! Tensor<Float>(numpy: crossattnNumpy)
  let crossattn = graph.variable(Tensor<FloatType>(from: crossattnTensor).toGPU(0))
  debugPrint(crossattn)
  */
  let unetBaseFixed = UNetXLFixed(
    batchSize: 2, startHeight: 128, startWidth: 128, channels: [320, 640, 1280],
    attentionRes: [2: 2, 4: 10])
  unetBaseFixed.maxConcurrency = .limit(1)
  unetBaseFixed.compile(inputs: crossattn)
  graph.openStore("/home/liu/workspace/swift-diffusion/sd_xl_base_0.9_f16.ckpt") {
    $0.read("unet_fixed", model: unetBaseFixed)
  }
  return unetBaseFixed(inputs: crossattn).map { $0.as(of: FloatType.self) }
}

let kvs1 = graph.withNoGrad {
  let unetRefinerFixed = UNetXLFixed(
    batchSize: 2, startHeight: 128, startWidth: 128, channels: [384, 768, 1536, 1536],
    attentionRes: [2: 4, 4: 4])
  unetRefinerFixed.maxConcurrency = .limit(1)
  unetRefinerFixed.compile(inputs: c1)
  graph.openStore("/home/liu/workspace/swift-diffusion/sd_xl_refiner_0.9_f16.ckpt") {
    $0.read("unet_fixed", model: unetRefinerFixed)
  }
  return unetRefinerFixed(inputs: c1).map { $0.as(of: FloatType.self) }
}

public struct DiffusionModel {
  public var linearStart: Float
  public var linearEnd: Float
  public var timesteps: Int
  public var steps: Int
}

extension DiffusionModel {
  public var betas: [Float] {  // Linear for now.
    var betas = [Float]()
    let start = linearStart.squareRoot()
    let length = linearEnd.squareRoot() - start
    for i in 0..<timesteps {
      let beta = start + Float(i) * length / Float(timesteps - 1)
      betas.append(beta * beta)
    }
    return betas
  }
  public var alphasCumprod: [Float] {
    var cumprod: Float = 1
    return betas.map {
      cumprod *= 1 - $0
      return cumprod
    }
  }
  // This is Karras scheduler sigmas.
  public func karrasSigmas(_ range: ClosedRange<Float>, rho: Float = 7.0) -> [Float] {
    let minInvRho = pow(range.lowerBound, 1.0 / rho)
    let maxInvRho = pow(range.upperBound, 1.0 / rho)
    var sigmas = [Float]()
    for i in 0..<steps {
      sigmas.append(pow(maxInvRho + Float(i) * (minInvRho - maxInvRho) / Float(steps - 1), rho))
    }
    sigmas.append(0)
    return sigmas
  }

  public func fixedStepSigmas(_ range: ClosedRange<Float>, sigmas sigmasForTimesteps: [Float])
    -> [Float]
  {
    var sigmas = [Float]()
    for i in 0..<steps {
      let timestep = Float(steps - 1 - i) / Float(steps - 1) * Float(timesteps - 1)
      let lowIdx = Int(floor(timestep))
      let highIdx = min(lowIdx + 1, timesteps - 1)
      let w = timestep - Float(lowIdx)
      let logSigma =
        (1 - w) * log(sigmasForTimesteps[lowIdx]) + w * log(sigmasForTimesteps[highIdx])
      sigmas.append(exp(logSigma))
    }
    sigmas.append(0)
    return sigmas
  }

  public static func sigmas(from alphasCumprod: [Float]) -> [Float] {
    return alphasCumprod.map { ((1 - $0) / $0).squareRoot() }
  }

  public static func timestep(from sigma: Float, sigmas: [Float]) -> Float {
    guard sigma > sigmas[0] else {
      return 0
    }
    guard sigma < sigmas[sigmas.count - 1] else {
      return Float(sigmas.count - 1)
    }
    // Find in between which sigma resides.
    var highIdx: Int = sigmas.count - 1
    var lowIdx: Int = 0
    while lowIdx < highIdx - 1 {
      let midIdx = lowIdx + (highIdx - lowIdx) / 2
      if sigma < sigmas[midIdx] {
        highIdx = midIdx
      } else {
        lowIdx = midIdx
      }
    }
    assert(sigma >= sigmas[highIdx - 1] && sigma <= sigmas[highIdx])
    let low = log(sigmas[highIdx - 1])
    let high = log(sigmas[highIdx])
    let logSigma = log(sigma)
    let w = min(max((low - logSigma) / (low - high), 0), 1)
    return (1.0 - w) * Float(highIdx - 1) + w * Float(highIdx)
  }
}

DynamicGraph.setSeed(120)

let unconditionalGuidanceScale: Float = 5
let scaleFactor: Float = 0.13025
let startHeight = 128
let startWidth = 128
let refinerTimestep: Float = 300
let model = DiffusionModel(linearStart: 0.00085, linearEnd: 0.012, timesteps: 1_000, steps: 30)
let alphasCumprod = model.alphasCumprod
let sigmasForTimesteps = DiffusionModel.sigmas(from: alphasCumprod)
// This is for Karras scheduler (used in DPM++ 2M Karras)
let sigmas = model.karrasSigmas(sigmasForTimesteps[0]...sigmasForTimesteps[999])
// let sigmas: [Float] = [14.6146, 11.9484,  9.9172,  8.3028,  6.9739,  5.9347,  5.0878,  4.3728,
//          3.7997,  3.3211,  2.9183,  2.5671,  2.2765,  2.0260,  1.8024,  1.6129,
//          1.4458,  1.2931,  1.1606,  1.0410,  0.9324,  0.8299,  0.7380,  0.6524,
//          0.5693,  0.4924,  0.4179,  0.3417,  0.2653,  0.1793,  0.0000]

let startTime = Date()
let z = graph.withNoGrad {
  var vector0 = graph.variable(.GPU(0), .NC(2, 2816), of: FloatType.self)
  vector0[0..<2, 0..<1280] = pooled
  vector0[0..<1, 0..<1280].full(0)
  vector0[0..<1, 1280..<1792] = graph.variable(originalSize.toGPU(0))
  vector0[1..<2, 1280..<1792] = graph.variable(originalSize.toGPU(0))
  vector0[0..<1, 1792..<2304] = graph.variable(cropCoord.toGPU(0))
  vector0[1..<2, 1792..<2304] = graph.variable(cropCoord.toGPU(0))
  vector0[0..<1, 2304..<2816] = graph.variable(targetSize.toGPU(0))
  vector0[1..<2, 2304..<2816] = graph.variable(targetSize.toGPU(0))
  var vector1 = graph.variable(.GPU(0), .NC(2, 2560), of: FloatType.self)
  vector1[0..<2, 0..<1280] = pooled
  vector1[0..<1, 1280..<1792] = graph.variable(originalSize.toGPU(0))
  vector1[1..<2, 1280..<1792] = graph.variable(originalSize.toGPU(0))
  vector1[0..<1, 1792..<2304] = graph.variable(cropCoord.toGPU(0))
  vector1[1..<2, 1792..<2304] = graph.variable(cropCoord.toGPU(0))
  vector1[0..<1, 2304..<2560] = graph.variable(negativeAestheticScore.toGPU(0))
  vector1[1..<2, 2304..<2560] = graph.variable(aestheticScore.toGPU(0))
  /*
  let vectorNumpy = numpy.load("/home/liu/workspace/swift-diffusion/y.np.npy")
  let vectorTensor = try! Tensor<Float>(numpy: vectorNumpy)
  let vector = graph.variable(Tensor<FloatType>(from: vectorTensor).toGPU(0))
  debugPrint(vector)

  random.seed(42)
  numpy.random.seed(42)
  torch.manual_seed(42)
  torch.cuda.manual_seed_all(42)

  let torchX = torch.randn([1, 4, 128, 128]).numpy()
  let x_T = graph.variable(Tensor<FloatType>(from: try! Tensor<Float>(numpy: torchX)).toGPU(0))
  */
  let x_T = graph.variable(.GPU(0), .NCHW(1, 4, 128, 128), of: FloatType.self)
  x_T.randn(std: 1, mean: 0)
  var x = x_T
  var xIn = graph.variable(.GPU(0), .NCHW(2, 4, 128, 128), of: FloatType.self)
  let ts = timeEmbedding(timestep: 0, batchSize: 2, embeddingSize: 320, maxPeriod: 10_000).toGPU(0)
  /*
  let streamlit_helpers = Python.import("scripts.demo.streamlit_helpers")
  var version_dict: [String: PythonObject] = [
    "H": 1024,
    "W": 1024,
    "C": 4,
    "f": 8,
    "is_legacy": false,
    "config": "/home/liu/workspace/generative-models/configs/inference/sd_xl_base.yaml",
    "ckpt": "/home/liu/workspace/generative-models/checkpoints/sd_xl_base_0.9.safetensors",
    "is_guided": true,
  ]
  let state = streamlit_helpers.init_st(version_dict)
  let unetModel = state["model"].model
  let crossattnNumpy = numpy.load("/home/liu/workspace/swift-diffusion/context.np.npy")
  let crossattnPy = torch.from_numpy(crossattnNumpy).cuda()
  let vectorPy = torch.from_numpy(vectorNumpy).cuda()
  */
  var unet = UNetXL(
    batchSize: 2, startHeight: 128, startWidth: 128, channels: [320, 640, 1280],
    attentionRes: [2: 2, 4: 10])
  unet.maxConcurrency = .limit(1)
  unet.compile(inputs: [xIn, graph.variable(Tensor<FloatType>(from: ts)), vector0] + kvs0)
  graph.openStore("/home/liu/workspace/swift-diffusion/sd_xl_base_0.9_f16.ckpt") {
    $0.read("unet", model: unet)
  }
  var oldDenoised: DynamicGraph.Tensor<FloatType>? = nil
  // Now do DPM++ 2M Karras sampling. (DPM++ 2S a Karras requires two denoising per step, not ideal for my use case).
  x = sigmas[0] * x
  var refinerPass = false
  // let timesteps = [999, 965, 932, 899, 865, 832, 799, 765, 732, 699, 666, 632, 599, 566, 532, 499, 466, 432, 399, 366, 333, 299, 266, 233, 199, 166, 133, 99, 66, 33]
  for i in 0..<model.steps {
    let sigma = sigmas[i]
    let timestep = DiffusionModel.timestep(from: sigma, sigmas: sigmasForTimesteps)
    if timestep <= refinerTimestep && !refinerPass {
      unet = UNetXL(
        batchSize: 2, startHeight: 128, startWidth: 128, channels: [384, 768, 1536, 1536],
        attentionRes: [2: 4, 4: 4])
      let ts = timeEmbedding(
        timestep: timestep, batchSize: 2, embeddingSize: 384, maxPeriod: 10_000
      )
      .toGPU(0)
      unet.maxConcurrency = .limit(1)
      unet.compile(inputs: [xIn, graph.variable(Tensor<FloatType>(from: ts)), vector1] + kvs1)
      graph.openStore("/home/liu/workspace/swift-diffusion/sd_xl_refiner_0.9_f16.ckpt") {
        $0.read("unet", model: unet)
      }
      refinerPass = true
    }
    let ts = timeEmbedding(
      timestep: timestep, batchSize: 2, embeddingSize: refinerPass ? 384 : 320, maxPeriod: 10_000
    ).toGPU(0)
    let t = graph.variable(Tensor<FloatType>(from: ts))
    let cIn = 1.0 / (sigma * sigma + 1).squareRoot()
    let cOut = -sigma
    xIn[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] = cIn * x
    xIn[1..<2, 0..<4, 0..<startHeight, 0..<startWidth] = cIn * x
    // let etPy = unetModel(torch.from_numpy(xIn.rawValue.toCPU()).cuda(), torch.full([1], timestep).cuda(), ["crossattn": crossattnPy, "vector": vectorPy])
    // var et = graph.variable(Tensor<FloatType>(from: try! Tensor<Float>(numpy: etPy.detach().cpu().numpy())).toGPU(0))
    var et = unet(inputs: xIn, refinerPass ? [t, vector1] + kvs1 : [t, vector0] + kvs0)[0].as(
      of: FloatType.self)
    var etUncond = graph.variable(
      .GPU(0), .NCHW(1, 4, startHeight, startWidth), of: FloatType.self)
    var etCond = graph.variable(
      .GPU(0), .NCHW(1, 4, startHeight, startWidth), of: FloatType.self)
    etUncond[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] =
      et[0..<1, 0..<4, 0..<startHeight, 0..<startWidth]
    etCond[0..<1, 0..<4, 0..<startHeight, 0..<startWidth] =
      et[1..<2, 0..<4, 0..<startHeight, 0..<startWidth]
    et = etUncond + unconditionalGuidanceScale * (etCond - etUncond)
    let denoised = x + cOut * et
    let h = log(sigmas[i]) - log(sigmas[i + 1])
    if let oldDenoised = oldDenoised, i < model.steps - 1 {
      let hLast = log(sigmas[i - 1]) - log(sigmas[i])
      let r = (h / hLast) / 2
      let denoisedD = (1 + r) * denoised - r * oldDenoised
      let w = sigmas[i + 1] / sigmas[i]
      x = w * x - (w - 1) * denoisedD
    } else if i == model.steps - 1 {
      x = denoised
    } else {
      let w = sigmas[i + 1] / sigmas[i]
      x = w * x - (w - 1) * denoised
    }
    oldDenoised = denoised
  }
  return 1.0 / scaleFactor * x
}

graph.withNoGrad {
  let decoder = ModelBuilder {
    let startWidth = $0[0].shape[3]
    let startHeight = $0[0].shape[2]
    return Decoder(
      channels: [128, 256, 512, 512], numRepeat: 2, batchSize: 1, startWidth: startWidth,
      startHeight: startHeight)
  }
  let z32 = DynamicGraph.Tensor<Float>(from: z)
  decoder.compile(inputs: z32)
  graph.openStore("/home/liu/workspace/swift-diffusion/sdxl_vae_v1.0_f16.ckpt") {
    $0.read("decoder", model: decoder)
  }
  let img = decoder(inputs: z32)[0].as(of: Float.self)
    .toCPU()
  print("Total time \(Date().timeIntervalSince(startTime))")
  var rgba = [PNG.RGBA<UInt8>](repeating: .init(0), count: startWidth * 8 * startHeight * 8)
  for y in 0..<startHeight * 8 {
    for x in 0..<startWidth * 8 {
      let (r, g, b) = (img[0, 0, y, x], img[0, 1, y, x], img[0, 2, y, x])
      rgba[y * startWidth * 8 + x].r = UInt8(
        min(max(Int(Float((r + 1) / 2) * 255), 0), 255))
      rgba[y * startWidth * 8 + x].g = UInt8(
        min(max(Int(Float((g + 1) / 2) * 255), 0), 255))
      rgba[y * startWidth * 8 + x].b = UInt8(
        min(max(Int(Float((b + 1) / 2) * 255), 0), 255))
    }
  }
  let image = PNG.Data.Rectangular(
    packing: rgba, size: (startWidth * 8, startHeight * 8),
    layout: PNG.Layout(format: .rgb8(palette: [], fill: nil, key: nil)))
  try! image.compress(path: "/home/liu/workspace/swift-diffusion/txt2img.png", level: 4)
}
