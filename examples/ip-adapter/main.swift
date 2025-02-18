import Diffusion
import Foundation
import NNC
import NNCPythonConversion
import PNG
import PythonKit

let torch = Python.import("torch")
let diffusers = Python.import("diffusers")
let Image = Python.import("PIL.Image")
let ip_adapter = Python.import("ip_adapter")

let base_model_path = "SG161222/RealVisXL_V1.0"
let image_encoder_path = "/home/liu/workspace/IP-Adapter/models/image_encoder"
let ip_ckpt = "/home/liu/workspace/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
let device = "cuda"

// load SDXL pipeline
let pipe = diffusers.StableDiffusionXLPipeline.from_pretrained(
  base_model_path,
  torch_dtype: torch.float16,
  add_watermarker: false
)

// load ip-adapter
let ip_model = ip_adapter.IPAdapterPlusXL(pipe, image_encoder_path, ip_ckpt, device, num_tokens: 16)

// read image prompt
let image = Image.open("/home/liu/workspace/IP-Adapter/assets/images/woman.png")
image.resize(PythonObject(tupleOf: 512, 512))

// generate image variations with only image prompt
let num_samples = 1
let images = ip_model.generate(
  pil_image: image, num_samples: num_samples, num_inference_steps: 30, seed: 42)
let state_dict = ip_model.image_encoder.state_dict()
let proj_state_dict = ip_model.image_proj_model.state_dict()
print(proj_state_dict.keys())

var clip_image = ip_model.clip_image_processor(images: image, return_tensors: "pt").pixel_values
clip_image = clip_image.to(ip_model.device, dtype: torch.float16)

// multimodal prompts
// images = ip_model.generate(pil_image: image, num_samples: num_samples, num_inference_steps: 30, seed: 42, prompt: "best quality, high quality, wearing sunglasses on the beach", scale: 0.5)
// print(images)

func CLIPSelfAttention(k: Int, h: Int, b: Int, t: Int) -> (Model, Model, Model, Model, Model) {
  let x = Input()
  let tokeys = Dense(count: k * h)
  let toqueries = Dense(count: k * h)
  let tovalues = Dense(count: k * h)
  let keys = tokeys(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(x)).reshaped([b, t, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(x).reshaped([b, t, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t, t])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t, t])
  var out = dot * values
  out = out.reshaped([b, h, t, k]).transposed(1, 2).reshaped([b * t, h * k])
  let unifyheads = Dense(count: k * h)
  out = unifyheads(out)
  return (toqueries, tokeys, tovalues, unifyheads, Model([x], [out]))
}

func CLIPResidualAttentionBlock(prefix: String, k: Int, h: Int, b: Int, t: Int) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let ln1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let (toqueries, tokeys, tovalues, unifyheads, attention) = CLIPSelfAttention(
    k: k, h: h, b: b, t: t)
  var out = x.reshaped([b * t, h * k]) + attention(ln1(x))
  let ln2 = LayerNorm(epsilon: 1e-5, axis: [1])
  let fc = Dense(count: k * h * 4)
  let gelu = GELU()
  let proj = Dense(count: k * h)
  out = out + proj(gelu(fc(ln2(out))))
  let reader: (PythonObject) -> Void = { state_dict in
    let ln_1_weight = state_dict["\(prefix).layer_norm1.weight"].type(torch.float).cpu().numpy()
    let ln_1_bias = state_dict["\(prefix).layer_norm1.bias"].type(torch.float).cpu().numpy()
    ln1.weight.copy(from: try! Tensor<Float>(numpy: ln_1_weight))
    ln1.bias.copy(from: try! Tensor<Float>(numpy: ln_1_bias))
    let q_proj_weight = state_dict["\(prefix).self_attn.q_proj.weight"].type(torch.float).cpu()
      .numpy()
    let q_proj_bias = state_dict["\(prefix).self_attn.q_proj.bias"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: q_proj_weight))
    toqueries.bias.copy(from: try! Tensor<Float>(numpy: q_proj_bias))
    let k_proj_weight = state_dict["\(prefix).self_attn.k_proj.weight"].type(torch.float).cpu()
      .numpy()
    let k_proj_bias = state_dict["\(prefix).self_attn.k_proj.bias"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: k_proj_weight))
    tokeys.bias.copy(from: try! Tensor<Float>(numpy: k_proj_bias))
    let v_proj_weight = state_dict["\(prefix).self_attn.v_proj.weight"].type(torch.float).cpu()
      .numpy()
    let v_proj_bias = state_dict["\(prefix).self_attn.v_proj.bias"].type(torch.float).cpu().numpy()
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: v_proj_weight))
    tovalues.bias.copy(from: try! Tensor<Float>(numpy: v_proj_bias))
    let out_proj_weight = state_dict["\(prefix).self_attn.out_proj.weight"].type(torch.float).cpu()
      .numpy()
    let out_proj_bias = state_dict["\(prefix).self_attn.out_proj.bias"].type(torch.float).cpu()
      .numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: out_proj_weight))
    unifyheads.bias.copy(from: try! Tensor<Float>(numpy: out_proj_bias))
    let ln_2_weight = state_dict["\(prefix).layer_norm2.weight"].type(torch.float).cpu().numpy()
    let ln_2_bias = state_dict["\(prefix).layer_norm2.bias"].type(torch.float).cpu().numpy()
    ln2.weight.copy(from: try! Tensor<Float>(numpy: ln_2_weight))
    ln2.bias.copy(from: try! Tensor<Float>(numpy: ln_2_bias))
    let c_fc_weight = state_dict["\(prefix).mlp.fc1.weight"].type(torch.float).cpu().numpy()
    let c_fc_bias = state_dict["\(prefix).mlp.fc1.bias"].type(torch.float).cpu().numpy()
    fc.weight.copy(from: try! Tensor<Float>(numpy: c_fc_weight))
    fc.bias.copy(from: try! Tensor<Float>(numpy: c_fc_bias))
    let c_proj_weight = state_dict["\(prefix).mlp.fc2.weight"].type(torch.float).cpu().numpy()
    let c_proj_bias = state_dict["\(prefix).mlp.fc2.bias"].type(torch.float).cpu().numpy()
    proj.weight.copy(from: try! Tensor<Float>(numpy: c_proj_weight))
    proj.bias.copy(from: try! Tensor<Float>(numpy: c_proj_bias))
  }
  return (reader, Model([x], [out]))
}

func VisionTransformer(
  grid: Int, width: Int, outputDim: Int, layers: Int, heads: Int, batchSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let classEmbedding = Parameter<Float>(.GPU(0), .CHW(1, 1, width))
  let positionalEmbedding = Parameter<Float>(.GPU(0), .CHW(1, grid * grid + 1, width))
  let conv1 = Convolution(
    groups: 1, filters: width, filterSize: [14, 14], noBias: true,
    hint: Hint(stride: [14, 14]))
  var out = conv1(x).reshaped([batchSize, width, grid * grid]).transposed(1, 2)
  out = Functional.concat(axis: 1, classEmbedding, out)
  out = out + positionalEmbedding
  let lnPre = LayerNorm(epsilon: 1e-5, axis: [2])
  out = lnPre(out)
  var readers = [(PythonObject) -> Void]()
  for i in 0..<layers {
    let (reader, block) = CLIPResidualAttentionBlock(
      prefix: "vision_model.encoder.layers.\(i)", k: width / heads, h: heads, b: batchSize,
      t: grid * grid + 1)
    out = block(out.reshaped([batchSize, grid * grid + 1, width]))
    readers.append(reader)
  }
  /*
  let lnPost = LayerNorm(epsilon: 1e-5, axis: [1])
  out = lnPost(out.reshaped([batchSize, width], strides: [width, 1]))
  let proj = Dense(count: outputDim, noBias: true)
  out = proj(out)
  */
  let reader: (PythonObject) -> Void = { state_dict in
    let conv1_weight = state_dict["vision_model.embeddings.patch_embedding.weight"].type(
      torch.float
    ).cpu().numpy()
    conv1.weight.copy(from: try! Tensor<Float>(numpy: conv1_weight))
    let class_embedding_weight = state_dict["vision_model.embeddings.class_embedding"].type(
      torch.float
    ).cpu().numpy()
    classEmbedding.weight.copy(from: try! Tensor<Float>(numpy: class_embedding_weight))
    let positional_embedding_weight = state_dict[
      "vision_model.embeddings.position_embedding.weight"
    ].type(torch.float).cpu().numpy()
    positionalEmbedding.weight.copy(from: try! Tensor<Float>(numpy: positional_embedding_weight))
    let ln_pre_weight = state_dict["vision_model.pre_layrnorm.weight"].type(torch.float).cpu()
      .numpy()
    let ln_pre_bias = state_dict["vision_model.pre_layrnorm.bias"].type(torch.float).cpu().numpy()
    lnPre.weight.copy(from: try! Tensor<Float>(numpy: ln_pre_weight))
    lnPre.bias.copy(from: try! Tensor<Float>(numpy: ln_pre_bias))
    for reader in readers {
      reader(state_dict)
    }
    /*
    let ln_post_weight = state_dict["vision_model.post_layernorm.weight"].type(torch.float).cpu().numpy()
    let ln_post_bias = state_dict["vision_model.post_layernorm.bias"].type(torch.float).cpu().numpy()
    lnPost.weight.copy(from: try! Tensor<Float>(numpy: ln_post_weight))
    lnPost.bias.copy(from: try! Tensor<Float>(numpy: ln_post_bias))
    let proj_weight = state_dict["visual_projection"].type(torch.float).cpu().numpy()
    // Somehow I have problems with transposed numpy array.
    var projTensor = Tensor<Float>(.CPU, .NC(outputDim, width))
    for i in 0..<outputDim {
      for j in 0..<width {
        projTensor[i, j] = Float(proj_weight[j, i])!
      }
    }
    proj.weight.copy(from: projTensor)
    */
  }
  return (reader, Model([x], [out]))
}

func PerceiverAttention(prefix: String, k: Int, h: Int, b: Int, t: (Int, Int)) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let norm1 = LayerNorm(epsilon: 1e-5, axis: [2])
  let outX = norm1(x)
  let c = Input()
  let norm2 = LayerNorm(epsilon: 1e-5, axis: [2])
  let outC = norm2(c)
  let outXC = Functional.concat(axis: 1, outX, outC)
  let tokeys = Dense(count: k * h, noBias: true)
  let toqueries = Dense(count: k * h, noBias: true)
  let tovalues = Dense(count: k * h, noBias: true)
  let keys = tokeys(outXC).reshaped([b, t.0 + t.1, h, k]).permuted(0, 2, 1, 3)
  let queries = ((1.0 / Float(k).squareRoot()) * toqueries(outC)).reshaped([b, t.1, h, k])
    .permuted(0, 2, 1, 3)
  let values = tovalues(outXC).reshaped([b, t.0 + t.1, h, k]).permuted(0, 2, 1, 3)
  var dot = Matmul(transposeB: (2, 3))(queries, keys)
  dot = dot.reshaped([b * h * t.1, t.0 + t.1])
  dot = dot.softmax()
  dot = dot.reshaped([b, h, t.1, t.0 + t.1])
  var out = dot * values
  out = out.reshaped([b, h, t.1, k]).transposed(1, 2).reshaped([b * t.1, h * k])
  let unifyheads = Dense(count: k * h, noBias: true)
  out = unifyheads(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let norm1_weight = state_dict["\(prefix).norm1.weight"].type(torch.float).cpu().numpy()
    norm1.weight.copy(from: try! Tensor<Float>(numpy: norm1_weight))
    let norm1_bias = state_dict["\(prefix).norm1.bias"].type(torch.float).cpu().numpy()
    norm1.bias.copy(from: try! Tensor<Float>(numpy: norm1_bias))
    let norm2_weight = state_dict["\(prefix).norm2.weight"].type(torch.float).cpu().numpy()
    norm2.weight.copy(from: try! Tensor<Float>(numpy: norm2_weight))
    let norm2_bias = state_dict["\(prefix).norm2.bias"].type(torch.float).cpu().numpy()
    norm2.bias.copy(from: try! Tensor<Float>(numpy: norm2_bias))
    let to_q_weight = state_dict["\(prefix).to_q.weight"].type(torch.float).cpu().numpy()
    toqueries.weight.copy(from: try! Tensor<Float>(numpy: to_q_weight))
    let to_kv_weight = state_dict["\(prefix).to_kv.weight"].type(torch.float).cpu().numpy()
    tokeys.weight.copy(from: try! Tensor<Float>(numpy: to_kv_weight[0..<(k * h), ...]))
    tovalues.weight.copy(from: try! Tensor<Float>(numpy: to_kv_weight[(k * h)..., ...]))
    let to_out_weight = state_dict["\(prefix).to_out.weight"].type(torch.float).cpu().numpy()
    unifyheads.weight.copy(from: try! Tensor<Float>(numpy: to_out_weight))
  }
  return (reader, Model([x, c], [out]))
}

func ResamplerLayer(prefix: String, k: Int, h: Int, b: Int, t: (Int, Int)) -> (
  (PythonObject) -> Void, Model
) {
  let x = Input()
  let c = Input()
  let (attentionReader, attention) = PerceiverAttention(
    prefix: prefix + ".0", k: k, h: h, b: b, t: t)
  var out = c + attention(x, c).reshaped([b, t.1, h * k])
  let layerNorm = LayerNorm(epsilon: 1e-5, axis: [2])
  let fc1 = Dense(count: k * h * 4, noBias: true)
  let gelu = GELU()
  let fc2 = Dense(count: k * h, noBias: true)
  out = out + fc2(gelu(fc1(layerNorm(out))))
  let reader: (PythonObject) -> Void = { state_dict in
    attentionReader(state_dict)
    let layerNorm_weight = state_dict["\(prefix).1.0.weight"].type(torch.float).cpu().numpy()
    layerNorm.weight.copy(from: try! Tensor<Float>(numpy: layerNorm_weight))
    let layerNorm_bias = state_dict["\(prefix).1.0.bias"].type(torch.float).cpu().numpy()
    layerNorm.bias.copy(from: try! Tensor<Float>(numpy: layerNorm_bias))
    let fc1_weight = state_dict["\(prefix).1.1.weight"].type(torch.float).cpu().numpy()
    fc1.weight.copy(from: try! Tensor<Float>(numpy: fc1_weight))
    let fc2_weight = state_dict["\(prefix).1.3.weight"].type(torch.float).cpu().numpy()
    fc2.weight.copy(from: try! Tensor<Float>(numpy: fc2_weight))
  }
  return (reader, Model([x, c], [out]))
}

func Resampler(
  width: Int, outputDim: Int, heads: Int, grid: Int, queries: Int, layers: Int, batchSize: Int
) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let latents = Parameter<Float>(.GPU(0), .CHW(1, queries, width))
  let projIn = Dense(count: width)
  let projX = projIn(x)
  var readers = [(PythonObject) -> Void]()
  let (firstReader, firstLayer) = ResamplerLayer(
    prefix: "layers.0", k: width / heads, h: heads, b: batchSize, t: (grid * grid + 1, queries))
  readers.append(firstReader)
  var out = firstLayer(projX, latents)
  for i in 1..<layers {
    let (reader, layer) = ResamplerLayer(
      prefix: "layers.\(i)", k: width / heads, h: heads, b: batchSize, t: (grid * grid + 1, queries)
    )
    readers.append(reader)
    out = layer(projX, out)
  }
  let projOut = Dense(count: outputDim)
  out = projOut(out)
  let normOut = LayerNorm(epsilon: 1e-5, axis: [2])
  out = normOut(out)
  let reader: (PythonObject) -> Void = { state_dict in
    let proj_in_weight = state_dict["proj_in.weight"].type(torch.float).cpu().numpy()
    projIn.weight.copy(from: try! Tensor<Float>(numpy: proj_in_weight))
    let proj_in_bias = state_dict["proj_in.bias"].type(torch.float).cpu().numpy()
    projIn.bias.copy(from: try! Tensor<Float>(numpy: proj_in_bias))
    let latents_weight = state_dict["latents"].type(torch.float).cpu().numpy()
    latents.weight.copy(from: try! Tensor<Float>(numpy: latents_weight))
    for reader in readers {
      reader(state_dict)
    }
    let proj_out_weight = state_dict["proj_out.weight"].type(torch.float).cpu().numpy()
    projOut.weight.copy(from: try! Tensor<Float>(numpy: proj_out_weight))
    let proj_out_bias = state_dict["proj_out.bias"].type(torch.float).cpu().numpy()
    projOut.bias.copy(from: try! Tensor<Float>(numpy: proj_out_bias))
    let norm_out_weight = state_dict["norm_out.weight"].type(torch.float).cpu().numpy()
    normOut.weight.copy(from: try! Tensor<Float>(numpy: norm_out_weight))
    let norm_out_bias = state_dict["norm_out.bias"].type(torch.float).cpu().numpy()
    normOut.bias.copy(from: try! Tensor<Float>(numpy: norm_out_bias))
  }
  return (reader, Model([x], [out]))
}

let graph = DynamicGraph()
graph.withNoGrad {
  let (reader, vit) = VisionTransformer(
    grid: 16, width: 1280, outputDim: 1024, layers: 31, heads: 16, batchSize: 1)
  let clip_image = clip_image.type(torch.float).cpu().numpy()
  let clipImageTensor = graph.variable(try! Tensor<Float>(numpy: clip_image)).toGPU(0)
  vit.compile(inputs: clipImageTensor)
  reader(state_dict)
  let imageEmbeds = vit(inputs: clipImageTensor)[0].as(of: Float.self).reshaped(.CHW(1, 257, 1280))
  debugPrint(imageEmbeds)
  let (resamplerReader, resampler) = Resampler(
    width: 1280, outputDim: 2048, heads: 20, grid: 16, queries: 16, layers: 4, batchSize: 1)
  resampler.compile(inputs: imageEmbeds)
  resamplerReader(proj_state_dict)
  let imagePromptEmebeds = resampler(inputs: imageEmbeds)[0].as(of: Float.self)
  debugPrint(imagePromptEmebeds)

}
