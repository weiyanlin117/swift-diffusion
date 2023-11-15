
import Foundation
import NNC
import NNCPythonConversion
import PythonKit
import Diffusion
import PNG

let torch = Python.import("torch")
let getopt = Python.import("getopt")
let numpy = Python.import("numpy")
let Image = Python.import("PIL.Image")
let sys = Python.import("sys")
let plt =  Python.import("matplotlib.pyplot")  
let os = Python.import("sys")

let np = Python.import("numpy")
let cv2 = Python.import("cv2")
let customPath = "/home/wlin1/drawThings/swift-diffusion/examples/sam"
sys.path.append(customPath)
let segment_anything = Python.import("segment_anything")
let sam_model_registry = segment_anything.sam_model_registry
let show_anns = Python.import("segment_anything.utils.amg").show_anns

// print(show_anns)
var image = cv2.imread("/home/wlin1/drawThings/swift-diffusion/examples/sam/images/dog.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
// print(image)
let figsize: PythonObject = Python.tuple([20, 20])




let sam_checkpoint = "/home/wlin1/drawThings/segment-anything/sam_vit_h_4b8939.pth"
let model_type = "vit_h"

let device = "cuda"
/*
let sam = sam_model_registry[model_type](checkpoint:sam_checkpoint)

sam.to(device:device)
let mask_generator = segment_anything.SamAutomaticMaskGenerator(sam)
let masks = mask_generator.generate(image)
print(masks.count)
print(masks[0].keys())

plt.figure(figsize:figsize)
plt.imshow(image)
show_anns(masks)
plt.axis("off")
plt.savefig("/home/wlin1/drawThings/swift-diffusion/examples/sam/images/my_plot3.png")

*/

// let data =  torch.tensor(data:[0, 1, 2], dtype:tenInput.dtype, device:tenInput.device).view(1, 3, 1, 1)


func getRelPos(_ qSize: Int, _ kSize: Int, _ relPos: Tensor<Float>, graph: DynamicGraph ) -> DynamicGraph.Tensor<Float> {
    let maxRelDist = 2 * max(qSize, kSize) - 1
    var relPosResized = graph.variable(relPos).toGPU(0)

    // Check if interpolation is needed
    if relPos.shape[0] != maxRelDist {
        // Interpolate rel pos
        relPosResized = Upsample(.bilinear, widthScale: Float(maxRelDist)/Float(relPosResized.shape[1]), heightScale: 1)(relPosResized.transposed(1,0)).transposed(0,1)
    }
    
    let qt = (0...qSize).map { Float($0) } 
    let kt = (0...kSize).map { Float($0) } 
    let qr = graph.variable(Tensor<Float32>(qt, .CPU, .NC(qSize, 1))).toGPU(0)
    let kr = graph.variable(Tensor<Float32>(kt, .CPU, .NC(1, kSize))).toGPU(0)
    // Compute the relative coordinates
    let qCoords = qr * max(Float(kSize) / Float(qSize), 1.0)
    let kCoords = kr * max(Float(qSize) / Float(kSize), 1.0)
    var relativeCoords = (qCoords - kCoords) + Float(kSize - 1) * max(Float(qSize) / Float(kSize), 1.0)

    var result = graph.variable(Tensor<Float>(.CPU, .HWC(qSize, kSize, relPos.shape[1]))).toGPU(0)

    relativeCoords = relativeCoords.toCPU()
    for i in 0..<relativeCoords.shape[0] {
      for j in 0..<relativeCoords.shape[1] {
        let index = Int(relativeCoords.rawValue[i,j])
        result[i..<i+1, j..<j+1, 0..<relPos.shape[1]] = relPosResized[index..<index+1, 0..<relPos.shape[1]].reshaped(.HWC(1, 1, relPos.shape[1]))
      }
    }
    
    return result
}

func addDecomposedRelPos(
    attn: Tensor<Float>,
    q: Tensor<Float>,
    relPosH: Tensor<Float>,
    relPosW: Tensor<Float>,
    qSize: (Int, Int),
    kSize: (Int, Int), graph: DynamicGraph
) -> DynamicGraph.Tensor<Float> {
    let (qH, qW) = qSize
    let (kH, kW) = kSize

    // Get relative position for height and width
    var Rh = getRelPos(qH, kH, relPosH, graph:graph)
    var Rw = getRelPos(qW, kW, relPosW, graph:graph)


    let B = attn.shape[0]
    let dim = q.shape[2]

    let rQ = graph.variable(q.reshaped(.NHWC(B, qH, qW, dim))).toGPU(0)
    let rhShape = Rh.shape
    let rwShape = Rw.shape

    precondition(rhShape[2] == dim)
    precondition(rwShape[2] == dim)

    var relH = graph.variable(Tensor<Float>(.CPU, .NHWC(B, qH, qW, rhShape[1]))).toGPU(0)
    var relW = graph.variable(Tensor<Float>(.CPU, .NHWC(B, qH, qW, rwShape[1]))).toGPU(0)

    Rh = Rh.reshaped(.NHWC(1, rhShape[0], rhShape[1], rhShape[2]))
    Rw = Rw.reshaped(.NHWC(1, rwShape[0], rwShape[1], rwShape[2]))


    for i in 0..<B {
        let relH_i = Matmul(transposeB: (2, 3))(rQ[i..<i+1, 0..<qH, 0..<qW, 0..<dim],Rh)
        relH[i..<i+1, 0..<qH, 0..<qW, 0..<rhShape[1]] = relH_i
        
        let transposedA = rQ[i..<i+1, 0..<qH, 0..<qW, 0..<dim].transposed(1,2)
        let relW_i = Matmul(transposeB: (2, 3))(transposedA,Rw).transposed(1,2)
        relW[i..<i+1, 0..<qH, 0..<qW, 0..<rwShape[1]] = relW_i
    }

    let qN = qH * qW
    let kN = kH * kW
    let attnTensor = graph.variable(attn.reshaped(.NHWC(1, B, qH * qW, kH * kW))).toGPU(0)
    var result = graph.variable(Tensor<Float>(.CPU, .NHWC(1, B, qH * qW, kH * kW))).toGPU(0)
    for i in 0..<B {
        var attn = attnTensor[0..<1, i..<i+1, 0..<qN, 0..<kN]
        attn = attn.reshaped(.C(qN*kN)).reshaped(.NHWC(qH, qW, kH, kW))
        let rel_h_row = relH[i..<i+1, 0..<qH, 0..<qW, 0..<rhShape[1]].reshaped(.C(qN*rhShape[1])).reshaped(.NHWC(qH, qW, rhShape[1], 1))
        let rel_w_row = relW[i..<i+1, 0..<qH, 0..<qW, 0..<rwShape[1]].reshaped(.C(qN*rwShape[1])).reshaped(.NHWC(qH, qW,  1, rwShape[1]))
        var row_result = attn + rel_h_row + rel_w_row

        row_result = row_result.reshaped(.NHWC(1, 1, qH * qW, kH * kW))
        result[0..<1, i..<i+1, 0..<qN, 0..<kN] = row_result
    }

    return result
}

func padTensor(_ x: DynamicGraph.Tensor<Float> , _ padH: Int, _ padW: Int, graph: DynamicGraph) ->  DynamicGraph.Tensor<Float> {
    let B = x.shape[0]
    let H = x.shape[1]
    let W = x.shape[2]
    let C = x.shape[3]
    
    // New dimensions after padding
    let newH = H + padH
    let newW = W + padW

    // Create a new tensor with the padded size filled with zeros
    var paddedTensor = graph.variable(.CPU, .NHWC(B, newH, newW, C), of: Float.self)
    paddedTensor.full(0)
    // Copy the contents of 'x' into 'paddedTensor'
    for b in 0..<B {
        for h in 0..<newH {
            for w in 0..<newW {
                for c in 0..<C {
                    if h < H, w < W {
                        paddedTensor[b, h, w, c] = x[b, h, w, c]
                    }
                }
            }
        }
    }

    return paddedTensor
}

func windowPartition( _ x: DynamicGraph.Tensor<Float>, window_size: Int, graph:DynamicGraph ) -> (DynamicGraph.Tensor<Float>, Int, Int) {
    let shape = x.shape
    let B = shape[0]
    let H = shape[1]
    let W = shape[2]
    let C = shape[3]
    var xTensor = x
    let padH = (window_size - H % window_size) % window_size
    let padW = (window_size - W % window_size) % window_size
        if padH > 0 || padW > 0 {
            xTensor = padTensor(x, padH, padW, graph:graph)
        }
    let Hp = H + padH
    let Wp = W + padW

    return (xTensor, Hp, Wp)
}

let graph = DynamicGraph()
graph.withNoGrad {

    let get_rel_pos = Python.import("segment_anything.modeling.image_encoder").get_rel_pos
    let add_decomposed_rel_pos = Python.import("segment_anything.modeling.image_encoder").add_decomposed_rel_pos
    let window_partition = Python.import("segment_anything.modeling.image_encoder").window_partition

    /* test for getRelPos
    let rtensor = torch.randn(27, 80)
    print(rtensor)
    print(rtensor.shape)
    let o = get_rel_pos(14, 14, rtensor)
    print(o)
    print(o.shape)
    let tensor = try! Tensor<Float>(numpy: rtensor.numpy())
    debugPrint(tensor)
    let get = getRelPos(qSize:14, kSize:14, relPos: tensor, graph: graph)
        debugPrint(get)   
    */

    // test for add_decomposed_rel_pos 
    /*
    let attn = torch.randn(3, 4, 4)
    let q = torch.randn(3, 4, 4)
    let rel_pos_h = torch.randn(3, 4)
    let rel_pos_w = torch.randn(3, 4)
    let qsize = Python.tuple([2, 2])
    let ksize = Python.tuple([2, 2])
    let decomposed = add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, qsize, ksize)
    // print(decomposed)

    let attntensor = try! Tensor<Float>(numpy: attn.numpy())
    let qtensor = try! Tensor<Float>(numpy: q.numpy())
    let rphtensor = try! Tensor<Float>(numpy: rel_pos_h.numpy())
    let rpwtensor = try! Tensor<Float>(numpy: rel_pos_w.numpy())
    let relPos = addDecomposedRelPos(attn:attntensor, q:qtensor, relPosH: rphtensor, relPosW: rpwtensor, qSize: (2,2), kSize:(2,2), graph:graph)
    debugPrint(relPos)   
    */

    // windowwindow_partition
    let x = torch.randn(1, 4, 4, 3)
    let windowP = window_partition(x, 3)
    // debugPrint(x)
    let pypad = windowP[0].numpy()
    print(pypad)
    // debugPrint(windowP[0])
    let xtensor = try! Tensor<Float>(numpy: x.numpy())

    let wp = windowPartition(graph.variable(xtensor), window_size:3, graph: graph)
    let pad = wp.0.rawValue.makeNumpyArray()
    debugPrint(pad)

    // for i in 0..<1 {
    //     for j in 0..<70 {
    //         for p in 0..<70 {
    //             for k in 0..<1280 {
    //                 if (abs(pad[i][j][p][k] - pypad[i][j][p][k]) > 0.0001) {
    //                     print(pad[i][j][p][k] - pypad[i][j][p][k])
    //                     print(pad[i][j][p][k])
    //                     print(pypad[i][j][p][k])
    //                     print("??")
    //                 }
    //             }
    //         }
    //         print(j)
    //     }
    // }
}   

/*
class MaskData {
    private var _stats: [String: Any]

    init(_ stats: [String: Any]) {
        for v in stats.values {
            assert(v is [[Float]] || v is [Float], "MaskData only supports [[Float]] and [Float].")
        }
        self._stats = stats
    }

    subscript(key: String) -> Any? {
        get {
            return _stats[key]
        }
        set(newValue) {
            assert(newValue is [[Float]] || newValue is [Float], "MaskData only supports [[Float]] and [Float].")
            _stats[key] = newValue
        }
    }

    func removeValue(forKey key: String) {
        _stats.removeValue(forKey: key)
    }

    func filter(keep: [Bool]) {
        for (key, value) in _stats {
            if let array = value as? [Float] {
                _stats[key] = zip(array, keep).compactMap { (element, shouldKeep) in
                    shouldKeep ? element : nil
                }
            } else if let arrayOfArrays = value as? [[Float]] {
                _stats[key] = zip(arrayOfArrays, keep).compactMap { (subArray, shouldKeep) in
                    shouldKeep ? subArray : nil
                }
            }
        }
    }

    func concatenate(with newStats: MaskData) {
        for (key, value) in newStats._stats {
            if let existingValue = _stats[key] {
                if let existingArray = existingValue as? [Float], let newArray = value as? [Float] {
                    _stats[key] = existingArray + newArray
                } else if let existingArrayOfArrays = existingValue as? [[Float]], let newArrayOfArrays = value as? [[Float]] {
                    _stats[key] = existingArrayOfArrays + newArrayOfArrays
                } else {
                    assertionFailure("Unsupported type for key \(key): \(type(of: value))")
                }
            } else {
                _stats[key] = value
            }
        }
    }

    func convertToNumpy() {
        // You can implement numpy conversion if needed
    }

    func items() -> Dictionary<String, Any>.Iterator {
        return _stats.makeIterator()
    }
}


func generateCropBoxes(imSize: (Int, Int), nLayers: Int, overlapRatio: Float) -> ([Array<Int>], [Int]) {
    var cropBoxes: [Array<Int>] = []
    var layerIdxs: [Int] = []
    let (imH, imW) = imSize
    let shortSide = min(imH, imW)

    // Original image
    cropBoxes.append([0, 0, imW, imH])
    layerIdxs.append(0)

    func cropLen(origLen: Int, nCrops: Int, overlap: Int) -> Int {
        return Int(ceilf((Float(overlap) * Float(nCrops - 1) + Float(origLen)) / Float(nCrops)))
    }

    for iLayer in 0..<nLayers {
        let nCropsPerSide = 1 << (iLayer + 1)
        let overlap = Int(overlapRatio * Float(shortSide) * (2.0 / Float(nCropsPerSide)))

        let cropW = cropLen(origLen: imW, nCrops: nCropsPerSide, overlap: overlap)
        let cropH = cropLen(origLen: imH, nCrops: nCropsPerSide, overlap: overlap)

        let cropBoxX0 = stride(from: 0, to: cropW * nCropsPerSide, by: cropW - overlap).map { Int($0) }
        let cropBoxY0 = stride(from: 0, to: cropH * nCropsPerSide, by: cropH - overlap).map { Int($0) }

        // Crops in XYWH format
        for x0 in cropBoxX0 {
            for y0 in cropBoxY0 {
                let box = [x0, y0, min(x0 + cropW, imW), min(y0 + cropH, imH)]
                cropBoxes.append(box)
                layerIdxs.append(iLayer + 1)
            }
        }
    }

    return (cropBoxes, layerIdxs)
}

let (cropBoxes, layerIdxs) = generateCropBoxes(imSize:(534, 800), nLayers:0, overlapRatio:0.341333)
print(cropBoxes)
print(layerIdxs)


// common.py
// MLPBlock
func MLPBlock(embedding_dim: Int, mlp_dim: Int) -> ((PythonObject) -> Void, Model) {
  let x = Input()
  let fc1 = Dense(count: mlp_dim)
  var out = fc1(x)
  out = GELU()(out)
  let fc2 = Dense(count: embedding_dim)
  out = fc2(out)
  let model = Model([x], [out])

    let reader: (PythonObject) -> Void = { state_dict in
        let projWeight = state_dict["proj.weight"].type(torch.float).cpu().numpy()
        let projBias = state_dict["proj.bias"].type(torch.float).cpu().numpy()
        fc1.weight.copy(from: try! Tensor<Float>(numpy: projWeight))
        fc1.bias.copy(from: try! Tensor<Float>(numpy: projBias))

         let projWeight2 = state_dict["proj.weight"].type(torch.float).cpu().numpy()
        let projBias2 = state_dict["proj.bias"].type(torch.float).cpu().numpy()
        fc2.weight.copy(from: try! Tensor<Float>(numpy: projWeight2))
        fc2.bias.copy(from: try! Tensor<Float>(numpy: projBias2))
    }

    return (reader, model)
}

// LayerNorm2d
//   let layerNorm1 = LayerNorm(epsilon: 1e-5, axis: [2])

// image_encoder.py
// ImageEncoderViT



// class PatchEmbed(nn.Module):

func PatchEmbed(
    kernelSize: (Int, Int) = (16, 16),
    stride: (Int, Int) = (16, 16),
    padding: (Int, Int) = (0, 0),
    inChans: Int = 3,
    embedDim: Int = 768
) -> ((PythonObject) -> Void, Model) {
    let x = Input()

    let proj = Convolution(
        groups: 1,
        filters: embedDim,
        filterSize: [kernelSize.0, kernelSize.1],
        hint: Hint(stride: [stride.0, stride.1], border: Hint.Border(begin: [padding.0, padding.1], end: [padding.0, padding.1])),
        format: .OHWI
    )

    var out = proj(x)
    // Adjusting the dimensions: B C H W -> B H W C
    out = out.transposed(1, 3).transposed(1, 2)

    let model = Model([x], [out])

    let reader: (PythonObject) -> Void = { state_dict in
        let projWeight = state_dict["proj.weight"].type(torch.float).cpu().numpy()
        let projBias = state_dict["proj.bias"].type(torch.float).cpu().numpy()
        proj.weight.copy(from: try! Tensor<Float>(numpy: projWeight))
        proj.bias.copy(from: try! Tensor<Float>(numpy: projBias))
    }

    return (reader, model)
}
*/
/*
func windowUnpartition(windows: Tensor<Float>, windowSize: Int, padHW: (Int, Int), hw: (Int, Int)) -> Tensor<Float> {
    let (Hp, Wp) = padHW
    let (H, W) = hw
    let B = windows.shape[0] / (Hp * Wp / windowSize / windowSize)
    
    // Reshape the windows tensor TODO!!!
    // var x = windows.reshaped([B, Hp / windowSize, Wp / windowSize, windowSize, windowSize, -1])
    
    // Permute the dimensions to get back to the original layout
    x = x.transposed(1, 3).transposed(2, 4).transposed(3, 5).reshaped([B, Hp, Wp, -1])
    
    // Remove the padding if the padded dimensions are larger than the original dimensions
    if Hp > H || Wp > W {
        x = x.slice(lowerBounds: [0, 0, 0, 0], upperBounds: [B, H, W, x.shape[3]])
    }
    
    return x
}

func windowPartition(x: Tensor<Float>, windowSize: Int) -> (Tensor<Float>, (Int, Int)) {
    let B = x.shape[0]
    let H = x.shape[1]
    let W = x.shape[2]
    let C = x.shape[3]

    let padH = (windowSize - H % windowSize) % windowSize
    let padW = (windowSize - W % windowSize) % windowSize
    var paddedX = x
    if padH > 0 || padW > 0 {
        // Add padding to the tensor
        // Note: s4nnc might not have a direct equivalent for F.pad. You may need a custom method to pad the tensor.
        paddedX = padTensor(x, padH: padH, padW: padW) // Implement padTensor function
    }
    let Hp = H + padH
    let Wp = W + padW

    // Reshape and permute the tensor to form windows reshape
    // paddedX = paddedX.reshaped([B, Hp / windowSize, windowSize, Wp / windowSize, windowSize, C])
    // let windows = paddedX.transposed(1, 3).transposed(2, 4).reshaped([-1, windowSize, windowSize, C])

    return (windows, (Hp, Wp))
}
*/


/* relative position


func einsumBhwcHkcToBhwk(_ rQ: Tensor<Float>, _ Rh: Tensor<Float>) -> Tensor<Float> {
    // Assuming rQ has shape [batch, height, width, channels]
    // and Rh has shape [height, some_size, channels]

    // 1. Expand dimensions of rQ and Rh to align them for element-wise multiplication
    let expandedRQ = rQ.expandDims(4) // New shape: [batch, height, width, channels, 1]
    let expandedRh = Rh.expandDims(0).expandDims(3) // New shape: [1, height, some_size, 1, channels]

    // 2. Perform element-wise multiplication
    // The result will have shape [batch, height, width, some_size, channels]
    let multiplied = expandedRQ * expandedRh

    // 3. Sum over the channels dimension
    let sumOverChannels = multiplied.sum(dimensions: [4]) // New shape: [batch, height, width, some_size]

    return sumOverChannels
}

func addDecomposedRelPos(
    attn: Tensor<Float>,
    q: Tensor<Float>,
    relPosH: Tensor<Float>,
    relPosW: Tensor<Float>,
    qSize: (Int, Int),
    kSize: (Int, Int)
) -> Tensor<Float> {
    let (qH, qW) = qSize
    let (kH, kW) = kSize

    // Get relative position for height and width
    let Rh = getRelPos(qH, kH, relPosH)
    let Rw = getRelPos(qW, kW, relPosW)

    let B = attn.shape[0]
    let dim = q.shape[2]

    // Reshape q and perform matrix multiplications similar to torch.einsum
    let rQ = q.reshaped([B, qH, qW, dim])
    // Note: You will need to implement matrix multiplications that correspond to the `einsum` operations in PyTorch.
    // This is a placeholder for the operation and might need to be adapted based on your specific implementation.
    let relH = einsumBhwcHkcToBhwk(rQ, Rh) // Implement matmulRelPos
    let relW = einsumBhwcHkcToBhwk(rQ, Rw) // Implement matmulRelPos

    // Reshape attn and add relative positions
    let attnReshaped = attn.reshaped([B, qH, qW, kH, kW])
    let attnAdded = attnReshaped + relH.expandingDims(4) + relW.expandingDims(3)

    return attnAdded.reshaped([B, qH * qW, kH * kW])
}*/