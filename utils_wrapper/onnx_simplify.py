# simplify_yolov5.py

import numpy as np
import onnx
import onnx.helper as helper
import onnx_graphsurgeon as gs
import onnxruntime
from onnxsim import simplify

onnx_model = onnx.load("/home/manu/mnt/8gpu_3090/test/runs/train/yolov9-s-fire-s1280_11/weights/yolov9-s-converted.onnx")
shape_dict = {"images": (1, 3, 736, 1280)}

# 图优化
onnx_model, check = simplify(onnx_model, dynamic_input_shape=False, overwrite_input_shapes=shape_dict)

graph = gs.import_onnx(onnx_model)
target_model = '/home/manu/tmp/modified_yolov9-s-converted-simplify.onnx'

# 修改输入、输出
graph.inputs = [graph.tensors()["images"].to_variable(dtype=np.float32, shape=shape_dict['images'])]
# graph.outputs = [graph.tensors()["reg0"].to_variable(dtype=np.float32),
#                  graph.tensors()["cls0"].to_variable(dtype=np.float32),
#                  graph.tensors()["reg1"].to_variable(dtype=np.float32),
#                  graph.tensors()["cls1"].to_variable(dtype=np.float32),
#                  graph.tensors()["reg2"].to_variable(dtype=np.float32),
#                  graph.tensors()["cls2"].to_variable(dtype=np.float32),]
graph.outputs = [graph.tensors()["output0"].to_variable(dtype=np.float32),]

# 裁剪无用的node
graph.cleanup().toposort()

opset = [
    helper.make_operatorsetid("ai.onnx", 11)
]

# 修改opset_version
onnx_model = gs.export_onnx(graph, opset_imports=opset, producer_name="pytorch", producer_version="1.7")
onnx.save(onnx_model, target_model)

# 让Netron显示形状
# onnx.save(onnx.shape_inference.infer_shapes(onnx.load(target_model)), target_model)

# onnxruntime推理测试
input_fp32 = np.zeros((1, 3, 736, 1280), dtype=np.float32)
ort_session = onnxruntime.InferenceSession(target_model)
ort_inputs = {ort_session.get_inputs()[0].name: input_fp32}
ort_outs = ort_session.run(None, ort_inputs)
for i in ort_outs:
    print(i.shape)
