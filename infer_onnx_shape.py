import onnx
from onnx import shape_inference

model_path = "/home/aalim/lightglue_aliked_3840.onnx"
original_model = onnx.load(model_path)
inferred_model = shape_inference.infer_shapes(original_model)

onnx.save(inferred_model, "/home/aalim/lightglue_aliked_3840_shaped.onnx")
