# Zero-DCE NVIDIA Jetson Nano Deployment

Steps to accelerate inference on the edge:
1. Convert ONNX to TensorRT engine:
   `trtexec --onnx=model.onnx --saveEngine=model.trt --fp16`
2. Run dynamic routing checks to balance frame latency (target: 35 FPS).
