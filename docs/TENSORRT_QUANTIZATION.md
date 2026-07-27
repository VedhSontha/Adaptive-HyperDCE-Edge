# TensorRT INT8 Quantization Workflow

Optimizing Zero-DCE on NVIDIA Jetson Nano:
1. Export PyTorch model to ONNX: `torch.onnx.export(...)`.
2. Generate INT8 calibration cache using LOL dataset subset.
3. Build TRT Engine: `trtexec --onnx=model.onnx --saveEngine=model.trt --int8`.
