# Zero-DCE ONNX Export & Runtime Guide

Exporting PyTorch model to ONNX representation:
- Command: `torch.onnx.export(model, dummy_input, 'model.onnx')`
- Validates input shape configurations (`[1, 3, 256, 256]`) for edge acceleration.
