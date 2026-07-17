# ONNX Engine Shape Validation

Validating image dimensions at runtime:
- ONNX runtime validates input frame shapes: `[batch, 3, height, width]`.
- Enforces dynamic width and height scaling to support multiple camera sensors.
