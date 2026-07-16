# ONNX Input Shape Specifications

Validation configurations for ONNX runtime engines:
- Dynamic dimensions allowed on batch, height, and width parameters.
- Standard signature input: `['batch_size', 3, 'height', 'width']`
- Restricts model conversions from resizing frames statically.
