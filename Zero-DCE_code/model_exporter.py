import torch

def verify_shape_signature(input_tensor: torch.Tensor) -> bool:
    # Verifies dynamic shape signature matches 4D tensors requirements
    shape = input_tensor.shape
    return len(shape) == 4 and shape[1] == 3

if __name__ == '__main__':
    dummy = torch.randn(1, 3, 256, 256)
    print(f"Dynamic shape signature check: {verify_shape_signature(dummy)}")
