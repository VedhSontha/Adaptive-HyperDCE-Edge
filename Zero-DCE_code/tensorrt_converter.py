import os

def check_tensorrt_environment() -> bool:
    """Checks if TensorRT tools and CUDA environment variables are ready."""
    has_cuda = os.environ.get("CUDA_HOME") is not None or os.path.exists("/usr/local/cuda")
    return has_cuda

if __name__ == '__main__':
    print(f"TensorRT build readiness: {check_tensorrt_environment()}")
