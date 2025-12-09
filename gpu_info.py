import torch

def get_gpu_info():
    props = torch.cuda.get_device_properties(0)
    return {
        'name': props.name,
        'compute_capability': f"{props.major}.{props.minor}",
        'multiprocessors': props.multi_processor_count,
        'cuda_cores': props.multi_processor_count * 128,  # Approximate
        'memory_gb': props.total_memory / 1e9,
        'has_tensor_cores': props.major >= 8
    }
print(get_gpu_info())