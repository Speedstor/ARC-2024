import torch
import functools

__all__ = [
    "is_hip",
    "DEVICE_TYPE",
    "DEVICE_COUNT",
]

@functools.cache
def is_hip():
    return bool(getattr(getattr(torch, "version", None), "hip", None))
pass

@functools.cache
def get_device_type():
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        if is_hip():
            return "hip"
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    # Check torch.accelerator
    if hasattr(torch, "accelerator"):
        if not torch.accelerator.is_available():
            raise NotImplementedError("Unsloth cannot find any torch accelerator? You need a GPU.")
        accelerator = str(torch.accelerator.current_accelerator())
        if accelerator in ("cuda", "xpu", "hip"):
            raise RuntimeError(
                f"Unsloth: Weirdly `torch.cuda.is_available()`, `torch.xpu.is_available()` and `is_hip` all failed.\n"\
                f"But `torch.accelerator.current_accelerator()` works with it being = `{accelerator}`\n"\
                f"Please reinstall torch - it's most likely broken :("
            )
    raise NotImplementedError("Unsloth currently only works on NVIDIA, AMD and Intel GPUs.")
pass
DEVICE_TYPE : str = get_device_type()


@functools.cache
def get_device_count():
    if DEVICE_TYPE in ("cuda", "hip"):
        return torch.cuda.device_count()
    elif DEVICE_TYPE == "xpu":
        return torch.xpu.device_count()
    else:
        return 1
pass
DEVICE_COUNT : int = get_device_count()