from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPUProfile:
    name: str
    train_batch_size: int
    eval_batch_size: int
    num_workers: int


def resolve_gpu_profile(requested: str, total_memory_gb: float) -> GPUProfile:
    """Resolve deterministic runtime settings for one NVIDIA GPU."""
    if requested not in {"auto", "t4", "large"}:
        raise ValueError(f"unknown GPU profile: {requested}")

    if requested == "t4":
        name = "t4"
    elif requested == "large":
        name = "large"
    elif total_memory_gb <= 16.0:
        name = "t4"
    elif total_memory_gb <= 40.0:
        name = "medium"
    else:
        name = "large"

    if name == "t4":
        return GPUProfile(name, train_batch_size=8, eval_batch_size=16, num_workers=2)
    if name == "medium":
        return GPUProfile(name, train_batch_size=8, eval_batch_size=32, num_workers=4)
    return GPUProfile(name, train_batch_size=8, eval_batch_size=64, num_workers=4)


def require_cuda(require: bool, detected_device: str) -> None:
    """Fail early instead of silently running a multi-hour experiment on CPU."""
    if require and detected_device != "cuda":
        raise RuntimeError(
            "CUDA is required but no NVIDIA GPU is available. "
            "Check nvidia-smi / the Colab GPU runtime and retry."
        )
