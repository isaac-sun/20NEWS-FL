from __future__ import annotations

import torch
from dataclasses import dataclass, field


def _detect_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    """Configuration for federated learning experiments."""

    # Data
    model_dir: str = ""                # local model cache dir (skip HF download if set)
    num_clients: int = 10
    iid: bool = True
    val_ratio: float = 0.1
    max_seq_length: int = 512             # 91% docs fully covered (vs 76% at 256)

    # Model
    model_name: str = "distilbert-base-uncased"
    num_classes: int = 20

    # Federated Learning
    num_rounds: int = 50
    local_epochs: int = 2
    local_lr: float = 2e-5             # full DistilBERT fine-tuning rate
    participation_ratio: float = 1.0   # attack derivations assume full participation
    batch_size: int = 8
    num_workers: int = 0

    # ── Shapley evaluation ─────────────────────────────────────────────
    eval_batch_size: int = 64

    # ── Local training optimizations ────────────────────────────────────
    label_smoothing: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # Attack: "none", "dfr", "sdfr", "afr"
    attack_type: str = "none"
    malicious_ratio: float = 0.4
    attack_warmup_rounds: int = 2

    # DFR: g = sigma * t^{-gamma} * N(0,I)  — Fraboni et al.
    dfr_gamma: float = 1.0
    dfr_sigma_override: float | None = None

    # SDFR: U_f = ||delta_t||/||delta_prev|| * delta_t  — Zhu et al.
    # No manual parameters; derived from consecutive global deltas.

    # AFR: SDFR + calibrated sparse noise  — Zhu et al.
    afr_c_override: float | None = None
    afr_noisy_frac: float = 0.1

    # Shapley
    num_mc_samples: int = 15

    # Utility
    utility_alpha: float = 0.5

    # General
    seed: int = 42
    device: str = field(default_factory=_detect_device)
    results_dir: str = "results"
    experiment_name: str = "default"
    gpu_profile: str = "cpu"
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
