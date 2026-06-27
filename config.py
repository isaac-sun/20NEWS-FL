from __future__ import annotations

import math
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
    lora_r: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05

    # Federated Learning
    num_rounds: int = 50
    local_epochs: int = 2              # 2 is stable; 4 causes client drift on small data
    local_lr: float = 0.0005           # typical LoRA lr range: 1e-4 to 5e-4
    server_lr: float = 0.7             # stronger aggregation compensates fewer local epochs
    participation_ratio: float = 0.8
    batch_size: int = 32               # ~32 batches/epoch/client → 512 steps/round

    # ── Shapley evaluation ─────────────────────────────────────────────
    eval_batch_size: int = 256           # large batch for MC Shapley inference

    # ── Local training optimizations ────────────────────────────────────
    label_smoothing: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1

    # ── Round-level cosine LR decay ─────────────────────────────────────
    local_lr_schedule: str = "constant" # LoRA doesn't need LR decay
    local_lr_min: float = 5e-5         # floor for cosine decay

    # ── Server-side optimizations ───────────────────────────────────────
    server_momentum: float = 0.9
    server_lr_decay: float = 1.0        # no server LR decay needed

    # Attack: "none", "dfr", "sdfr", "afr"
    attack_type: str = "none"
    malicious_ratio: float = 0.4

    # DFR: g = sigma * t^{-gamma} * N(0,I)  — Fraboni et al.
    dfr_sigma: float = 0.5
    dfr_gamma: float = 1.0
    dfr_estimate_sigma: bool = True

    # SDFR: U_f = ||delta_t||/||delta_prev|| * delta_t  — Zhu et al.
    # No manual parameters; derived from consecutive global deltas.

    # AFR: SDFR + calibrated sparse noise  — Zhu et al.
    afr_e_cos_beta_override: float | None = None
    afr_noisy_frac: float = 0.1
    afr_base_norm_ema_alpha: float = 0.3

    # Shapley
    num_mc_samples: int = 15

    # Utility
    utility_alpha: float = 0.5

    # General
    seed: int = 42
    device: str = field(default_factory=_detect_device)
    results_dir: str = "results"
    experiment_name: str = "default"

    def get_round_local_lr(self, round_num: int) -> float:
        """Cosine annealing from local_lr → local_lr_min over num_rounds."""
        if self.local_lr_schedule == "constant":
            return self.local_lr
        if self.num_rounds <= 1:
            return self.local_lr
        progress = round_num / (self.num_rounds - 1)
        return self.local_lr_min + 0.5 * (self.local_lr - self.local_lr_min) * (
            1.0 + math.cos(math.pi * progress)
        )
