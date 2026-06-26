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
    max_seq_length: int = 256

    # Model
    model_name: str = "distilbert-base-uncased"
    num_classes: int = 20
    lora_r: int = 32                   # ↑ 8→32: more capacity for 20-class task
    lora_alpha: float = 32.0           # ↑ 16→32: scale proportionally with r

    # Federated Learning
    num_rounds: int = 50               # ↑ 30→50: more FL rounds for convergence
    local_epochs: int = 2              # ↑ 1→2: better local learning per round
    local_lr: float = 0.0005           # typical LoRA lr range: 1e-4 to 5e-4
    server_lr: float = 0.5             # ↓ 1.0→0.5: more stable aggregation
    participation_ratio: float = 0.8
    batch_size: int = 32               # DistilBERT fits comfortably on L20 (48GB)

    # ── Local training optimizations (zero extra FWD/BWD cost) ──────────
    label_smoothing: float = 0.1       # soften one-hot targets → better generalization
    weight_decay: float = 0.01         # L2 regularization on LoRA params
    max_grad_norm: float = 1.0         # gradient clipping for stability
    warmup_ratio: float = 0.1          # linear warmup fraction of total local steps
    lora_dropout: float = 0.05         # dropout on LoRA outputs (standard in PEFT)

    # ── Round-level cosine LR decay (global curriculum) ─────────────────
    local_lr_schedule: str = "cosine"  # "cosine" or "constant"
    local_lr_min: float = 5e-5         # floor for cosine decay (10× below initial)

    # ── Server-side optimizations (zero extra FWD/BWD cost) ─────────────
    server_momentum: float = 0.9       # momentum coefficient for global updates
    server_lr_decay: float = 0.98      # per-round exponential decay of server_lr

    # Attack: "none", "dfr", "sdfr", "afr"
    attack_type: str = "none"
    malicious_ratio: float = 0.4

    # DFR: g = sigma * t^{-gamma} * N(0,I)  — [11] Fraboni et al.
    dfr_sigma: float = 0.5              # fallback; auto-estimated when possible
    dfr_gamma: float = 1.0
    dfr_estimate_sigma: bool = True     # estimate σ from initial global delta

    # SDFR: U_f = ||delta_t||/||delta_prev|| * delta_t  — [12] Zhu et al.
    # No manual parameters; derived from consecutive global deltas.

    # AFR: SDFR + calibrated sparse noise  — [12] Zhu et al.
    # E[cos β] is estimated dynamically from validation loss trajectory.
    # Override below is used ONLY when dynamic estimation is unavailable.
    afr_e_cos_beta_override: float | None = None
    afr_noisy_frac: float = 0.1         # fraction of params to perturb (d/D)
    afr_base_norm_ema_alpha: float = 0.3 # EMA smoothing for |E[U_f(θ)]|

    # Shapley
    num_mc_samples: int = 30

    # Utility
    utility_alpha: float = 0.5

    # General
    seed: int = 42
    device: str = field(default_factory=_detect_device)
    results_dir: str = "results"
    experiment_name: str = "default"

    def get_round_local_lr(self, round_num: int) -> float:
        """Compute local_lr for a given FL round via cosine schedule.

        Cosine annealing from local_lr → local_lr_min over num_rounds.
        Centralized training uses this per-step; FL uses it per-round
        (global curriculum — each round starts from a lower LR).
        """
        if self.local_lr_schedule == "constant":
            return self.local_lr
        if self.num_rounds <= 1:
            return self.local_lr
        progress = round_num / (self.num_rounds - 1)
        return self.local_lr_min + 0.5 * (self.local_lr - self.local_lr_min) * (
            1.0 + math.cos(math.pi * progress)
        )
