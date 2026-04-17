from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for federated learning experiments."""

    # Data
    num_clients: int = 10
    iid: bool = True
    val_ratio: float = 0.1
    max_features: int = 10000

    # Model
    hidden_dim: int = 256
    num_classes: int = 20

    # Federated Learning
    num_rounds: int = 30
    local_epochs: int = 3
    local_lr: float = 0.001
    server_lr: float = 1.0
    participation_ratio: float = 0.8
    batch_size: int = 64

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
    device: str = "cpu"
    results_dir: str = "results"
    experiment_name: str = "default"
