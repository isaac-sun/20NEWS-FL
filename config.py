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
    malicious_ratio: float = 0.3
    dfr_noise_scale: float = 0.001
    sdfr_scale: float = 0.5
    afr_noise_scale: float = 0.0005

    # Shapley
    num_mc_samples: int = 30

    # Detection
    detection_threshold_h: float = 200.0
    utility_alpha: float = 0.5

    # General
    seed: int = 42
    device: str = "cpu"
    results_dir: str = "results"
    experiment_name: str = "default"
