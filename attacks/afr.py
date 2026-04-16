import math
import torch
from collections import OrderedDict
from typing import List

from attacks.sdfr import sdfr_attack


def afr_attack(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
    noise_scale: float = 0.005,
) -> OrderedDict:
    """
    Advanced Free-Rider (AFR): SDFR base delta + dimension-calibrated
    Gaussian noise.

    Paper: builds on SDFR; adds Gaussian noise whose magnitude is
    calibrated by model dimensionality d and E(cos β).

    Implementation:
        base   = SDFR delta  (3-round normalised global change)
        d      = total number of model parameters
        sigma_n = noise_scale * ||base|| / sqrt(d)
        noise  ~ N(0, sigma_n²)
        update = base + noise

    Scaling by 1/sqrt(d) ensures the per-dimension noise is proportional
    to the average parameter perturbation magnitude regardless of model
    size, matching the E(cos β) calibration intent in the paper.
    """
    # Reuse SDFR to get the base delta
    base_update = sdfr_attack(global_state_dict, global_history)

    # Compute total model dimension and ||base|| for calibration
    total_dim = sum(v.numel() for v in base_update.values())
    base_norm = math.sqrt(
        sum(v.float().pow(2).sum().item() for v in base_update.values())
    )
    sigma_n = noise_scale * base_norm / math.sqrt(max(total_dim, 1))

    update = OrderedDict()
    for key in base_update:
        noise = torch.randn_like(base_update[key]) * sigma_n
        update[key] = base_update[key] + noise
    return update
