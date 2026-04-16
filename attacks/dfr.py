import torch
from collections import OrderedDict


def dfr_attack(
    global_state_dict: OrderedDict,
    sigma: float = 0.01,
    round_num: int = 1,
) -> OrderedDict:
    """
    Disguised Free-Rider (DFR): send Gaussian white noise as model update.

    Paper formula: φ(t) = σ · t^{-γ}, γ = 1  →  noise_scale(t) = σ / t
    The noise amplitude decays with round number, mimicking a client that
    gradually "blends in" by reducing its anomalous signal over time.
    The free-rider does NOT perform any local training.
    """
    noise_scale = sigma / max(round_num, 1)
    update = OrderedDict()
    for key in global_state_dict:
        update[key] = torch.randn_like(global_state_dict[key]) * noise_scale
    return update
