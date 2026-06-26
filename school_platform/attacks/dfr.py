import torch
from collections import OrderedDict
from typing import List, Optional


def estimate_dfr_sigma(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
    trainable_keys: list = None,
) -> Optional[float]:
    """
    Estimate DFR sigma from the first observed global model delta,
    computed only over trainable parameters.
    """
    if len(global_history) < 1:
        return None

    w_init = global_history[0]
    w_after_first = global_history[1] if len(global_history) >= 2 else global_state_dict
    keys = trainable_keys if trainable_keys is not None else list(w_init.keys())

    delta_flat = torch.cat([
        (w_after_first[k].float() - w_init[k].float()).flatten()
        for k in keys
    ])
    return delta_flat.std().item()


def dfr_attack(
    global_state_dict: OrderedDict,
    sigma: float = 0.5,
    round_num: int = 1,
    gamma: float = 1.0,
    trainable_keys: list = None,
) -> OrderedDict:
    """
    Disguised Free-Rider (DFR) — Fraboni et al. [11], SVRFL Sec. 4.

    Generates Gaussian noise only for trainable (LoRA + head) parameters.
    The frozen backbone params always have delta=0 (not transmitted).

    If trainable_keys is None, generates noise for all keys (backward compat).
    """
    phi_t = sigma * (max(round_num, 1) ** (-gamma))
    keys = trainable_keys if trainable_keys is not None else list(global_state_dict.keys())
    update = OrderedDict()
    for key in keys:
        update[key] = torch.randn_like(global_state_dict[key]) * phi_t
    return update
