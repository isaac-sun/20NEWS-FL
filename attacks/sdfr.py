import math

import torch
from collections import OrderedDict
from typing import List


def sdfr_attack(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
    eps: float = 1e-10,
    trainable_keys: list = None,
) -> OrderedDict:
    """
    Scaled Delta Free-Rider (SDFR) — Zhu et al. [12], SVRFL Sec. 4.

    Generates updates only for trainable (LoRA + head) parameters.
    """
    keys = trainable_keys if trainable_keys is not None else list(global_state_dict.keys())
    k = len(global_history)
    update = OrderedDict()

    if k == 0:
        for key in keys:
            update[key] = torch.randn_like(global_state_dict[key]) * 1e-4
        return update

    if k == 1:
        w_prev = global_history[-1]
        for key in keys:
            update[key] = global_state_dict[key] - w_prev[key]
        return update

    # Full paper formula (k ≥ 2)
    w_t = global_state_dict
    w_t1 = global_history[-1]
    w_t2 = global_history[-2]

    delta_t_norm_sq = 0.0
    delta_prev_norm_sq = 0.0
    delta_t = OrderedDict()

    for key in keys:
        dt = w_t[key].float() - w_t1[key].float()
        dp = w_t1[key].float() - w_t2[key].float()
        delta_t[key] = w_t[key] - w_t1[key]
        delta_t_norm_sq += dt.pow(2).sum().item()
        delta_prev_norm_sq += dp.pow(2).sum().item()

    scale = math.sqrt(delta_t_norm_sq) / (math.sqrt(delta_prev_norm_sq) + eps)

    for key in delta_t:
        update[key] = delta_t[key] * scale

    return update
