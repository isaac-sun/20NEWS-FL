import math

import torch
from collections import OrderedDict
from typing import List


def sdfr_attack(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
    eps: float = 1e-10,
) -> OrderedDict:
    """
    Scaled Delta Free-Rider (SDFR) — Zhu et al. [12], SVRFL Sec. 4.

    Paper formula:
        U_f(θ) = (||δ_t|| / ||δ_{t-1}||) · δ_t
    where
        δ_t     = θ(t) − θ(t−1)     (latest global delta)
        δ_{t-1} = θ(t−1) − θ(t−2)   (previous global delta)

    The free-rider replays the latest global change, scaled by the ratio
    of consecutive global-delta norms, so the update magnitude tracks the
    expected trend of honest updates.

    Requires ≥ 2 entries in global_history for the full formula.

    Early-round fallbacks (engineering necessity, not paper formula):
        round 0  (no history):   small random noise
        round 1  (one history):  raw δ_t without scaling

    Symbols → code:
        θ(t)     → global_state_dict  (current global model)
        θ(t−1)   → global_history[-1]
        θ(t−2)   → global_history[-2]
        δ_t      → delta_t dict
        ||·||    → Frobenius norm across all parameters

    Parameters
    ----------
    global_state_dict : current global model θ(t).
    global_history : list of past global states [oldest, …, θ(t−1)].
    eps : numerical stability constant for the norm denominator.
    """
    k = len(global_history)
    update = OrderedDict()

    if k == 0:
        # No history: return negligible random noise.
        for key in global_state_dict:
            update[key] = torch.randn_like(global_state_dict[key]) * 1e-4
        return update

    if k == 1:
        # Only θ(t−1) available; return raw δ_t (unscaled).
        w_prev = global_history[-1]
        for key in global_state_dict:
            update[key] = global_state_dict[key] - w_prev[key]
        return update

    # ── Full paper formula (k ≥ 2) ──────────────────────────────────────
    w_t = global_state_dict
    w_t1 = global_history[-1]   # θ(t−1)
    w_t2 = global_history[-2]   # θ(t−2)

    delta_t_norm_sq = 0.0
    delta_prev_norm_sq = 0.0
    delta_t = OrderedDict()

    for key in w_t:
        dt = w_t[key].float() - w_t1[key].float()
        dp = w_t1[key].float() - w_t2[key].float()
        delta_t[key] = w_t[key] - w_t1[key]       # keep original dtype
        delta_t_norm_sq += dt.pow(2).sum().item()
        delta_prev_norm_sq += dp.pow(2).sum().item()

    scale = math.sqrt(delta_t_norm_sq) / (math.sqrt(delta_prev_norm_sq) + eps)

    for key in delta_t:
        update[key] = delta_t[key] * scale

    return update
