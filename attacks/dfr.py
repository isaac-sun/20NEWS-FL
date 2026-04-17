import torch
from collections import OrderedDict
from typing import List, Optional


def estimate_dfr_sigma(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
) -> Optional[float]:
    """
    Estimate DFR sigma from the first observed global model delta.

    Paper [11] (Fraboni et al.): σ should be calibrated to match the
    scale of the initial global update distribution.  We compute:

        σ_est = std( flatten( θ(1) − θ(0) ) )

    where θ(0) is the initial random model and θ(1) is the model after
    the first aggregation round.

    Returns None when insufficient history (round 0, before any
    aggregation has occurred).
    """
    if len(global_history) < 1:
        return None

    w_init = global_history[0]  # θ(0)

    if len(global_history) >= 2:
        w_after_first = global_history[1]   # θ(1)
    else:
        # Only θ(0) in history; current model IS θ(1).
        w_after_first = global_state_dict

    delta_flat = torch.cat([
        (w_after_first[k].float() - w_init[k].float()).flatten()
        for k in w_init
    ])
    return delta_flat.std().item()


def dfr_attack(
    global_state_dict: OrderedDict,
    sigma: float = 0.5,
    round_num: int = 1,
    gamma: float = 1.0,
) -> OrderedDict:
    """
    Disguised Free-Rider (DFR) — Fraboni et al. [11], SVRFL Sec. 4.

    Paper formula:
        g = φ(t) · ε_t
    where
        φ(t) = σ · t^{−γ}       (noise amplitude, decays over rounds)
        ε_t  ~ N(0, I)           (Gaussian white noise)

    The free-rider performs NO local training.  It sends a noise vector
    whose magnitude shrinks each round so the fake update blends into the
    decreasing honest-update norms.

    Symbols → code:
        σ    → sigma   (ideally estimated via estimate_dfr_sigma();
                        manual value used as fallback when no history)
        γ    → gamma   (default 1.0, matching SVRFL experimental setting)
        φ(t) → phi_t
        ε_t  → randn_like(·)

    Parameters
    ----------
    global_state_dict : current global model (used only for shapes/device).
    sigma : initial noise standard deviation.  Prefer the output of
        estimate_dfr_sigma() over a hand-tuned constant.
    round_num : communication round number t (1-indexed).
    gamma : decay exponent; larger → faster decay.
    """
    phi_t = sigma * (max(round_num, 1) ** (-gamma))
    update = OrderedDict()
    for key in global_state_dict:
        update[key] = torch.randn_like(global_state_dict[key]) * phi_t
    return update
