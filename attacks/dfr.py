import torch
from collections import OrderedDict


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
        σ    → sigma
        γ    → gamma  (default 1.0, matching SVRFL experimental setting)
        φ(t) → phi_t
        ε_t  → randn_like(·)

    Parameters
    ----------
    global_state_dict : current global model (used only for shapes/device).
    sigma : initial noise standard deviation.
    round_num : communication round number t (1-indexed).
    gamma : decay exponent; larger → faster decay.
    """
    phi_t = sigma * (max(round_num, 1) ** (-gamma))
    update = OrderedDict()
    for key in global_state_dict:
        update[key] = torch.randn_like(global_state_dict[key]) * phi_t
    return update
