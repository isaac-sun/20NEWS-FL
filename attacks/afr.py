import math

import torch
from collections import OrderedDict
from typing import List, Optional

from attacks.sdfr import sdfr_attack


def afr_attack(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
    n_participants: int = 8,
    e_cos_beta: float = 0.5,
    noisy_frac: float = 0.1,
    seed: Optional[int] = None,
    eps: float = 1e-10,
) -> OrderedDict:
    """
    Advanced Free-Rider (AFR) — Zhu et al. [12], SVRFL Sec. 4.

    Extends SDFR by adding calibrated sparse Gaussian noise so that the
    cosine similarity between the AFR update and honest updates stays
    plausible.

    Paper formula for noise magnitude:
        |φ(t)| = sqrt( n² / (n + (n²−n)·E[cos β]) − 1 ) · |E[U_f(θ)]|

    Noise is applied to d randomly selected coordinates only:
        z_i ~ N(0, φ(t)² / d)   for each selected coordinate
        z_i  = 0                 elsewhere
    This yields ||z|| ≈ |φ(t)| regardless of d.

    Symbols → code:
        n            → n_participants
        D            → total_dim  (total model parameters)
        d            → num_noisy = int(D · noisy_frac)
        E[cos β]     → e_cos_beta
        |E[U_f(θ)]|  → base_norm = ||SDFR output||
        U_f(θ)       → base_update from sdfr_attack()
        φ(t)         → phi_t

    Parameters
    ----------
    global_state_dict : current global model θ(t).
    global_history : list of past global states.
    n_participants : number of clients participating this round (n).
    e_cos_beta : expected pairwise cosine similarity between honest
        updates.  The paper estimates this from training dynamics; here
        we accept a configurable value because exact estimation requires
        access to individual honest updates, unavailable to the free-rider.
    noisy_frac : fraction of model parameters to perturb (d / D).
    seed : optional RNG seed for reproducible coordinate selection.
    eps : numerical stability constant.
    """
    # ── Step 1: SDFR base update ────────────────────────────────────────
    base_update = sdfr_attack(global_state_dict, global_history, eps=eps)

    # ── Step 2: flatten into a single vector ────────────────────────────
    keys = list(base_update.keys())
    shapes = [base_update[k].shape for k in keys]
    flat_base = torch.cat([base_update[k].flatten() for k in keys])
    total_dim = flat_base.numel()                       # D

    # ── Step 3: compute noise magnitude |φ(t)| ─────────────────────────
    base_norm = flat_base.float().norm().item()         # |E[U_f(θ)]|

    n = max(n_participants, 2)
    denom = n + (n * n - n) * e_cos_beta
    ratio = (n * n) / (denom + eps)
    phi_t_sq = max(ratio - 1.0, 0.0) * (base_norm ** 2)
    phi_t = math.sqrt(phi_t_sq)

    # ── Step 4: select d random coordinates ─────────────────────────────
    num_noisy = max(int(total_dim * noisy_frac), 1)     # d

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    else:
        gen.manual_seed(torch.randint(0, 2**31, (1,)).item())

    indices = torch.randperm(total_dim, generator=gen)[:num_noisy]

    # ── Step 5: sparse noise  z ~ N(0, φ²/d) on selected coords ───────
    noise_std = phi_t / math.sqrt(num_noisy)
    noise_values = torch.randn(num_noisy, generator=gen).to(
        dtype=flat_base.dtype, device=flat_base.device
    ) * noise_std

    flat_noise = torch.zeros_like(flat_base)
    flat_noise[indices] = noise_values

    # ── Step 6: reconstruct update dict ─────────────────────────────────
    flat_result = flat_base + flat_noise

    update = OrderedDict()
    offset = 0
    for k, shape in zip(keys, shapes):
        numel = 1
        for s in shape:
            numel *= s
        update[k] = flat_result[offset:offset + numel].reshape(shape)
        offset += numel

    return update
