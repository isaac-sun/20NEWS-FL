import math

import torch
from collections import OrderedDict
from typing import List, Optional, Tuple

from attacks.sdfr import sdfr_attack


# ── Paper-faithful estimation helpers ──────────────────────────────────────

def estimate_e_cos_beta(val_loss_t: float, val_loss_init: float) -> float:
    """
    Estimate E[cos β] from the validation-loss trajectory.

    From [12] (Zhu et al.), the gradient alignment metric λ̄ is related to
    the pairwise cosine similarity of honest client updates by:

        λ̄ = (1 + (n−1)·E[cos β]) / n

    Under a convergence model where λ̄ tracks the loss ratio:

        λ̄(t) ≈ 1/n + (1 − 1/n) · (1 − l(t)/l(1))

    Solving for E[cos β] gives:

        E[cos β](t) = 1 − l(t)/l(1)

    Interpretation:
        • At round 1:  l(t) ≈ l(1) → E[cos β] ≈ 0  (uncorrelated updates)
        • Converged:   l(t) → 0   → E[cos β] → 1    (highly aligned updates)

    Parameters
    ----------
    val_loss_t : validation loss of the global model at current round.
    val_loss_init : validation loss of the global model at round 0 (before
        any training).
    """
    if val_loss_init is None or val_loss_init < 1e-10:
        return 0.0
    ratio = val_loss_t / val_loss_init
    return max(0.0, min(1.0 - ratio, 0.99))


class AFRState:
    """
    Track running statistics for paper-faithful AFR noise calibration.

    Maintains two quantities across rounds:

    1. **val_loss_init** — the initial validation loss l(1) for
       E[cos β](t) = 1 − l(t)/l(1).

    2. **base_norm_ema** — exponential moving average (EMA) of
       ||U_f(θ)||, used as the paper's |E[U_f(θ)]|.

       The paper formula uses |E[U_f(θ)]| (the norm of the *expected*
       free-rider update), not the single-round instance norm.  Because
       U_f is deterministic given global model history, the expectation is
       over the stochastic training process (client sampling, local SGD).
       We approximate it with an EMA of past SDFR output norms.
    """

    def __init__(self, ema_alpha: float = 0.3):
        self.val_loss_init: Optional[float] = None
        self.base_norm_ema: Optional[float] = None
        self.ema_alpha = ema_alpha

    def update(self, val_loss_t: float, base_norm_t: float):
        """Update tracked quantities after each round."""
        if self.val_loss_init is None:
            self.val_loss_init = val_loss_t
        if self.base_norm_ema is None:
            self.base_norm_ema = base_norm_t
        else:
            a = self.ema_alpha
            self.base_norm_ema = a * base_norm_t + (1 - a) * self.base_norm_ema

    def get_e_cos_beta(self, val_loss_t: float) -> float:
        """Estimate E[cos β] from current-round validation loss."""
        return estimate_e_cos_beta(val_loss_t, self.val_loss_init)

    def get_mean_base_norm(self) -> Optional[float]:
        """Return EMA of ||U_f(θ)||, or None if no history yet."""
        return self.base_norm_ema


# ── AFR attack ─────────────────────────────────────────────────────────────

def afr_attack(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
    n_total: int = 10,
    e_cos_beta: float = 0.0,
    mean_base_norm: Optional[float] = None,
    noisy_frac: float = 0.1,
    seed: Optional[int] = None,
    eps: float = 1e-10,
) -> Tuple[OrderedDict, float]:
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
        n            → n_total  (total number of clients in the FL system;
                        the paper derives the formula from the full
                        population of n honest updates, not the per-round
                        subset)
        D            → total_dim  (total model parameters)
        d            → num_noisy = int(D · noisy_frac)
        E[cos β]     → e_cos_beta  (estimated via estimate_e_cos_beta()
                        or AFRState from the validation-loss trajectory)
        |E[U_f(θ)]|  → mean_base_norm  (EMA of past SDFR output norms;
                        if None, falls back to single-round ||U_f(θ)||)
        U_f(θ)       → base_update from sdfr_attack()
        φ(t)         → phi_t

    Parameters
    ----------
    global_state_dict : current global model θ(t).
    global_history : list of past global states.
    n_total : total number of clients in the FL system.  The paper uses n
        to denote the full population size, not the per-round participants.
    e_cos_beta : estimated expected pairwise cosine similarity between
        honest updates.  Should be computed via estimate_e_cos_beta() or
        AFRState.get_e_cos_beta(), NOT kept as a fixed constant.
    mean_base_norm : EMA of ||U_f(θ)|| from past rounds, approximating
        |E[U_f(θ)]|.  If None, falls back to single-round ||U_f(θ)||.
    noisy_frac : fraction of model parameters to perturb (d / D).
    seed : optional RNG seed for reproducible coordinate selection.
    eps : numerical stability constant.

    Returns
    -------
    update : the AFR fake update (OrderedDict matching model structure).
    base_norm : ||U_f(θ)|| for this round (caller should feed this back
        into AFRState.update() for EMA tracking).
    """
    # ── Step 1: SDFR base update ────────────────────────────────────────
    base_update = sdfr_attack(global_state_dict, global_history, eps=eps)

    # ── Step 2: flatten into a single vector ────────────────────────────
    keys = list(base_update.keys())
    shapes = [base_update[k].shape for k in keys]
    flat_base = torch.cat([base_update[k].flatten() for k in keys])
    total_dim = flat_base.numel()                       # D

    # ── Step 3: compute norms ───────────────────────────────────────────
    base_norm = flat_base.float().norm().item()         # ||U_f(θ)|| this round

    # |E[U_f(θ)]|: use EMA from past rounds if available;
    # otherwise fall back to single-round ||U_f(θ)||.
    effective_base_norm = mean_base_norm if mean_base_norm is not None else base_norm

    # ── Step 4: compute noise magnitude |φ(t)| ─────────────────────────
    n = max(n_total, 2)
    denom = n + (n * n - n) * e_cos_beta
    ratio = (n * n) / (denom + eps)
    phi_t_sq = max(ratio - 1.0, 0.0) * (effective_base_norm ** 2)
    phi_t = math.sqrt(phi_t_sq)

    # ── Step 5: select d random coordinates ─────────────────────────────
    num_noisy = max(int(total_dim * noisy_frac), 1)     # d

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    else:
        gen.manual_seed(torch.randint(0, 2**31, (1,)).item())

    indices = torch.randperm(total_dim, generator=gen)[:num_noisy]

    # ── Step 6: sparse noise  z ~ N(0, φ²/d) on selected coords ───────
    noise_std = phi_t / math.sqrt(num_noisy)
    noise_values = torch.randn(num_noisy, generator=gen).to(
        dtype=flat_base.dtype, device=flat_base.device
    ) * noise_std

    flat_noise = torch.zeros_like(flat_base)
    flat_noise[indices] = noise_values

    # ── Step 7: reconstruct update dict ─────────────────────────────────
    flat_result = flat_base + flat_noise

    update = OrderedDict()
    offset = 0
    for k, shape in zip(keys, shapes):
        numel = 1
        for s in shape:
            numel *= s
        update[k] = flat_result[offset:offset + numel].reshape(shape)
        offset += numel

    return update, base_norm
