"""
A Little Is Enough (ALIE) poisoning attack.

ALIE is an untargeted poisoning attack where all malicious participants
in a round submit the same crafted update, built coordinate-wise as:

    p_mal_j = mu_j - z_max * sigma_j

where mu_j and sigma_j are the coordinate-wise mean and standard
deviation of a reference set of updates, and z_max is computed from
the standard normal inverse CDF based on round participation counts.

z_max computation:
    n = number of participating clients this round
    m = number of malicious participating clients this round
    s = floor(n/2 + 1) - m
    p = (n - m - s) / (n - m)
    p = clip(p, clip_eps, 1 - clip_eps)
    z_max = Φ^{-1}(p)

Two modes are supported:

1. "svrfl" (default): mu/sigma estimated from benign updates only.
2. "original": mu/sigma estimated from malicious clients' own clean
   updates only (as in the original ALIE paper).

Reference: SVRFL paper + "A Little Is Enough" (Baruch et al.).
"""

import math
import logging
from collections import OrderedDict
from typing import Dict, Tuple

import torch
import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


def _compute_z_max(
    n: int,
    m: int,
    clip_eps: float = 1e-6,
) -> Tuple[float, int, float]:
    """
    Compute z_max from the ALIE formula.

    Returns
    -------
    z_max, s, p
    """
    s = math.floor(n / 2 + 1) - m
    denom = n - m
    if denom <= 0:
        return 0.0, s, 0.5
    p = (denom - s) / denom
    p = max(clip_eps, min(p, 1.0 - clip_eps))
    z_max = float(norm.ppf(p))
    return z_max, s, p


def _flatten_updates(updates: Dict[int, OrderedDict]) -> Tuple[torch.Tensor, list]:
    """Stack updates into a 2D tensor: (num_clients, total_params)."""
    keys = None
    rows = []
    for cid in sorted(updates.keys()):
        u = updates[cid]
        if keys is None:
            keys = list(u.keys())
        flat = torch.cat([u[k].flatten().float() for k in keys])
        rows.append(flat)
    return torch.stack(rows, dim=0), keys


def _unflatten_update(flat: torch.Tensor, keys: list,
                      reference: OrderedDict) -> OrderedDict:
    """Reconstruct an OrderedDict from a flat tensor."""
    result = OrderedDict()
    offset = 0
    for k in keys:
        shape = reference[k].shape
        numel = reference[k].numel()
        result[k] = flat[offset:offset + numel].reshape(shape).to(
            dtype=reference[k].dtype, device=reference[k].device
        )
        offset += numel
    return result


def alie_attack(
    benign_updates: Dict[int, OrderedDict],
    malicious_clean_updates: Dict[int, OrderedDict],
    n_participants: int,
    n_malicious: int,
    mode: str = "svrfl",
    clip_prob_eps: float = 1e-6,
) -> Tuple[OrderedDict, dict]:
    """
    Compute the ALIE malicious update with full debug instrumentation.

    Parameters
    ----------
    benign_updates : dict of honest client_id -> update OrderedDict.
    malicious_clean_updates : dict of malicious client_id -> their
        honest (clean) update OrderedDict.
    n_participants : total number of selected clients this round (n).
    n_malicious : number of malicious clients among selected (m).
    mode : "svrfl" (default) — use benign_updates for mu/sigma.
           "original" — use malicious_clean_updates for mu/sigma.
    clip_prob_eps : clipping epsilon for p in z_max computation.

    Returns
    -------
    (p_mal_od, debug_info) where:
        p_mal_od : the single malicious update that all malicious clients submit.
        debug_info : dict with diagnostic metrics for this round.
    """
    debug = {
        "n_participants": n_participants,
        "n_malicious": n_malicious,
    }

    # Choose reference set based on mode
    if mode == "original":
        reference_updates = malicious_clean_updates
    else:
        reference_updates = benign_updates

    if len(reference_updates) == 0:
        reference_updates = benign_updates if benign_updates else malicious_clean_updates

    n_ref = len(reference_updates)
    debug["n_reference"] = n_ref

    # Get a reference OrderedDict for shapes
    any_cid = next(iter(reference_updates))
    ref_od = reference_updates[any_cid]

    # Flatten reference updates into matrix
    stacked, keys = _flatten_updates(reference_updates)

    # Compute coordinate-wise mean
    mu = stacked.mean(dim=0)

    # Compute coordinate-wise std with numerical safety
    # Use population std (correction=0) for stability with small reference sets.
    # With n_ref=1, std is all zeros — p_mal collapses to mu (safe but degenerate).
    if n_ref < 2:
        logger.warning(
            f"ALIE: only {n_ref} reference update(s); std is zero, "
            "p_mal will equal mu (attack is ineffective this round)"
        )
        sigma = torch.zeros_like(mu)
    else:
        sigma = stacked.std(dim=0, correction=0)

    # Clamp very small sigma to avoid numerical issues — but keep zeros
    # from genuinely identical coordinates as-is (those won't shift).
    # This prevents rare NaN from division, not from the craft formula itself.

    # Compute z_max
    z_max, s, p = _compute_z_max(n_participants, n_malicious, clip_eps=clip_prob_eps)

    if abs(z_max) < 0.1:
        logger.warning(
            f"ALIE: z_max={z_max:.4f} is very small (n={n_participants}, "
            f"m={n_malicious}, s={s}, p={p:.4f}). "
            "Attack may be too subtle to have meaningful impact."
        )

    debug["s"] = s
    debug["p"] = p
    debug["z_max"] = z_max

    # Craft malicious update: p_mal_j = mu_j - z_max * sigma_j
    p_mal_flat = mu - z_max * sigma

    # Compute debug norms
    mu_norm = float(torch.norm(mu).item())
    sigma_norm = float(torch.norm(sigma).item())
    p_mal_norm = float(torch.norm(p_mal_flat).item())
    diff = p_mal_flat - mu
    diff_norm = float(torch.norm(diff).item())

    # Cosine similarity between p_mal and mu
    if mu_norm > 0 and p_mal_norm > 0:
        cosine_pmal_mu = float(
            torch.dot(p_mal_flat, mu) / (p_mal_norm * mu_norm)
        )
    else:
        cosine_pmal_mu = 1.0  # degenerate case: both zero or one zero

    debug["mu_norm"] = mu_norm
    debug["sigma_norm"] = sigma_norm
    debug["p_mal_norm"] = p_mal_norm
    debug["diff_from_mu_norm"] = diff_norm
    debug["cosine_pmal_mu"] = cosine_pmal_mu

    # Log summary
    logger.info(
        f"ALIE round: n={n_participants}, m={n_malicious}, s={s}, "
        f"p={p:.4f}, z_max={z_max:.4f} | "
        f"||mu||={mu_norm:.4f}, ||sigma||={sigma_norm:.4f}, "
        f"||p_mal||={p_mal_norm:.4f}, ||p_mal-mu||={diff_norm:.4f}, "
        f"cos(p_mal,mu)={cosine_pmal_mu:.6f}"
    )

    # Reconstruct as OrderedDict
    p_mal_od = _unflatten_update(p_mal_flat, keys, ref_od)
    return p_mal_od, debug
