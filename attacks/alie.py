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
    p = clip(p, 1e-6, 1 - 1e-6)
    z_max = Φ^{-1}(p)

Two modes are supported:

1. "svrfl" (default): mu/sigma estimated from benign updates only.
2. "original": mu/sigma estimated from malicious clients' own clean
   updates only (as in the original ALIE paper).

Reference: SVRFL paper + "A Little Is Enough" (Baruch et al.).
"""

import math
from collections import OrderedDict
from typing import Dict, List

import torch
import numpy as np
from scipy.stats import norm


def _compute_z_max(n: int, m: int) -> float:
    """
    Compute z_max from the ALIE formula.

    Parameters
    ----------
    n : total number of participating clients this round.
    m : number of malicious participating clients this round.

    Returns
    -------
    z_max : float
    """
    s = math.floor(n / 2 + 1) - m
    denom = n - m
    if denom <= 0:
        return 0.0
    p = (denom - s) / denom
    # Clip for numerical stability
    p = max(1e-6, min(p, 1.0 - 1e-6))
    z_max = norm.ppf(p)
    return z_max


def _flatten_updates(updates: Dict[int, OrderedDict]) -> torch.Tensor:
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
) -> OrderedDict:
    """
    Compute the ALIE malicious update.

    Parameters
    ----------
    benign_updates : dict of honest client_id -> update OrderedDict.
    malicious_clean_updates : dict of malicious client_id -> their
        honest (clean) update OrderedDict.  Used only in "original" mode.
    n_participants : total number of selected clients this round (n).
    n_malicious : number of malicious clients among selected (m).
    mode : "svrfl" (default) — use benign_updates for mu/sigma.
           "original" — use malicious_clean_updates for mu/sigma.

    Returns
    -------
    The single malicious update that all malicious clients submit.
    """
    # Choose reference set based on mode
    if mode == "original":
        reference_updates = malicious_clean_updates
    else:
        # Default: svrfl_reproduction_mode
        reference_updates = benign_updates

    if len(reference_updates) == 0:
        # Fallback: if no reference updates available, use any available
        reference_updates = benign_updates if benign_updates else malicious_clean_updates

    # Get a reference OrderedDict for shapes
    any_cid = next(iter(reference_updates))
    ref_od = reference_updates[any_cid]

    # Flatten reference updates into matrix
    stacked, keys = _flatten_updates(reference_updates)

    # Compute coordinate-wise mean and std
    mu = stacked.mean(dim=0)
    sigma = stacked.std(dim=0)

    # Compute z_max
    z_max = _compute_z_max(n_participants, n_malicious)

    # Craft malicious update: p_mal_j = mu_j - z_max * sigma_j
    p_mal_flat = mu - z_max * sigma

    # Reconstruct as OrderedDict
    return _unflatten_update(p_mal_flat, keys, ref_od)
