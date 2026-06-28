"""
Round-level Shapley value estimation using Monte Carlo permutation sampling.

Supports both overall and **per-class** Shapley values.

Per-class value function for class c:
  v_c(S) = F_c(w_g, D_v) - F_c(w_S, D_v)
where F_c is cross-entropy loss averaged over validation samples of class c.

Overall Shapley = weighted average of per-class Shapley (by class frequency).

── Coalition Model (Friend's Standard) ──────────────────────────────

The coalition model is a **sample-size weighted average** of the full
local model parameters (not a FedAvg-style delta aggregation):

    w_S = Σ_{j∈S} n_j · w_j  /  Σ_{j∈S} n_j

where:
  - n_j  = number of local training samples at client j
  - w_j  = full local model parameters after local training
          = w_g + delta_j   (global params + uploaded update)

This is equivalent to:

    w_S = w_g + Σ_{j∈S} n_j · delta_j  /  Σ_{j∈S} n_j

which is a weighted delta aggregation with sample-size weights
(no server_lr scaling — it's a direct weighted average of local models).
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def _build_coalition_params(
    global_state_dict: OrderedDict,
    updates: dict,
    coalition: list,
    sample_counts: dict,
) -> OrderedDict:
    """Build model params for coalition S using sample-size weighted averaging.

    Friend's formula:
        w_S = Σ_{j∈S} n_j · w_j  /  Σ_{j∈S} n_j

    All full-model parameters are communicated in this experiment.

    Parameters
    ----------
    global_state_dict : current global trainable params w_g.
    updates : dict of client_id -> delta_i (trainable params only).
    coalition : list of client_ids in the coalition S.
    sample_counts : dict of client_id -> n_j.
    """
    if len(coalition) == 0:
        return OrderedDict({k: v.clone() for k, v in global_state_dict.items()})

    total_n = sum(sample_counts.get(cid, 1) for cid in coalition)
    if total_n == 0:
        total_n = 1

    new_state = OrderedDict()

    for key, global_value in global_state_dict.items():
        # Weighted sum of deltas: Σ n_j · delta_j
        agg = torch.zeros_like(global_value)
        for cid in coalition:
            n_j = sample_counts.get(cid, 1)
            agg.add_(updates[cid][key].to(
                device=global_value.device, dtype=global_value.dtype
            ), alpha=n_j)
        # w_S = w_g + Σ n_j·delta_j / Σ n_j
        new_state[key] = global_value + agg / total_n
        if not torch.isfinite(new_state[key]).all():
            raise FloatingPointError(f"non-finite coalition parameter: {key}")
    return new_state


@torch.no_grad()
def _evaluate_per_class_loss(
    model, state_dict: OrderedDict, data_loader, num_classes: int, device="cpu"
) -> np.ndarray:
    """Evaluate per-class average cross-entropy loss.

    Loads a full state dict into the model.
    Handles both 3-tuple (input_ids, attention_mask, labels) and
    2-tuple (features, labels) data loader formats.

    Returns ndarray of shape (num_classes,).
    """
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    class_loss_sum = np.zeros(num_classes, dtype=np.float64)
    class_count = np.zeros(num_classes, dtype=np.int64)

    for batch in data_loader:
        if len(batch) == 3:
            input_ids, attn_mask, y = [b.to(device) for b in batch]
            per_sample_loss = criterion(model(input_ids, attention_mask=attn_mask), y)
        else:
            X, y = batch[0].to(device), batch[1].to(device)
            per_sample_loss = criterion(model(X), y)
        if not torch.isfinite(per_sample_loss).all():
            raise FloatingPointError("non-finite loss during Shapley evaluation")
        for c in range(num_classes):
            mask = y == c
            if mask.any():
                class_loss_sum[c] += per_sample_loss[mask].sum().item()
                class_count[c] += mask.sum().item()

    safe_count = np.maximum(class_count, 1)
    return class_loss_sum / safe_count


def _class_weights_from_loader(val_loader, num_classes: int) -> np.ndarray:
    """Compute class frequency weights from a DataLoader.

    Handles both 3-tuple and 2-tuple formats.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for batch in val_loader:
        y = batch[-1]  # labels are always last
        for c in range(num_classes):
            counts[c] += (y == c).sum().item()
    total = counts.sum()
    return counts / max(total, 1)


def estimate_round_shapley_per_class(
    model,
    updates: dict,
    global_state_dict: OrderedDict,
    val_loader,
    num_classes: int = 20,
    num_mc_samples: int = 30,
    device: str = "cpu",
    sample_counts: dict | None = None,
) -> dict:
    """
    Estimate per-class Shapley values for one FL round using
    Monte Carlo permutation sampling.

    Same number of forward passes as the overall version, but returns
    per-class granularity.

    The coalition model uses **sample-size weighted averaging** of full
    local parameters (aligned with friend's standard):
        w_S = Σ_{j∈S} n_j · w_j  /  Σ_{j∈S} n_j

    Parameters
    ----------
    sample_counts : dict of client_id -> n_j (number of local samples).
        If None, defaults to uniform weight=1 for all clients.
        Must contain entries for ALL clients in `updates`.

    Returns dict of client_id -> np.ndarray of shape (num_classes,).
    """
    client_ids = list(updates.keys())
    if len(client_ids) == 0:
        return {}

    # Default: uniform sample counts if not provided
    if sample_counts is None:
        sample_counts = {cid: 1 for cid in client_ids}

    shapley_sums = {cid: np.zeros(num_classes, dtype=np.float64) for cid in client_ids}

    base_pc = _evaluate_per_class_loss(
        model, global_state_dict, val_loader, num_classes, device
    )

    for _ in range(num_mc_samples):
        perm = np.random.permutation(client_ids).tolist()
        coalition = []
        prev_pc = base_pc.copy()

        for cid in perm:
            coalition.append(cid)
            coal_params = _build_coalition_params(
                global_state_dict, updates, coalition, sample_counts
            )
            curr_pc = _evaluate_per_class_loss(
                model, coal_params, val_loader, num_classes, device
            )
            shapley_sums[cid] += (prev_pc - curr_pc)
            prev_pc = curr_pc

    return {cid: shapley_sums[cid] / num_mc_samples for cid in client_ids}


def per_class_to_overall(per_class_sv: dict, class_weights: np.ndarray) -> dict:
    """Derive overall Shapley from per-class Shapley via weighted sum."""
    return {cid: float(np.dot(class_weights, sv)) for cid, sv in per_class_sv.items()}


def compute_class_metrics(per_class_sv: dict, eps: float = 1e-10) -> dict:
    """Compute the four per-class Shapley detection metrics for each client.

    Aligned with friend's classwise_metrics.py:
      A_i = Σ_c max(SV_i,c, 0)     → positive_class_sv_sum
      P_i = max_c max(SV_i,c,0) / (A_i + ε) → concentration_ratio
      V_i = Var_c(SV_i,c)          → class_sv_variance
      C_i = argmax_c max(SV_i,c,0) → dominant_class

    Returns dict of client_id -> {
        'class_sv_variance':    float,
        'positive_class_sv_sum': float,
        'concentration_ratio':  float,
        'dominant_class':       int,
    }
    """
    metrics = {}
    for cid, sv_arr in per_class_sv.items():
        positive = np.maximum(sv_arr, 0.0)   # only positive contribution
        a_i = float(np.sum(positive))         # A_i^t: total positive contribution
        p_i = float(np.max(positive) / (a_i + eps)) if a_i > 0 else 0.0  # P_i^t
        v_i = float(np.var(sv_arr))           # V_i^t: inter-class variance
        c_i = int(np.argmax(positive))        # C_i^t: dominant class

        metrics[cid] = {
            "class_sv_variance": v_i,
            "positive_class_sv_sum": a_i,
            "concentration_ratio": p_i,
            "dominant_class": c_i,
        }
    return metrics


# Backward-compatible wrapper
def estimate_round_shapley(
    model,
    updates: dict,
    global_state_dict: OrderedDict,
    val_loader,
    num_mc_samples: int = 30,
    device: str = "cpu",
    sample_counts: dict | None = None,
) -> dict:
    """Overall Shapley values (backward compatible).
    Returns dict of client_id -> float.
    """
    pc_sv = estimate_round_shapley_per_class(
        model, updates, global_state_dict, val_loader,
        num_classes=20,
        num_mc_samples=num_mc_samples, device=device,
        sample_counts=sample_counts,
    )
    weights = _class_weights_from_loader(val_loader, 20)
    return per_class_to_overall(pc_sv, weights)
