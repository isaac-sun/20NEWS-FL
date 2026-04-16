"""
Round-level Shapley value estimation using Monte Carlo permutation sampling.

Supports both overall and **per-class** Shapley values.

Per-class value function for class c:
  v_c(S) = F_c(w_g, D_v) - F_c(w_S, D_v)
where F_c is cross-entropy loss averaged over validation samples of class c.

Overall Shapley = weighted average of per-class Shapley (by class frequency).
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def _build_coalition_params(
    global_state_dict: OrderedDict,
    updates: dict,
    coalition: list,
    server_lr: float,
) -> OrderedDict:
    """Build model params for coalition S:
    w_S = w_g + (server_lr / |S|) * sum_{i in S} delta_i
    """
    if len(coalition) == 0:
        return OrderedDict({k: v.clone() for k, v in global_state_dict.items()})

    n = len(coalition)
    new_state = OrderedDict()
    for key in global_state_dict:
        agg = torch.zeros_like(global_state_dict[key])
        for cid in coalition:
            agg = agg + updates[cid][key]
        new_state[key] = global_state_dict[key] + (server_lr / n) * agg
    return new_state


@torch.no_grad()
def _evaluate_per_class_loss(
    model, state_dict: OrderedDict, data_loader, num_classes: int, device="cpu"
) -> np.ndarray:
    """Evaluate per-class average cross-entropy loss.
    Returns ndarray of shape (num_classes,).
    """
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    class_loss_sum = np.zeros(num_classes, dtype=np.float64)
    class_count = np.zeros(num_classes, dtype=np.int64)

    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        per_sample_loss = criterion(model(X), y)
        for c in range(num_classes):
            mask = y == c
            if mask.any():
                class_loss_sum[c] += per_sample_loss[mask].sum().item()
                class_count[c] += mask.sum().item()

    safe_count = np.maximum(class_count, 1)
    return class_loss_sum / safe_count


def _class_weights_from_loader(val_loader, num_classes: int) -> np.ndarray:
    """Compute class frequency weights from a DataLoader."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in val_loader:
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
    server_lr: float = 1.0,
    num_mc_samples: int = 30,
    device: str = "cpu",
) -> dict:
    """
    Estimate per-class Shapley values for one FL round using
    Monte Carlo permutation sampling.

    Same number of forward passes as the overall version, but returns
    per-class granularity.

    Returns dict of client_id -> np.ndarray of shape (num_classes,).
    """
    client_ids = list(updates.keys())
    if len(client_ids) == 0:
        return {}

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
                global_state_dict, updates, coalition, server_lr
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


def compute_class_metrics(per_class_sv: dict) -> dict:
    """Compute the two per-class detection metrics for each client.

    Returns dict of client_id -> {
        'class_sv_variance': variance of per-class SV,
        'positive_class_sv_sum': sum of positive per-class SVs,
    }
    """
    metrics = {}
    for cid, sv_arr in per_class_sv.items():
        metrics[cid] = {
            "class_sv_variance": float(np.var(sv_arr)),
            "positive_class_sv_sum": float(np.sum(sv_arr[sv_arr > 0])),
        }
    return metrics


# Backward-compatible wrapper
def estimate_round_shapley(
    model,
    updates: dict,
    global_state_dict: OrderedDict,
    val_loader,
    server_lr: float = 1.0,
    num_mc_samples: int = 30,
    device: str = "cpu",
) -> dict:
    """Overall Shapley values (backward compatible).
    Returns dict of client_id -> float.
    """
    pc_sv = estimate_round_shapley_per_class(
        model, updates, global_state_dict, val_loader,
        num_classes=20, server_lr=server_lr,
        num_mc_samples=num_mc_samples, device=device,
    )
    weights = _class_weights_from_loader(val_loader, 20)
    return per_class_to_overall(pc_sv, weights)
