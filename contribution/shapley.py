"""
Round-level Shapley value estimation using Monte Carlo permutation sampling
following the SVRFL paper approach.

Value function:  v(S) = F(w_g, D_v) - F(w_S, D_v)
  where F is cross-entropy loss on the server validation set,
  w_S = w_g + (server_lr / |S|) * sum_{i in S} delta_i

Marginal contribution of client i added to coalition S:
  v(S ∪ {i}) - v(S) = F(w_S) - F(w_{S∪{i}})  (loss decrease)
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
def _evaluate_loss(model, state_dict: OrderedDict, data_loader, device="cpu") -> float:
    """Evaluate cross-entropy loss of model with given state_dict."""
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        out = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * X.size(0)
        total += X.size(0)
    return total_loss / max(total, 1)


def estimate_round_shapley(
    model,
    updates: dict,
    global_state_dict: OrderedDict,
    val_loader,
    server_lr: float = 1.0,
    num_mc_samples: int = 30,
    device: str = "cpu",
) -> dict:
    """
    Estimate per-client Shapley values for one FL round using
    Monte Carlo permutation sampling (SVRFL SVEstimate).

    For each sampled permutation, walk through clients in order and
    record each client's marginal contribution (= loss decrease when
    that client is added to the growing coalition).

    Returns dict of client_id -> shapley_value (float).
    """
    client_ids = list(updates.keys())
    n = len(client_ids)
    if n == 0:
        return {}

    shapley_sums = {cid: 0.0 for cid in client_ids}

    # Loss of the global model (empty-coalition baseline)
    base_loss = _evaluate_loss(model, global_state_dict, val_loader, device)

    for _ in range(num_mc_samples):
        perm = np.random.permutation(client_ids).tolist()
        coalition = []
        prev_loss = base_loss

        for cid in perm:
            coalition.append(cid)
            coal_params = _build_coalition_params(
                global_state_dict, updates, coalition, server_lr
            )
            curr_loss = _evaluate_loss(model, coal_params, val_loader, device)
            # Marginal contribution = loss decrease from adding this client
            shapley_sums[cid] += (prev_loss - curr_loss)
            prev_loss = curr_loss

    shapley_values = {cid: shapley_sums[cid] / num_mc_samples for cid in client_ids}
    return shapley_values
