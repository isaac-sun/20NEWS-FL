from __future__ import annotations

from collections import OrderedDict


def fedavg_aggregate(
    global_state_dict: OrderedDict,
    updates: dict,
    server_lr: float = 1.0,
    weights: dict = None,
    momentum: float = 0.0,
    momentum_buffer: OrderedDict = None,
) -> tuple[OrderedDict, OrderedDict | None]:
    """
    FedAvg aggregation with optional server momentum.

    Standard FedAvg:
        w_new = w_g + server_lr * weighted_avg(updates)

    With momentum:
        v_t = momentum * v_{t-1} + weighted_avg(updates)
        w_new = w_g + server_lr * v_t

    Args:
        global_state_dict: current global model params.
        updates: dict of client_id -> OrderedDict of param deltas.
        server_lr: learning rate for the server update.
        weights: optional dict of client_id -> weight (defaults to uniform 1/n).
        momentum: momentum coefficient (0.0 = no momentum).
        momentum_buffer: previous momentum values (None on first call).

    Returns:
        (new_state_dict, new_momentum_buffer)
    """
    client_ids = list(updates.keys())
    n = len(client_ids)

    if n == 0:
        return (
            OrderedDict({k: v.clone() for k, v in global_state_dict.items()}),
            momentum_buffer,
        )

    if weights is None:
        weights = {cid: 1.0 / n for cid in client_ids}

    new_state_dict = OrderedDict()
    new_momentum = OrderedDict() if momentum > 0 else None

    for key in global_state_dict:
        device = global_state_dict[key].device
        dtype = global_state_dict[key].dtype

        # Weighted average of client deltas
        avg_delta = sum(
            weights[cid] * updates[cid][key].to(device) for cid in client_ids
        )

        if momentum > 0 and momentum_buffer is not None and key in momentum_buffer:
            # v_t = β * v_{t-1} + avg_delta
            avg_delta = (
                momentum * momentum_buffer[key].to(device) + avg_delta
            )

        if new_momentum is not None:
            new_momentum[key] = avg_delta.clone()

        new_state_dict[key] = global_state_dict[key] + server_lr * avg_delta

    return new_state_dict, new_momentum
