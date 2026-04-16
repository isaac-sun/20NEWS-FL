from collections import OrderedDict


def fedavg_aggregate(
    global_state_dict: OrderedDict,
    updates: dict,
    server_lr: float = 1.0,
    weights: dict = None,
) -> OrderedDict:
    """
    FedAvg aggregation: w_new = w_g + server_lr * weighted_avg(updates).
    updates: dict of client_id -> OrderedDict of param deltas.
    weights: optional dict of client_id -> weight (defaults to uniform 1/n).
    """
    client_ids = list(updates.keys())
    n = len(client_ids)

    if n == 0:
        return OrderedDict({k: v.clone() for k, v in global_state_dict.items()})

    if weights is None:
        weights = {cid: 1.0 / n for cid in client_ids}

    new_state_dict = OrderedDict()
    for key in global_state_dict:
        device = global_state_dict[key].device
        agg_update = sum(
            weights[cid] * updates[cid][key].to(device) for cid in client_ids
        )
        new_state_dict[key] = global_state_dict[key] + server_lr * agg_update

    return new_state_dict
