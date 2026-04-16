import torch
from collections import OrderedDict
from typing import List, Optional


def sdfr_attack(
    global_state_dict: OrderedDict,
    global_history: List[OrderedDict],
) -> OrderedDict:
    """
    Scaled Delta Free-Rider (SDFR): reconstruct the expected per-round
    global update from a 3-round sliding window of global model states.

    Paper: construct directly from the proportional difference of the last
    3 global models; no extra hand-tuned scale parameter.

    When ≥3 rounds of history are available:
        delta = (w_t - w_{t-3}) / 3   (average per-round global change)
    When only 1–2 rounds are available:
        delta = w_t - w_{t-k}  (k = available history length, normalised)
    Round 0 fallback: small random noise.
    """
    update = OrderedDict()
    k = len(global_history)  # number of stored past states (oldest → newest)

    if k == 0:
        # Round 0: no history yet
        for key in global_state_dict:
            update[key] = torch.randn_like(global_state_dict[key]) * 0.001
    else:
        # Use oldest available history point; normalise by the span
        oldest = global_history[0]  # up to 3 rounds ago
        span = min(k, 3)
        for key in global_state_dict:
            delta = (global_state_dict[key] - oldest[key]) / span
            update[key] = delta
    return update
