import torch
from collections import OrderedDict
from typing import Optional


def afr_attack(
    global_state_dict: OrderedDict,
    prev_global_state_dict: Optional[OrderedDict],
    noise_scale: float = 0.0005,
) -> OrderedDict:
    """
    Advanced Free-Rider (AFR): send the previous round's global model
    change plus calibrated random noise as the model update.
    """
    update = OrderedDict()
    if prev_global_state_dict is None:
        for key in global_state_dict:
            update[key] = torch.randn_like(global_state_dict[key]) * 0.001
    else:
        for key in global_state_dict:
            delta = global_state_dict[key] - prev_global_state_dict[key]
            noise = torch.randn_like(delta) * noise_scale
            update[key] = delta + noise
    return update
