import torch
from collections import OrderedDict
from typing import Optional


def sdfr_attack(
    global_state_dict: OrderedDict,
    prev_global_state_dict: Optional[OrderedDict],
    scale: float = 0.5,
) -> OrderedDict:
    """
    Scaled Delta Free-Rider (SDFR): send a scaled version of the
    previous round's global model change as the model update.
    """
    update = OrderedDict()
    if prev_global_state_dict is None:
        # First round fallback: small random noise
        for key in global_state_dict:
            update[key] = torch.randn_like(global_state_dict[key]) * 0.001
    else:
        for key in global_state_dict:
            delta = global_state_dict[key] - prev_global_state_dict[key]
            update[key] = delta * scale
    return update
