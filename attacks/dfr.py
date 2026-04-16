import torch
from collections import OrderedDict


def dfr_attack(
    global_state_dict: OrderedDict, noise_scale: float = 0.001
) -> OrderedDict:
    """
    Disguised Free-Rider (DFR): send small random noise as model update.
    The free-rider does not train locally.
    """
    update = OrderedDict()
    for key in global_state_dict:
        update[key] = torch.randn_like(global_state_dict[key]) * noise_scale
    return update
