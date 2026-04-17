"""
Sign-Flipping (SF) poisoning attack.

A malicious client performs normal local training to compute the true
local update g_i, then submits:

    g_i_poisoned = u * g_i

where u is a negative constant (default u = -1).

This reverses the direction of the honest gradient, pushing the global
model away from the correct optimum.

Reference: SVRFL paper, untargeted model-poisoning attack.
"""

from collections import OrderedDict


def sign_flip_attack(
    honest_update: OrderedDict,
    u: float = -1.0,
) -> OrderedDict:
    """
    Apply sign-flipping to an honest local update.

    Parameters
    ----------
    honest_update : the true local update g_i = local_params - global_params,
        computed by honest local training.
    u : scaling constant (must be negative for poisoning effect).
        Default u = -1 (negate the update).
        Optional stronger mode: u = -4.

    Returns
    -------
    Poisoned update: u * g_i
    """
    poisoned = OrderedDict()
    for key in honest_update:
        poisoned[key] = u * honest_update[key]
    return poisoned
