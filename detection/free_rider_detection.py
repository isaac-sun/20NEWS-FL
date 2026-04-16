"""
SVRFL-style feature-value-based free-rider detection.

Feature value:  d_i = |sv_i| / (L_cosine_i ^ 2)
where L_cosine_i = 1 - cosine(local_model_i, global_model_t)

Detection: cluster feature values into 2 groups using K-means.
If larger centroid > h * smaller centroid, flag the larger-centroid
cluster as suspected free riders.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def _flatten(state_dict_or_update: dict) -> torch.Tensor:
    """Flatten an OrderedDict of tensors into a single 1-D tensor."""
    return torch.cat([v.flatten().float() for v in state_dict_or_update.values()])


def compute_feature_values(
    updates: dict,
    shapley_values: dict,
    global_state_dict: dict,
) -> dict:
    """
    Compute SVRFL-style feature values for free-rider detection.
    d_i = |sv_i| / (L_cosine_i ^ 2)
    L_cosine_i = 1 - cosine(local_model_i, global_model_t)
    where local_model_i = global_model + update_i.
    """
    global_flat = _flatten(global_state_dict)

    feature_values = {}
    for cid, update in updates.items():
        update_flat = _flatten(update)
        local_flat = global_flat + update_flat

        cos_sim = F.cosine_similarity(
            local_flat.unsqueeze(0), global_flat.unsqueeze(0)
        ).item()

        L_cosine = max(1.0 - cos_sim, 1e-10)  # avoid division by zero
        sv = abs(shapley_values.get(cid, 0.0))
        feature_values[cid] = sv / (L_cosine ** 2)

    return feature_values


def detect_free_riders(feature_values: dict, threshold_h: float = 200.0) -> dict:
    """
    Cluster feature values into 2 groups with K-means.
    If larger centroid > h * smaller centroid, flag clients in
    the larger-centroid cluster as suspected free riders.
    Returns dict of client_id -> bool (True = suspected).

    Uses log1p transform before clustering to handle the extreme
    dynamic range of feature values (d_i can span many orders of
    magnitude when L_cosine is near zero).
    """
    if len(feature_values) < 2:
        return {cid: False for cid in feature_values}

    cids = list(feature_values.keys())
    raw = np.array([feature_values[cid] for cid in cids])

    # Log-transform to tame extreme range before K-means
    values = np.log1p(np.abs(raw)).reshape(-1, 1)

    # Edge case: all values identical
    if np.std(values) < 1e-12:
        return {cid: False for cid in cids}

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(values)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_.flatten()

    larger_idx = int(np.argmax(centroids))
    smaller_idx = 1 - larger_idx

    suspected = {}
    # Compare in original (exp) scale for the threshold check
    orig_centroids = np.expm1(centroids)
    if (
        orig_centroids[smaller_idx] > 1e-12
        and orig_centroids[larger_idx] > threshold_h * orig_centroids[smaller_idx]
    ):
        for i, cid in enumerate(cids):
            suspected[cid] = bool(labels[i] == larger_idx)
    else:
        suspected = {cid: False for cid in cids}

    return suspected
