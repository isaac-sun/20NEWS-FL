"""
Label-Flipping (LF) targeted poisoning attack.

A malicious client changes labels in its local training data from a
chosen source class to a chosen target class before local training.
Features are NOT changed.  The client then trains normally on the
poisoned dataset and submits the resulting update.

This is a targeted attack: the attacker wants the global model to
misclassify samples of the source class as the target class.

Default mappings:
    - MNIST:       source=1 → target=7
    - CIFAR-10:    source=1 → target=9
    - 20Newsgroups: source=0 (alt.atheism) → target=10 (sci.crypt)
      (chosen as two semantically distant classes for maximum effect)

Evaluation metrics:
    - ASR: fraction of source-class test samples predicted as target class
    - TACC: accuracy on source-class test samples (lower = more effective)

Reference: SVRFL paper, targeted data-poisoning attack.
"""

import copy

import torch
from torch.utils.data import TensorDataset, DataLoader


def create_label_flipped_dataset(
    dataset,
    source_class: int,
    target_class: int,
) -> TensorDataset:
    """
    Create a copy of the dataset with source_class labels changed to
    target_class.  Features are untouched.

    Parameters
    ----------
    dataset : a TensorDataset with (features, labels).
    source_class : original label to flip.
    target_class : new label to assign.

    Returns
    -------
    A new TensorDataset with flipped labels.
    """
    X = dataset.tensors[0]  # features tensor (or via .data for Subset)
    y = dataset.tensors[1].clone()  # labels tensor, cloned to avoid mutation

    mask = y == source_class
    y[mask] = target_class

    return TensorDataset(X, y)


def create_label_flipped_subset(
    original_dataset,
    indices,
    source_class: int,
    target_class: int,
):
    """
    Create a label-flipped version of a Subset of a TensorDataset.

    Parameters
    ----------
    original_dataset : the full TensorDataset.
    indices : list of indices (the client's partition).
    source_class : label to flip from.
    target_class : label to flip to.

    Returns
    -------
    A TensorDataset containing only the specified indices, with
    source_class labels flipped to target_class.
    """
    X_full = original_dataset.tensors[0]
    y_full = original_dataset.tensors[1]

    X_sub = X_full[indices]
    y_sub = y_full[indices].clone()

    mask = y_sub == source_class
    n_source = int(mask.sum().item())
    y_sub[mask] = target_class

    return TensorDataset(X_sub, y_sub), n_source


@torch.no_grad()
def evaluate_targeted_attack(model, test_loader, source_class: int,
                             target_class: int, device: str = "cpu"):
    """
    Evaluate targeted attack metrics.

    Returns
    -------
    asr : Attack Success Rate — fraction of source-class samples
        predicted as target class.
    tacc : accuracy on source-class samples (lower = attack more effective).
    source_total : number of source-class test samples (the denominator).
    """
    model.eval()
    source_total = 0
    source_correct = 0
    source_predicted_target = 0

    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        preds = model(X).argmax(dim=1)

        source_mask = y == source_class
        if source_mask.any():
            source_total += source_mask.sum().item()
            source_correct += (preds[source_mask] == y[source_mask]).sum().item()
            source_predicted_target += (preds[source_mask] == target_class).sum().item()

    if source_total == 0:
        return 0.0, 0.0, 0

    asr = source_predicted_target / source_total
    tacc = source_correct / source_total
    return asr, tacc, source_total
