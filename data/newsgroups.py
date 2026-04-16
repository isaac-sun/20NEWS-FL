import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer


def load_newsgroups(max_features: int = 10000, val_ratio: float = 0.1):
    """
    Load 20 Newsgroups, apply TF-IDF, split into train/val/test.
    Returns (train_dataset, val_dataset, test_dataset, input_dim, train_labels).
    """
    train_raw = fetch_20newsgroups(
        subset="train", remove=("headers", "footers", "quotes")
    )
    test_raw = fetch_20newsgroups(
        subset="test", remove=("headers", "footers", "quotes")
    )

    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_all = vectorizer.fit_transform(train_raw.data).toarray().astype(np.float32)
    X_test = vectorizer.transform(test_raw.data).toarray().astype(np.float32)
    y_train_all = np.array(train_raw.target)
    y_test = np.array(test_raw.target)

    # Split train into train + validation
    n = len(y_train_all)
    indices = np.random.permutation(n)
    val_size = int(n * val_ratio)
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    X_train = X_train_all[train_idx]
    y_train = y_train_all[train_idx]
    X_val = X_train_all[val_idx]
    y_val = y_train_all[val_idx]

    train_dataset = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val).long()
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test).long()
    )

    input_dim = X_train.shape[1]
    return train_dataset, val_dataset, test_dataset, input_dim, y_train
