"""
BERT-embedding classifier for 20 Newsgroups federated learning.

Takes pre-computed 768-dim BERT embeddings as input and passes them
through an MLP classifier head.  The BERT encoder itself is NOT part
of this model — embeddings are pre-computed once in data/newsgroups.py
using a frozen bert-base-uncased, so clients only train the lightweight
classifier.

Architecture:
    input (768) → Linear(768, hidden) → ReLU → Dropout → Linear(hidden, 20)
"""

import torch.nn as nn


class BERTClassifier(nn.Module):
    """
    MLP classifier that operates on frozen BERT embeddings.

    This replaces the TF-IDF MLP.  The input dimension is fixed at 768
    (bert-base-uncased hidden size).  The classifier head is identical
    in structure to the original MLP so all FL infrastructure
    (FedAvg, Shapley estimation, attacks) works unchanged.
    """

    def __init__(self, hidden_dim: int = 256, num_classes: int = 20,
                 dropout: float = 0.3):
        super().__init__()
        self.input_dim = 768  # bert-base-uncased hidden size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Linear(768, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
