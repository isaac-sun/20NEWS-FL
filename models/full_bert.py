"""
Full DistilBERT classifier for federated learning.

All parameters are trainable; no layers are frozen.
The full model (~66M params) is communicated in every FL round.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class FullDistilBERTClassifier(nn.Module):
    """DistilBERT + classification head.  ALL parameters trainable."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 20,
        dropout: float = 0.3,
        model_dir: str = "",
    ):
        super().__init__()
        model_kwargs = {"cache_dir": model_dir} if model_dir else {}
        self.bert = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.dim, num_classes)
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    def get_state_dict(self) -> OrderedDict:
        """Return ALL parameters (clone for safety)."""
        return OrderedDict(
            {k: v.detach().clone() for k, v in self.state_dict().items()}
        )

    def load_state_dict_from(self, sd: OrderedDict):
        """Load a full state dict."""
        self.load_state_dict(sd, strict=True)


def get_tokenizer(
    model_name: str = "distilbert-base-uncased", model_dir: str = ""
) -> AutoTokenizer:
    tokenizer_kwargs = {"cache_dir": model_dir} if model_dir else {}
    return AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
