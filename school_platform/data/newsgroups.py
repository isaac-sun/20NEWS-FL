"""
20 Newsgroups data loading with DistilBERT tokenization.

Platform edition: direct HuggingFace access + local cache support.
Each sample is (input_ids, attention_mask, label).
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# ── 诊断辅助 — 确保云平台能看到进度 ──────────────────────────────────
def _stamp(msg: str):
    for s in (sys.stdout, sys.stderr):
        try:
            s.write(f"[data] {msg}\n")
            s.flush()
        except Exception:
            pass


def load_newsgroups(
    model_name: str = "distilbert-base-uncased",
    val_ratio: float = 0.1,
    random_state: int = 42,
    max_seq_length: int = 256,
    max_samples: int | None = None,
    model_dir: str = "",
):
    """Load 20 Newsgroups with tokenized inputs for DistilBERT + LoRA.

    Returns: (train_ds, val_ds, test_ds, input_dim=None, train_labels)
    """
    _stamp("Fetching 20 Newsgroups (this may take a moment on first run)...")
    logger.info("Fetching 20 Newsgroups (this may take a moment)...")
    news_train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    news_test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    _stamp("20 Newsgroups data fetched successfully")

    train_texts = news_train.data
    test_texts = news_test.data
    y_train_full = np.array(news_train.target)
    y_test = np.array(news_test.target)

    if max_samples is not None and max_samples < len(train_texts):
        train_texts = train_texts[:max_samples]
        y_train_full = y_train_full[:max_samples]
        logger.info("Truncated training set to %d samples", max_samples)

    train_texts, val_texts, y_train, y_val = train_test_split(
        train_texts, y_train_full,
        test_size=val_ratio, random_state=random_state, stratify=y_train_full,
    )

    _stamp(f"Loading tokenizer: {model_name}")
    logger.info("Loading tokenizer: %s", model_name)
    tok_kwargs = {}
    if model_dir:
        tok_kwargs["cache_dir"] = model_dir
        _stamp(f"  Using model cache: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    _stamp("Tokenizer loaded")

    def _tokenize(texts):
        encoded = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_seq_length, return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    _stamp(f"Tokenizing {len(train_texts)} train texts...")
    train_ids, train_mask = _tokenize(train_texts)
    _stamp(f"Tokenizing {len(val_texts)} val texts...")
    val_ids, val_mask = _tokenize(val_texts)
    _stamp(f"Tokenizing {len(test_texts)} test texts...")
    test_ids, test_mask = _tokenize(test_texts)

    train_ds = TensorDataset(train_ids, train_mask, torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(val_ids, val_mask, torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(test_ids, test_mask, torch.tensor(y_test, dtype=torch.long))

    _stamp(f"Data ready: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    logger.info("Data ready: train=%d, val=%d, test=%d, max_seq_length=%d",
                len(train_ds), len(val_ds), len(test_ds), max_seq_length)
    return train_ds, val_ds, test_ds, None, y_train
