"""
20 Newsgroups data loading with DistilBERT tokenization.

Returns tokenized TensorDatasets for use with DistilBERTWithLoRA.
Each sample is (input_ids, attention_mask, label).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# ── 云平台诊断：确保输出实时可见 ──────────────────────────────────────
import sys as _sys
def _echo(msg: str):
    """Print message directly to stdout, bypassing logger. Works even if
    stdout is a non-standard wrapper (Jupyter, Databricks, etc.)."""
    try:
        _sys.stdout.write(msg + "\n")
        _sys.stdout.flush()
    except Exception:
        pass

def _flush():
    try:
        _sys.stdout.flush()
        _sys.stderr.flush()
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
    """
    Load 20 Newsgroups with tokenized inputs for DistilBERT + LoRA.

    Parameters
    ----------
    model_name : HuggingFace model id.
    val_ratio : fraction of training data to reserve for validation.
    random_state : random seed for the train/val split.
    max_seq_length : maximum token length (trim longer, pad shorter).
    max_samples : if set, truncate to this many training samples (for quick tests).

    Returns
    -------
    train_ds : TensorDataset  (input_ids, attention_mask, labels)
    val_ds : TensorDataset
    test_ds : TensorDataset
    input_dim : None  (not used with DistilBERT)
    train_labels : np.ndarray
    """
    # ── Fetch raw data ──────────────────────────────────────────────────────
    _echo("Fetching 20 Newsgroups...")
    logger.info("Fetching 20 Newsgroups...")
    _flush()
    news_train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    news_test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    _flush()

    train_texts = news_train.data
    test_texts = news_test.data
    y_train_full = np.array(news_train.target)
    y_test = np.array(news_test.target)

    if max_samples is not None and max_samples < len(train_texts):
        train_texts = train_texts[:max_samples]
        y_train_full = y_train_full[:max_samples]
        logger.info("Truncated training set to %d samples", max_samples)

    # ── Train / validation split (on raw texts first) ───────────────────────
    train_texts, val_texts, y_train, y_val = train_test_split(
        train_texts, y_train_full,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_train_full,
    )

    # ── Tokenize ────────────────────────────────────────────────────────────
    _echo(f"Loading tokenizer: {model_name}")
    logger.info("Loading tokenizer: %s", model_name)
    _flush()
    tok_kwargs = {}
    if model_dir:
        tok_kwargs["cache_dir"] = model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)

    def _tokenize(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_seq_length, return_tensors="pt",
        )
        return encoded["input_ids"], encoded["attention_mask"]

    _echo(f"Tokenizing {len(train_texts)} train texts...")
    logger.info("Tokenizing %d train texts...", len(train_texts))
    train_ids, train_mask = _tokenize(train_texts)

    logger.info("Tokenizing %d val texts...", len(val_texts))
    val_ids, val_mask = _tokenize(val_texts)

    logger.info("Tokenizing %d test texts...", len(test_texts))
    test_ids, test_mask = _tokenize(test_texts)

    # ── Wrap in TensorDatasets ──────────────────────────────────────────────
    train_ds = TensorDataset(
        train_ids, train_mask,
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        val_ids, val_mask,
        torch.tensor(y_val, dtype=torch.long),
    )
    test_ds = TensorDataset(
        test_ids, test_mask,
        torch.tensor(y_test, dtype=torch.long),
    )

    logger.info(
        "Data ready: train=%d, val=%d, test=%d, max_seq_length=%d",
        len(train_ds), len(val_ds), len(test_ds), max_seq_length,
    )
    return train_ds, val_ds, test_ds, None, y_train
