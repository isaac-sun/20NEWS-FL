"""
DistilBERT with manual LoRA (Low-Rank Adaptation) for 20 Newsgroups FL.

Architecture:
  DistilBERT (66M, frozen) + LoRA on Q,K,V,O of all 6 attention layers
  → [CLS] token (768-dim) → Linear(768, 20)

Only LoRA weights + classifier head are trainable (~630K params).
The frozen DistilBERT backbone is shared by all clients and never transmitted.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA Linear Layer
# ═══════════════════════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with low-rank adaptation:
        output = W @ x + b + (alpha/r) * dropout(B @ A @ x)

    where A ∈ R^{r × in} and B ∈ R^{out × r}.  A is Kaiming-initialized,
    B is zero-initialized so LoRA starts as an identity perturbation.
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Steal the frozen weights
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach(), requires_grad=False)
        self.bias = nn.Parameter(linear.bias.detach(), requires_grad=False) if linear.bias is not None else None

        # LoRA low-rank matrices (trainable)
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.empty(self.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # LoRA dropout (standard in PEFT, typically 0.05)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = self.lora_dropout(x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base + lora

    @staticmethod
    def from_linear(linear: nn.Linear, r: int = 8, alpha: float = 16.0,
                    dropout: float = 0.0) -> "LoRALinear":
        return LoRALinear(linear, r=r, alpha=alpha, dropout=dropout)


# ═══════════════════════════════════════════════════════════════════════════════
# DistilBERT transformer layer keys
# ═══════════════════════════════════════════════════════════════════════════════

LORA_TARGETS = {"q_lin", "k_lin", "v_lin", "out_lin"}  # all attention projections


# ═══════════════════════════════════════════════════════════════════════════════
# Injection helpers
# ═══════════════════════════════════════════════════════════════════════════════

def inject_lora_to_distilbert(
    model: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.0
) -> nn.Module:
    """
    Walk through the DistilBERT transformer layers and replace
    attention projection Linear layers with LoRALinear wrappers.

    Returns the modified model (same object, in-place).
    """
    for name, child in model.named_children():
        if isinstance(child, nn.Linear) and any(t in name for t in LORA_TARGETS):
            setattr(model, name, LoRALinear.from_linear(child, r=r, alpha=alpha,
                                                        dropout=dropout))
        else:
            inject_lora_to_distilbert(child, r=r, alpha=alpha, dropout=dropout)
    return model


def get_lora_state_dict(model: nn.Module) -> OrderedDict:
    """Extract only LoRA + classifier-head parameters (trainable)."""
    sd = OrderedDict()
    for name, param in model.named_parameters():
        if param.requires_grad:
            sd[name] = param.detach()
    return sd


def load_lora_state_dict(model: nn.Module, state_dict: OrderedDict):
    """Load only LoRA/head parameters; skip keys not present or frozen."""
    model_sd = model.state_dict()
    for key, val in state_dict.items():
        if key in model_sd:
            model_sd[key].copy_(val.to(model_sd[key].dtype))
    model.load_state_dict(model_sd, strict=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Main LoRA model
# ═══════════════════════════════════════════════════════════════════════════════

class DistilBERTWithLoRA(nn.Module):
    """
    DistilBERT + LoRA adapters + classification head.

    Parameters
    ----------
    model_name : HuggingFace model id (default: distilbert-base-uncased).
    num_classes : number of output classes.
    lora_r : LoRA rank.
    lora_alpha : LoRA scaling factor.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased",
                 num_classes: int = 20, lora_r: int = 8, lora_alpha: float = 16.0,
                 lora_dropout: float = 0.0, model_dir: str = ""):
        super().__init__()
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # ── Load DistilBERT ──────────────────────────────────────────────
        bert_kwargs = {}
        if model_dir:
            bert_kwargs["cache_dir"] = model_dir
        self.bert: nn.Module = AutoModel.from_pretrained(model_name, **bert_kwargs)

        # ── Inject LoRA ──────────────────────────────────────────────────
        inject_lora_to_distilbert(self.bert, r=lora_r, alpha=lora_alpha,
                                  dropout=lora_dropout)

        # ── Freeze base params, unfreeze LoRA ────────────────────────────
        for name, param in self.bert.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # ── Classification head ──────────────────────────────────────────
        hidden_size = self.bert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes),
        )

        self._trainable_keys: list[str] | None = None

    @property
    def trainable_keys(self) -> list[str]:
        """Cached list of trainable parameter names."""
        if self._trainable_keys is None:
            self._trainable_keys = [n for n, p in self.named_parameters() if p.requires_grad]
        return self._trainable_keys

    def get_trainable_state_dict(self) -> OrderedDict:
        """Return state dict containing only trainable (LoRA + head) parameters."""
        sd = OrderedDict()
        for name in self.trainable_keys:
            sd[name] = self.state_dict()[name].clone()
        return sd

    def load_trainable_state_dict(self, partial_sd: OrderedDict):
        """Load only trainable parameters into the model."""
        load_lora_state_dict(self, partial_sd)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        # DistilBERT forward
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (first token)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        return self.classifier(cls_emb)

    def total_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer helper (shared by data pipeline and clients)
# ═══════════════════════════════════════════════════════════════════════════════

def get_tokenizer(model_name: str = "distilbert-base-uncased") -> AutoTokenizer:
    """Get the tokenizer for the LoRA model."""
    return AutoTokenizer.from_pretrained(model_name)
