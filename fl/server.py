from __future__ import annotations

import copy
from collections import OrderedDict, deque

import numpy as np
from torch.utils.data import DataLoader

from utils.metrics import evaluate_model


class FLServer:
    """Federated learning server managing global model and evaluation.

    With LoRA, global_state_dict stores only trainable parameters.
    The frozen DistilBERT backbone is loaded once in the model and never changes.

    Supports server-side momentum and learning rate decay for better convergence.
    """

    def __init__(self, model, val_dataset, test_dataset, config):
        self.model = model
        self.config = config
        self.global_state_dict = model.get_state_dict()
        # Sliding window of up to 3 previous global state dicts
        self.global_history: deque = deque(maxlen=3)
        # Server momentum buffer (only used when config.server_momentum > 0)
        self.momentum_buffer: OrderedDict | None = None
        # Track current server_lr (starts from config.server_lr, decays each round)
        self.current_server_lr: float = config.server_lr
        self.val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    def select_clients(self, num_clients: int, participation_ratio: float) -> list:
        """Randomly select a subset of clients for this round."""
        num_selected = max(1, int(num_clients * participation_ratio))
        selected = sorted(
            np.random.choice(num_clients, num_selected, replace=False).tolist()
        )
        return selected

    def get_global_state_dict(self) -> OrderedDict:
        return OrderedDict(
            {k: v.clone() for k, v in self.global_state_dict.items()}
        )

    def update_global_model(self, new_state_dict: OrderedDict):
        """Update trainable params with new aggregated values.

        Applies server_lr exponential decay after this update.
        """
        # Push current state into history before overwriting
        self.global_history.append(
            OrderedDict({k: v.clone() for k, v in self.global_state_dict.items()})
        )
        self.global_state_dict = OrderedDict(
            {k: v.clone() for k, v in new_state_dict.items()}
        )
        self.model.load_state_dict_from(self.global_state_dict)

        # ── Apply server learning rate decay ────────────────────────────
        if self.config.server_lr_decay < 1.0:
            self.current_server_lr *= self.config.server_lr_decay
            self.current_server_lr = max(self.current_server_lr, 0.01)

    def evaluate(self):
        """Evaluate global model on test set. Returns (loss, accuracy)."""
        self.model.load_state_dict_from(self.global_state_dict)
        return evaluate_model(self.model, self.test_loader, self.config.device)

    def evaluate_val(self):
        """Evaluate global model on validation set. Returns (loss, accuracy)."""
        self.model.load_state_dict_from(self.global_state_dict)
        return evaluate_model(self.model, self.val_loader, self.config.device)
