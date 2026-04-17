import copy
from collections import OrderedDict, deque

import numpy as np
from torch.utils.data import DataLoader

from utils.metrics import evaluate_model


class FLServer:
    """Federated learning server managing global model and evaluation."""

    def __init__(self, model, val_dataset, test_dataset, config):
        self.model = model
        self.config = config
        self.global_state_dict = copy.deepcopy(model.state_dict())
        self.prev_global_state_dict = None
        # Sliding window of up to 3 previous global state dicts (oldest first)
        # Used by SDFR / AFR attacks for 3-round delta estimation
        self.global_history: deque = deque(maxlen=3)
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
        """Return a clone of the current global state dict."""
        return OrderedDict(
            {k: v.clone() for k, v in self.global_state_dict.items()}
        )

    def update_global_model(self, new_state_dict: OrderedDict):
        """Update the global model with new parameters."""
        self.prev_global_state_dict = self.global_state_dict
        # Push current state into 3-round history before overwriting
        self.global_history.append(
            OrderedDict({k: v.clone() for k, v in self.global_state_dict.items()})
        )
        self.global_state_dict = OrderedDict(
            {k: v.clone() for k, v in new_state_dict.items()}
        )
        self.model.load_state_dict(self.global_state_dict)

    def evaluate(self):
        """Evaluate global model on test set. Returns (loss, accuracy)."""
        self.model.load_state_dict(self.global_state_dict)
        return evaluate_model(self.model, self.test_loader, self.config.device)

    def evaluate_val(self):
        """Evaluate global model on validation set. Returns (loss, accuracy)."""
        self.model.load_state_dict(self.global_state_dict)
        return evaluate_model(self.model, self.val_loader, self.config.device)
