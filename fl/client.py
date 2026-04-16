import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FLClient:
    """Federated learning client with local training."""

    def __init__(self, client_id: int, dataset, model_fn, config):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.model_fn = model_fn
        self.data_loader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, drop_last=False
        )

    def train(self, global_state_dict: OrderedDict) -> OrderedDict:
        """
        Local training starting from global model.
        Returns the model update delta = local_params - global_params.
        """
        model = self.model_fn()
        model.load_state_dict(copy.deepcopy(global_state_dict))
        model.to(self.config.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.local_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.local_epochs,
            eta_min=self.config.local_lr * 0.1,
        )
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.config.local_epochs):
            for X, y in self.data_loader:
                X, y = X.to(self.config.device), y.to(self.config.device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        # Compute update: delta = local - global
        local_sd = model.state_dict()
        update = OrderedDict()
        for key in global_state_dict:
            update[key] = local_sd[key].cpu() - global_state_dict[key].cpu()

        return update
