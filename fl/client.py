from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class FLClient:
    """Federated learning client with local LoRA training."""

    def __init__(self, client_id: int, dataset, model_fn, config):
        self.client_id = client_id
        self.dataset = dataset
        self.config = config
        self.model_fn = model_fn

        # GPU-optimized DataLoader when CUDA available
        loader_kwargs = dict(
            batch_size=config.batch_size, shuffle=True, drop_last=False,
        )
        if config.device == "cuda":
            loader_kwargs.update(num_workers=4, pin_memory=True)
        self.data_loader = DataLoader(dataset, **loader_kwargs)

    def train(self, global_state_dict: OrderedDict) -> OrderedDict:
        """Local training: load global state, train, return delta."""
        return self._do_train(global_state_dict, self.data_loader)

    def train_with_dataset(self, global_state_dict: OrderedDict,
                           dataset) -> OrderedDict:
        """Train on a different dataset (used by Shapley evaluation)."""
        loader_kwargs = dict(
            batch_size=self.config.batch_size, shuffle=True, drop_last=False,
        )
        if self.config.device == "cuda":
            loader_kwargs.update(num_workers=4, pin_memory=True)
        loader = DataLoader(dataset, **loader_kwargs)
        return self._do_train(global_state_dict, loader)

    def _do_train(self, global_state_dict: OrderedDict,
                  data_loader) -> OrderedDict:
        """Core local training loop — full model, all params trainable."""
        model = self.model_fn()
        model.load_state_dict(global_state_dict, strict=True)
        model.to(self.config.device)
        model.train()

        all_params = list(model.parameters())
        optimizer = torch.optim.AdamW(
            all_params,
            lr=self.config.local_lr,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )

        total_steps = self.config.local_epochs * len(data_loader)
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))

        use_amp = self.config.device == "cuda"
        global_step = 0
        for _ in range(self.config.local_epochs):
            for batch in data_loader:
                if global_step < warmup_steps:
                    lr = self.config.local_lr * (global_step + 1) / max(warmup_steps, 1)
                else:
                    progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                    lr = self.config.local_lr * (1.0 - 0.9 * progress)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                input_ids, attn_mask, y = [
                    b.to(self.config.device, non_blocking=use_amp)
                    for b in batch
                ]
                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        loss = criterion(model(input_ids, attention_mask=attn_mask), y)
                else:
                    loss = criterion(model(input_ids, attention_mask=attn_mask), y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    all_params, max_norm=self.config.max_grad_norm,
                )
                optimizer.step()
                global_step += 1

        # Compute delta on ALL parameters
        local_sd = model.state_dict()
        update = OrderedDict()
        for key in local_sd:
            update[key] = local_sd[key].detach() - global_state_dict[key].to(
                local_sd[key].device, dtype=local_sd[key].dtype)
        return update
