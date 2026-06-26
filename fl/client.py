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
            dataset, batch_size=config.batch_size, shuffle=True, drop_last=False,
            num_workers=8, pin_memory=True,
        )

    def train(self, global_state_dict: OrderedDict) -> OrderedDict:
        """
        Local training starting from global model.
        Returns the model update delta = local_params - global_params.
        """
        return self._do_train(global_state_dict, self.data_loader)

    def train_with_dataset(self, global_state_dict: OrderedDict,
                           dataset) -> OrderedDict:
        """
        Local training on an alternative dataset (e.g., label-flipped).
        Returns the model update delta = local_params - global_params.
        """
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size,
            shuffle=True, drop_last=False,
            num_workers=8, pin_memory=True,
        )
        return self._do_train(global_state_dict, loader)

    def _do_train(self, global_state_dict: OrderedDict,
                  data_loader) -> OrderedDict:
        """Core local training loop (LoRA-aware).

        Optimizations (zero extra FWD/BWD cost):
          - Label smoothing → better generalization for 20-class
          - Linear warmup + linear decay → stabilizes transformer fine-tuning
          - Gradient clipping → prevents instability
          - Weight decay → mild L2 regularization on LoRA params
        """
        model = self.model_fn()
        from models.lora_classifier import load_lora_state_dict
        load_lora_state_dict(model, global_state_dict)
        model.to(self.config.device)
        model.train()

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.local_lr,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )

        total_steps = self.config.local_epochs * len(data_loader)
        warmup_steps = max(1, int(total_steps * self.config.warmup_ratio))

        global_step = 0
        for _ in range(self.config.local_epochs):
            for batch in data_loader:
                # ── Linear warmup + linear decay per-step LR ─────────────
                if global_step < warmup_steps:
                    lr = self.config.local_lr * global_step / warmup_steps
                else:
                    progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                    lr = self.config.local_lr * (1.0 - 0.9 * progress)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                input_ids, attn_mask, y = [b.to(self.config.device, non_blocking=True)
                                            for b in batch]
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = criterion(model(input_ids, attention_mask=attn_mask), y)
                loss.backward()

                # ── Gradient clipping ────────────────────────────────────
                torch.nn.utils.clip_grad_norm_(
                    trainable_params,
                    max_norm=self.config.max_grad_norm,
                )

                optimizer.step()
                global_step += 1

        # Compute update: delta for trainable params only
        trainable_keys = model.trainable_keys
        local_trainable_sd = model.get_trainable_state_dict()
        update = OrderedDict()
        for key in trainable_keys:
            update[key] = local_trainable_sd[key] - global_state_dict[key].to(
                local_trainable_sd[key].device, dtype=local_trainable_sd[key].dtype)

        return update
