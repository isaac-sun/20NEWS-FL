import unittest
from collections import OrderedDict
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from contribution.shapley import _build_coalition_params
from fl.aggregation import fedavg_aggregate
from fl.client import FLClient


class FederatedMathTests(unittest.TestCase):
    def setUp(self):
        self.global_state = OrderedDict(weight=torch.tensor([10.0]))
        self.updates = {
            0: OrderedDict(weight=torch.tensor([2.0])),
            1: OrderedDict(weight=torch.tensor([-1.0])),
        }

    def test_fedavg_uses_normalized_sample_counts(self):
        result = fedavg_aggregate(
            self.global_state, self.updates, weights={0: 1, 1: 3}
        )
        torch.testing.assert_close(result["weight"], torch.tensor([9.75]))

    def test_shapley_coalition_uses_same_local_model_weighting(self):
        result = _build_coalition_params(
            self.global_state, self.updates, [0, 1], {0: 1, 1: 3}
        )
        torch.testing.assert_close(result["weight"], torch.tensor([9.75]))

    def test_non_finite_update_is_rejected(self):
        bad = {0: OrderedDict(weight=torch.tensor([float("nan")]))}
        with self.assertRaises(FloatingPointError):
            fedavg_aggregate(self.global_state, bad)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.classifier = nn.Linear(4, 2)

    def forward(self, input_ids, attention_mask):
        hidden = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.classifier(pooled)


class ClientTrainingTests(unittest.TestCase):
    def test_full_model_client_update_is_finite_and_on_cpu(self):
        config = SimpleNamespace(
            batch_size=2,
            device="cpu",
            local_lr=1e-3,
            weight_decay=0.0,
            label_smoothing=0.0,
            local_epochs=1,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
        )
        dataset = TensorDataset(
            torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            torch.ones(4, 2, dtype=torch.long),
            torch.tensor([0, 1, 0, 1]),
        )
        model = TinyClassifier()
        global_state = OrderedDict(
            (key, value.detach().clone()) for key, value in model.state_dict().items()
        )
        client = FLClient(0, dataset, TinyClassifier, config)
        update = client.train(global_state)
        self.assertEqual(set(update), set(global_state))
        self.assertTrue(all(value.device.type == "cpu" for value in update.values()))
        self.assertTrue(all(torch.isfinite(value).all() for value in update.values()))


if __name__ == "__main__":
    unittest.main()
