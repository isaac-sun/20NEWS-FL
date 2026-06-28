import math
import unittest
from collections import OrderedDict

import torch

from attacks.afr import (
    afr_attack,
    estimate_c_from_norms,
    estimate_decay_rate,
    estimate_e_cos_beta,
)
from attacks.dfr import dfr_attack, estimate_dfr_sigma, plain_free_rider_update
from attacks.sdfr import sdfr_attack


def state(values):
    return OrderedDict(weight=torch.tensor(values, dtype=torch.float32))


class DFRTests(unittest.TestCase):
    def test_sigma_is_fitted_from_first_global_increment(self):
        theta_0 = state([0.0, 0.0, 0.0, 0.0])
        theta_1 = state([-1.0, -1.0, 1.0, 1.0])
        sigma = estimate_dfr_sigma(theta_1, [theta_0])
        self.assertAlmostEqual(sigma, 1.0, places=6)

    def test_disguised_update_has_paper_decay(self):
        generator = torch.Generator().manual_seed(7)
        update = dfr_attack(
            state([0.0] * 100_000), sigma=0.8, round_num=4,
            gamma=1.0, generator=generator,
        )
        self.assertAlmostEqual(
            update["weight"].std(unbiased=False).item(), 0.2, delta=0.003
        )

    def test_unobservable_sigma_uses_plain_free_rider(self):
        update = plain_free_rider_update(state([1.0, 2.0]))
        self.assertTrue(torch.equal(update["weight"], torch.zeros(2)))


class SDFRTests(unittest.TestCase):
    def test_scaled_delta_matches_equation_9(self):
        theta_0 = state([0.0, 0.0])
        theta_1 = state([1.0, 0.0])
        theta_2 = state([3.0, 0.0])
        update = sdfr_attack(theta_2, [theta_0, theta_1])
        torch.testing.assert_close(update["weight"], torch.tensor([4.0, 0.0]))

    def test_requires_three_global_states(self):
        with self.assertRaises(ValueError):
            sdfr_attack(state([1.0]), [state([0.0])])


class AFRTests(unittest.TestCase):
    def test_decay_and_cosine_follow_lemma_1(self):
        decay = estimate_decay_rate(0.25, 1.0, round_num=3)
        self.assertAlmostEqual(decay, math.log(2.0), places=7)
        expected = 1.0 / (1.0 + math.exp(2.0 * decay * 3))
        self.assertAlmostEqual(
            estimate_e_cos_beta(1.0, decay, 3), expected, places=7
        )

    def test_c_estimation_combines_lemmas_1_and_2(self):
        n = 10
        cosine = 0.25
        mean_norm = 2.0
        local_norm = mean_norm * math.sqrt(
            n * n / (n + (n * n - n) * cosine)
        )
        c = estimate_c_from_norms(
            local_norm, mean_norm, n, decay_rate=0.0, round_num=1
        )
        self.assertAlmostEqual(c, math.sqrt(cosine / (1.0 - cosine)), places=6)

    def test_sparse_noise_norm_matches_algorithm_1(self):
        size = 100_000
        theta_0 = state([0.0] * size)
        theta_1 = state([0.1] * size)
        theta_2 = state([0.15] * size)
        base = sdfr_attack(theta_2, [theta_0, theta_1])["weight"]
        attacked, base_norm = afr_attack(
            theta_2, [theta_0, theta_1], n_total=10,
            e_cos_beta=0.5, noisy_frac=0.2, seed=42,
        )
        noise_norm = (attacked["weight"] - base).norm().item()
        expected_multiplier = math.sqrt(100.0 / 55.0 - 1.0)
        self.assertAlmostEqual(
            noise_norm, expected_multiplier * base_norm,
            delta=0.04 * expected_multiplier * base_norm,
        )
        self.assertTrue(torch.isfinite(attacked["weight"]).all())


if __name__ == "__main__":
    unittest.main()
