# Federated Learning on 20 Newsgroups — Free-Rider Attacks & Shapley Detection

End-to-end pipeline for federated learning experiments with:
- FedAvg baseline on 20 Newsgroups (TF-IDF + MLP)
- Three free-rider attacks: DFR, SDFR, AFR
- SVRFL-style round-level Shapley value contribution estimation
- Feature-value-based free-rider detection with K-means clustering
- Utility-score tracking
- Excel export and plots

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Project Structure

```
config.py                          — experiment configuration (dataclass)
main.py                            — experiment runner (4 experiments)
data/newsgroups.py                 — 20 Newsgroups loading + TF-IDF
models/mlp.py                      — simple 2-layer MLP
fl/server.py                       — FL server (selection, eval)
fl/client.py                       — FL client (local training)
fl/aggregation.py                  — FedAvg aggregation
attacks/dfr.py                     — Disguised Free-Rider
attacks/sdfr.py                    — Scaled Delta Free-Rider
attacks/afr.py                     — Advanced Free-Rider
contribution/shapley.py            — Monte Carlo Shapley estimation (SVRFL-style)
detection/free_rider_detection.py  — feature-value detection + K-means
detection/utility_score.py         — EMA utility score tracker
utils/seed.py                      — reproducibility
utils/logger.py                    — logging
utils/partition.py                 — IID / non-IID data partitioning
utils/metrics.py                   — model evaluation
utils/export.py                    — Excel export
```

## Outputs

All outputs are saved under `results/`:

| File | Description |
|------|-------------|
| `experiment_results.xlsx` | Sheet 1: per-round Shapley details; Sheet 2: experiment summary |
| `test_accuracy.png` | Global test accuracy curves |
| `test_loss.png` | Global test loss curves |
| `shapley_comparison.png` | Shapley values: honest vs malicious clients |

## Configuration

Edit `Config` in `config.py` or modify the `base` config in `main.py`.
Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clients` | 10 | Total FL clients |
| `num_rounds` | 30 | Communication rounds |
| `local_epochs` | 3 | Local training epochs per round |
| `participation_ratio` | 0.8 | Fraction of clients selected each round |
| `malicious_ratio` | 0.3 | Fraction of clients that are attackers |
| `num_mc_samples` | 30 | Monte Carlo permutations for Shapley |
| `detection_threshold_h` | 200.0 | K-means detection threshold |
| `iid` | True | IID partitioning (set False for non-IID) |

## Shapley Value Definition

Following the SVRFL paper, the round-level Shapley value is computed as:

```
v(S, D_v) = F(w_g^t, D_v) − F(w_S^t, D_v)
```

where `w_S^t = w_g^t + (η / |S|) · Σ_{i∈S} δ_i` and `F` is cross-entropy
loss on the validation set. Approximated via Monte Carlo permutation sampling.
