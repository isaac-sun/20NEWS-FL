# 20NEWS-FL: Federated Learning Free-Rider Attack Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/isaac-sun/20NEWS-FL/blob/main/colab_demo.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**DistilBERT + LoRA fine-tuning + Per-Class Shapley Value analysis for detecting free-rider attacks in Federated Learning on the 20 Newsgroups dataset.**

---

## Overview

Federated Learning (FL) is vulnerable to **free-rider attacks**: malicious clients contribute fake updates (noise or scaled deltas) while still benefiting from the aggregated global model. This project implements three state-of-the-art attack variants and detects them via **per-class Shapley value decomposition**.

### Threat Model

- **10 clients** participate in FL with a DistilBERT + LoRA classifier
- **4 malicious clients** (40%) generate fake updates using DFR / SDFR / AFR
- Detection: each client's contribution is decomposed per-class via Monte Carlo Shapley estimation, exposing the attack's class-agnostic signature

### Architecture

```
Client data → DistilBERT (frozen) → LoRA adapters (trainable) → Classifier head
                                      ↑
                              FL: FedAvg aggregation
                              Detection: Per-class Shapley values
```

---

## Experiments

| # | Experiment | Attack | Paper |
|---|---|---|---|
| 1 | `baseline_no_attack` | None | — |
| 2 | `attack_dfr` | **DFR** — Disguised Free-Rider | Fraboni et al. |
| 3 | `attack_sdfr` | **SDFR** — Scaled Delta Free-Rider | Zhu et al. |
| 4 | `attack_afr` | **AFR** — Advanced Free-Rider | Zhu et al. |

### Attack Formulations

**DFR** (Disguised Free-Rider):
$$g_i^{(t)} = \sigma \cdot t^{-\gamma} \cdot \mathcal{N}(0, I)$$

**SDFR** (Scaled Delta Free-Rider):
$$U_f(\theta) = \frac{\|\Delta_t\|}{\|\Delta_{t-1}\|} \cdot \Delta_t, \quad \Delta_t = \theta(t) - \theta(t-1)$$

**AFR** (Advanced Free-Rider):
$$U_{\text{AFR}} = U_f(\theta) + z, \quad z_i \sim \mathcal{N}\!\left(0, \frac{|\varphi(t)|^2}{d}\right)$$
where $|\varphi(t)|^2 = \left(\frac{n^2}{n + (n^2 - n) \cdot \mathbb{E}[\cos\beta]} - 1\right) \cdot |\mathbb{E}[U_f(\theta)]|^2$

### Detection: Per-Class Shapley Values

For each client $i$ and class $c$, the Shapley value $\phi_i^c$ is estimated via Monte Carlo permutation sampling:

$$\phi_i^c = \frac{1}{M} \sum_{m=1}^{M} \left[ \mathcal{L}_c(\theta_{S \cup \{i\}}) - \mathcal{L}_c(\theta_S) \right]$$

where $S$ is a random coalition preceding $i$ in the permutation. The overall Shapley value is the class-frequency-weighted sum: $\phi_i = \sum_c w_c \cdot \phi_i^c$.

### Key Metrics

| Metric | What it captures |
|---|---|
| **Class SV Variance** | Honest clients specialize → high variance; free-riders contribute uniformly → low variance |
| **Positive Class SV Sum** | Free-riders produce near-zero or negative class contributions |
| **Concentration Ratio** | Honest clients dominate a few classes; free-riders spread evenly |
| **Shapley Gap** | $\phi_{\text{honest}} - \phi_{\text{malicious}}$ → positive gap confirms detection |

### Output

| Artifact | Content |
|---|---|
| `results/experiment_results.xlsx` | 3 sheets: round-level Shapley details, experiment summaries, per-class records |
| `results/plots/fig_01.png` | Accuracy & Loss curves across all experiments |
| `results/plots/fig_02.png` | Round-level Shapley comparison + cumulative SV bars |
| `results/plots/fig_03.png` | Per-client per-round Shapley heatmap (all experiments) |
| `results/plots/fig_04.png` | Per-class SV fingerprint (honest vs malicious bar chart) |
| `results/plots/fig_05.png` | Two-metric scatter: variance vs positive sum |
| `results/plots/fig_06.png` | Metric distribution boxplots |
| `results/plots/fig_07.png` | Per-class SV deep-dive heatmap (single honest vs malicious) |
| `results/plots/fig_08.png` | Class metrics over rounds (variance / positive sum) |
| `results/plots/fig_09.png` | Per-client cumulative SV bar chart |
| `results/plots/fig_10.png` | Multi-attack summary comparison |

### Expected Results

| Experiment | Global Accuracy | Shapley Gap (H − M) | Detection |
|---|---|---|---|
| Baseline (no attack) | ~82–85% | ~0 (no gap) | N/A |
| DFR | ~80–83% | Honest > Malicious | ✓ via variance/pos-sum |
| SDFR | ~78–82% | Honest > Malicious | ✓ via variance/pos-sum |
| AFR | ~76–80% | Honest > Malicious | ✓ via variance/pos-sum |

Malicious clients consistently show **lower class SV variance**, **near-zero positive class SV sum**, and **lower overall Shapley values** — clear separation from honest clients.

---

## Quick Start

### Google Colab (Recommended)

Click the badge above, or go to:

```
https://colab.research.google.com/github/isaac-sun/20NEWS-FL/blob/main/colab_demo.ipynb
```

1. Set `GITHUB_USER` to your GitHub username in Cell 1
2. Runtime → Run all
3. Download `results.zip`

**Runtime**: ~2–4 hours on T4 GPU, ~6–10 hours on CPU.

### Local / Cloud

```bash
git clone https://github.com/isaac-sun/20NEWS-FL.git
cd 20NEWS-FL
pip install -r requirements.txt
python main.py
```

For a quick smoke test (5 rounds, ~20 min):

```python
import main as m
base = m.Config(num_rounds=5, num_mc_samples=10, batch_size=16)
# ... see colab_demo.ipynb Cell 4 for the full quick-test snippet
```

---

## Project Structure

```
├── main.py                     # Entry point — launches all 4 experiments
├── config.py                   # Hyperparameters (LoRA rank, FL rounds, attack params)
├── data/
│   └── newsgroups.py           # 20 Newsgroups loading + DistilBERT tokenization
├── models/
│   ├── lora_classifier.py      # DistilBERT + LoRA adapters + classification head
│   └── bert_classifier.py      # Legacy MLP classifier (frozen BERT embeddings)
├── fl/
│   ├── server.py               # FL server: FedAvg aggregation, global model, eval
│   ├── client.py               # FL client: local LoRA fine-tuning with warmup + decay
│   └── aggregation.py          # FedAvg with optional server momentum
├── attacks/
│   ├── dfr.py                  # DFR attack + sigma auto-estimation
│   ├── sdfr.py                 # SDFR attack (scaled delta from global history)
│   └── afr.py                  # AFR attack + calibrated sparse noise + EMA state
├── contribution/
│   └── shapley.py              # MC per-class Shapley + coalition weighting + metrics
├── detection/
│   ├── utility_score.py        # EMA utility score tracker
│   └── free_rider_detection.py # K-means clustering on feature values
├── utils/
│   ├── export.py               # Multi-sheet Excel export
│   ├── metrics.py              # Model evaluation (loss, accuracy)
│   ├── partition.py            # IID / Non-IID data partitioning
│   ├── seed.py                 # Deterministic seeding (PyTorch + numpy + random)
│   └── logger.py               # Logging configuration
├── visualization/
│   └── plots.py                # 10-chart publication-quality visualization suite
├── colab_demo.ipynb            # Colab notebook (clone → install → run → download)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Requirements

| Package | Purpose |
|---|---|
| `torch ≥ 2.0` | Deep learning framework |
| `transformers` | DistilBERT model + tokenizer |
| `scikit-learn` | Dataset loading, K-means clustering |
| `numpy` | Numerical computation |
| `pandas` | Data wrangling |
| `matplotlib` / `seaborn` | Visualization |
| `openpyxl` | Excel export |
| `tqdm` | Progress bars |

Install all at once:

```bash
pip install torch transformers scikit-learn numpy pandas matplotlib seaborn openpyxl tqdm
```

---

## Configuration

Key hyperparameters in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `num_clients` | 10 | Total FL clients |
| `num_rounds` | 50 | FL communication rounds |
| `local_epochs` | 2 | Local training epochs per round |
| `lora_r` | 32 | LoRA rank |
| `lora_alpha` | 32.0 | LoRA scaling factor |
| `malicious_ratio` | 0.4 | Fraction of malicious clients |
| `participation_ratio` | 0.8 | Clients selected per round |
| `num_mc_samples` | 30 | Monte Carlo permutations for Shapley |
| `server_momentum` | 0.9 | Server-side momentum for FedAvg |
| `label_smoothing` | 0.1 | Cross-entropy label smoothing |
| `batch_size` | 32 | Training batch size |

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fraboni2021free,
  title     = {Free-rider Attacks on Model Aggregation in Federated Learning},
  author    = {Fraboni, Yann and Vidal, Richard and Kameni, Laetitia and Lorenzi, Marco},
  journal   = {AISTATS},
  year      = {2021}
}

@article{zhu2023svrfl,
  title     = {SVRFL: Shapley Value-based Robust Federated Learning},
  author    = {Zhu, Jiyue and Anwar, Aamir and others},
  year      = {2023}
}
```
