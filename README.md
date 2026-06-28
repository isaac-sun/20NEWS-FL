# Full-DistilBERT Federated Learning on 20 Newsgroups

This repository evaluates free-rider attacks and per-class Shapley values in
federated full-model fine-tuning. The executable path uses full
`distilbert-base-uncased`; all 66.4M parameters are trained and communicated.

## Experiments

`main.py` runs four experiments with identical initialization and data splits:

1. FedAvg baseline
2. Disguised free-rider (DFR), Fraboni et al.
3. Scaled-delta free-rider (SDFR), Zhu et al. Eq. (9)
4. Advanced free-rider (AFR), Zhu et al. Algorithm 1

SDFR and AFR use two honest warm-up rounds because their construction requires
`theta(t)`, `theta(t-1)`, and `theta(t-2)`. All clients participate and standard
sample-size-weighted FedAvg is used, matching the assumptions of the attacks.

## NVIDIA GPU

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
nvidia-smi
python3 -m unittest discover -s tests -v
FORCE_DEVICE=cuda python -u main.py --require-cuda --gpu-profile auto
```

For a detached Linux run:

```bash
nohup env FORCE_DEVICE=cuda PYTHONUNBUFFERED=1 \
  python -u main.py --require-cuda --gpu-profile auto \
  > experiment.log 2>&1 &
tail -f experiment.log
```

`auto` keeps the training batch at 8 and selects a safe Shapley evaluation
batch from detected VRAM. Use `--results-dir PATH` to redirect all outputs.

## Google Colab

Open [`colab_demo.ipynb`](colab_demo.ipynb), select a GPU runtime, and run all
cells. The notebook preserves Colab's CUDA-enabled PyTorch, installs only the
remaining dependencies, runs with `--require-cuda`, and copies the completed
`results.zip` to Google Drive.

Results are written to `results/experiment_results.xlsx`, including an
`experiment_config` sheet, and figures are written to `results/plots/`.

See [EXPERIMENTS.md](EXPERIMENTS.md) for formulas and the complete active
configuration.
