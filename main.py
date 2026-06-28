#!/usr/bin/env python3
"""
Federated Learning on 20 Newsgroups — full DistilBERT fine-tuning
Free-Rider Attacks + Per-Class Shapley Detection.

Works on: Colab / school GPU platform / local machine.
Auto-detects environment and applies appropriate optimizations.

Runs four experiments:
  1. baseline (no attack)
  2. DFR  (Disguised Free-Rider)
  3. SDFR (Scaled Delta Free-Rider)
  4. AFR  (Advanced Free-Rider)
"""

import os
import sys
import argparse
import gc

# ═══════════════════════════════════════════════════════════════════════════════
# 0. Bootstrap — earliest possible diagnostics, works everywhere
# ═══════════════════════════════════════════════════════════════════════════════

def _stamp(msg: str):
    """Write directly to stdout/stderr — survives broken loggers, containers, notebooks."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.write(f"[20NEWS-FL] {msg}\n")
            stream.flush()
        except Exception:
            pass

_stamp("Python bootstrap starting")
_stamp(f"  Python {sys.version.split()[0]}  |  cwd: {os.getcwd()}")

# ── Force unbuffered output (containers / cloud platforms) ───────────────
os.environ.setdefault("PYTHONUNBUFFERED", "1")
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):
        try:
            _s.reconfigure(write_through=True)
        except (OSError, ValueError, AttributeError):
            pass

# ── Project paths ────────────────────────────────────────────────────────
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_stamp(f"  Project dir: {_PROJECT_DIR}")

# sklearn data cache (writable dir for platforms with restricted /home)
_SKLEARN_DATA = os.path.join(_PROJECT_DIR, "sklearn_data")
os.environ.setdefault("SCIKIT_LEARN_DATA", _SKLEARN_DATA)
os.makedirs(_SKLEARN_DATA, exist_ok=True)

# HuggingFace model cache (local-first, offline-capable)
_MODEL_CACHE = os.path.join(_PROJECT_DIR, "model_cache")
if os.path.isdir(_MODEL_CACHE):
    os.environ.setdefault("MODEL_DIR", _MODEL_CACHE)
    _stamp(f"  model_cache found: {_MODEL_CACHE}")
else:
    _stamp("  model_cache not found (will download from HuggingFace)")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Environment tuning — safe defaults, no side effects
# ═══════════════════════════════════════════════════════════════════════════════

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# Suppress HuggingFace model-loading log spam
import logging as _logging
for _name in ("transformers", "transformers.modeling_utils", "safetensors", "filelock"):
    _logging.getLogger(_name).setLevel(_logging.ERROR)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Device detection — safe, FORCE_DEVICE=cpu skips CUDA entirely
# ═══════════════════════════════════════════════════════════════════════════════

_FORCE = os.environ.get("FORCE_DEVICE", "").lower()
_stamp(f"  FORCE_DEVICE={'auto' if not _FORCE else _FORCE}")

_stamp("Importing torch...")
import torch
_stamp(f"  torch {torch.__version__} imported")

def _detect_device() -> str:
    if _FORCE in ("cpu",):
        return "cpu"
    if _FORCE in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "FORCE_DEVICE=cuda was requested, but PyTorch cannot access CUDA. "
                "Check the NVIDIA driver, CUDA-enabled PyTorch, or Colab runtime."
            )
        return "cuda"
    try:
        if torch.cuda.is_available():
            _stamp(f"  CUDA: {torch.cuda.get_device_name(0)}")
            return "cuda"
    except Exception as e:
        _stamp(f"  CUDA detection failed: {e}")
    if torch.backends.mps.is_available():
        _stamp("  MPS available")
        return "mps"
    return "cpu"

_DEVICE = _detect_device()
_stamp(f"  Device: {_DEVICE}")

# ── GPU optimizations ────────────────────────────────────────────────────
if _DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')       # TF32 on Ampere+
    _stamp("  GPU opts: cudnn.benchmark + TF32")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Project imports
# ═══════════════════════════════════════════════════════════════════════════════

_stamp("Importing project modules...")

import copy
import numpy as np
from dataclasses import asdict
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import Config
from utils.seed import set_seed
from utils.logger import get_logger
from data.newsgroups import load_newsgroups
from models import FullDistilBERTClassifier as BERTModel, get_tokenizer
from fl.client import FLClient
from fl.server import FLServer
from fl.aggregation import fedavg_aggregate
from attacks.dfr import dfr_attack, estimate_dfr_sigma, plain_free_rider_update
from attacks.sdfr import sdfr_attack
from attacks.afr import (
    afr_attack,
    estimate_decay_rate,
    estimate_e_cos_beta,
    estimate_c_from_norms,
    global_update_norm,
    update_norm,
)
from contribution.shapley import (
    estimate_round_shapley_per_class,
    per_class_to_overall,
    compute_class_metrics,
    _class_weights_from_loader,
)
from detection.utility_score import UtilityScoreTracker
from utils.partition import iid_partition, non_iid_partition
from utils.export import export_results
from utils.runtime import require_cuda, resolve_gpu_profile

_stamp("All imports OK")

logger = get_logger()


# ─── helpers ─────────────────────────────────────────────────────────────────

def create_model(config):
    """Create a full DistilBERT classifier with all parameters trainable."""
    return BERTModel(
        model_name=config.model_name,
        num_classes=config.num_classes,
        model_dir=config.model_dir,
    )


def _apply_attack(attack_type, global_sd, global_history, config,
                  round_index=0, dfr_sigma_est=None, client_id=0,
                  afr_c_est=None):
    """Generate a free-rider update (ALL parameters, full model)."""
    if attack_type == "dfr":
        sigma = config.dfr_sigma_override
        if sigma is None:
            sigma = dfr_sigma_est
        if sigma is None:
            # theta(1)-theta(0) is not observable before the first aggregation.
            return plain_free_rider_update(global_sd), {"dfr_sigma": None}
        update = dfr_attack(global_sd, sigma=sigma,
                            round_num=round_index + 1, gamma=config.dfr_gamma)
        return update, {"dfr_sigma": sigma}
    elif attack_type == "sdfr":
        update = sdfr_attack(global_sd, global_history)
        return update, {}
    elif attack_type == "afr":
        afr_c = config.afr_c_override
        if afr_c is None:
            afr_c = afr_c_est
        if afr_c is None:
            raise RuntimeError("AFR C was not estimated during honest warm-up")
        first_norm = global_update_norm(global_history[1], global_history[0])
        current_norm = global_update_norm(global_sd, global_history[-1])
        decay_rate = estimate_decay_rate(
            current_norm, first_norm, round_num=round_index
        )
        e_cos_beta = estimate_e_cos_beta(
            afr_c, decay_rate, round_num=round_index
        )
        update, base_norm = afr_attack(
            global_sd, global_history,
            n_total=config.num_clients,
            e_cos_beta=e_cos_beta,
            noisy_frac=config.afr_noisy_frac,
            seed=config.seed + round_index * config.num_clients + client_id,
        )
        return update, {
            "afr_base_norm": base_norm,
            "afr_decay_rate": decay_rate,
            "afr_e_cos_beta": e_cos_beta,
        }
    raise ValueError(f"Unknown attack type: {attack_type}")


def _assert_finite_update(update, label: str):
    for key, value in update.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{label} contains non-finite values in {key}")


# ─── single experiment ───────────────────────────────────────────────────────

def run_experiment(config, train_dataset, val_dataset, test_dataset,
                   input_dim, train_labels):
    """Run one full FL experiment and return metrics."""
    set_seed(config.seed)
    logger.info(
        f"=== Experiment: {config.experiment_name} | "
        f"Attack: {config.attack_type} ==="
    )
    if config.attack_type in {"sdfr", "afr"} and config.participation_ratio != 1.0:
        raise ValueError(
            "SDFR/AFR paper alignment requires participation_ratio=1.0"
        )
    if config.attack_warmup_rounds < 2:
        raise ValueError("SDFR/AFR require at least two warm-up rounds")

    # ── data partitioning ────────────────────────────────────────────────
    if config.iid:
        partition = iid_partition(train_dataset, config.num_clients)
    else:
        partition = non_iid_partition(train_labels, config.num_clients)

    client_datasets = {
        cid: Subset(train_dataset, indices)
        for cid, indices in partition.items()
    }
    client_sample_counts = {
        cid: len(indices) for cid, indices in partition.items()
    }

    # ── model / server / clients ─────────────────────────────────────────
    model_fn = lambda: create_model(config)
    model = model_fn()

    server = FLServer(model, val_dataset, test_dataset, config)

    clients = {}
    for cid in range(config.num_clients):
        clients[cid] = FLClient(cid, client_datasets[cid], model_fn, config)

    # ── malicious client ids ─────────────────────────────────────────────
    num_mal = (
        int(config.num_clients * config.malicious_ratio)
        if config.attack_type != "none"
        else 0
    )
    malicious_ids = set(range(num_mal))

    # ── tracking structures ──────────────────────────────────────────────
    utility_tracker = UtilityScoreTracker(alpha=config.utility_alpha)
    round_details: list[dict] = []
    per_class_records: list[dict] = []
    test_accs: list[float] = []
    test_losses: list[float] = []
    participation_counts = {cid: 0 for cid in range(config.num_clients)}
    cumulative_sv = {cid: 0.0 for cid in range(config.num_clients)}
    cumulative_class_var = {cid: [] for cid in range(config.num_clients)}
    cumulative_pos_sum = {cid: [] for cid in range(config.num_clients)}
    cumulative_per_class_sv = {
        cid: np.zeros(config.num_classes, dtype=np.float64)
        for cid in range(config.num_clients)
    }

    # DFR sigma becomes observable after theta(1)-theta(0) exists.
    dfr_sigma_est = None
    afr_c_est = None
    last_warmup_mean_local_norm = None

    val_loader_kwargs = {"batch_size": config.eval_batch_size}
    if config.device == "cuda":
        val_loader_kwargs.update(
            num_workers=config.num_workers,
            pin_memory=True,
        )
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    eval_model = model_fn()
    # torch.compile is intentionally disabled: the evaluation model's full
    # state dict changes for every sampled coalition.
    # if config.device == "cuda":
    #     try:
    #         eval_model = torch.compile(eval_model, mode="reduce-overhead")
    #     except Exception:
    #         pass
    class_weights = _class_weights_from_loader(val_loader, config.num_classes)

    # ── FL rounds ────────────────────────────────────────────────────────
    for round_t in tqdm(range(config.num_rounds), desc=config.experiment_name):
        selected = server.select_clients(
            config.num_clients, config.participation_ratio
        )
        global_sd = server.get_global_state_dict()
        global_history = list(server.global_history)

        if (config.attack_type == "afr" and afr_c_est is None
                and config.afr_c_override is None
                and round_t >= config.attack_warmup_rounds):
            first_norm = global_update_norm(global_history[1], global_history[0])
            current_norm = global_update_norm(global_sd, global_history[-1])
            decay_rate = estimate_decay_rate(
                current_norm, first_norm, round_num=round_t
            )
            afr_c_est = estimate_c_from_norms(
                last_warmup_mean_local_norm,
                current_norm,
                config.num_clients,
                decay_rate,
                round_num=round_t - 1,
            )
            logger.info(f"AFR C estimated from warm-up updates: {afr_c_est:.6f}")

        # ── pre-attack: DFR sigma estimation ─────────────────────────
        if (config.attack_type == "dfr" and dfr_sigma_est is None
                and config.dfr_sigma_override is None):
            est = estimate_dfr_sigma(global_sd, global_history)
            if est is not None:
                dfr_sigma_est = est
                logger.info(f"DFR sigma auto-estimated: {dfr_sigma_est:.6f}")

        # collect updates
        updates = {}
        for cid in selected:
            participation_counts[cid] += 1
            needs_honest_warmup = (
                config.attack_type in {"sdfr", "afr"}
                and round_t < config.attack_warmup_rounds
            )
            if cid in malicious_ids and not needs_honest_warmup:
                update, meta = _apply_attack(
                    config.attack_type, global_sd, global_history, config,
                    round_index=round_t,
                    dfr_sigma_est=dfr_sigma_est,
                    client_id=cid,
                    afr_c_est=afr_c_est,
                )
                updates[cid] = update
            else:
                updates[cid] = clients[cid].train(global_sd)
            _assert_finite_update(updates[cid], f"client {cid} update")

        if config.attack_type == "afr" and needs_honest_warmup:
            last_warmup_mean_local_norm = float(np.mean([
                update_norm(updates[cid]) for cid in selected
            ]))

        # ── per-class Shapley estimation ─────────────────────────────────
        round_sample_counts = {cid: client_sample_counts[cid] for cid in selected}
        per_class_sv = estimate_round_shapley_per_class(
            eval_model, updates, global_sd, val_loader,
            num_classes=config.num_classes,
            num_mc_samples=config.num_mc_samples,
            device=config.device,
            sample_counts=round_sample_counts,
        )

        shapley_vals = per_class_to_overall(per_class_sv, class_weights)
        class_metrics = compute_class_metrics(per_class_sv)

        # ── utility scores ───────────────────────────────────────────────
        utility_tracker.update(shapley_vals)

        # ── standard sample-size-weighted FedAvg ────────────────────────
        aggregation_weights = {
            cid: client_sample_counts[cid] for cid in selected
        }
        new_sd = fedavg_aggregate(
            global_sd, updates, weights=aggregation_weights,
        )
        _assert_finite_update(new_sd, "aggregated global state")
        server.update_global_model(new_sd)

        # ── evaluate ─────────────────────────────────────────────────────
        test_loss, test_acc = server.evaluate()
        test_accs.append(test_acc)
        test_losses.append(test_loss)

        logger.info(
            f"Round {round_t:>2d}: acc={test_acc:.4f}  loss={test_loss:.4f}  "
            f"selected={selected}"
        )

        # ── record per-client details ────────────────────────────────────
        for cid in selected:
            sv = shapley_vals.get(cid, 0.0)
            cumulative_sv[cid] += sv
            pc = participation_counts[cid]

            cm = class_metrics.get(cid, {})
            cls_var = cm.get("class_sv_variance", 0.0)
            pos_sum = cm.get("positive_class_sv_sum", 0.0)
            conc_ratio = cm.get("concentration_ratio", 0.0)
            dom_class = cm.get("dominant_class", -1)
            cumulative_class_var[cid].append(cls_var)
            cumulative_pos_sum[cid].append(pos_sum)

            round_details.append({
                "experiment_name": config.experiment_name,
                "attack_type": config.attack_type,
                "round": round_t,
                "client_id": cid,
                "is_malicious": cid in malicious_ids,
                "participation_count_so_far": pc,
                "round_shapley_value": sv,
                "cumulative_shapley_value": cumulative_sv[cid],
                "mean_shapley_value": cumulative_sv[cid] / pc,
                "class_sv_variance": cls_var,
                "positive_class_sv_sum": pos_sum,
                "concentration_ratio": conc_ratio,
                "dominant_class": dom_class,
                "mean_class_sv_variance": float(np.mean(cumulative_class_var[cid])),
                "mean_positive_class_sv_sum": float(np.mean(cumulative_pos_sum[cid])),
                "utility_score": utility_tracker.scores.get(cid, 0.0),
            })

            # Accumulate per-class SV for final fingerprint
            if cid in per_class_sv:
                cumulative_per_class_sv[cid] += per_class_sv[cid]

            # Record raw per-class SV for this round
            pc_arr = per_class_sv.get(cid, np.zeros(config.num_classes))
            rec = {
                "experiment_name": config.experiment_name,
                "round": round_t,
                "client_id": cid,
                "is_malicious": cid in malicious_ids,
            }
            for c in range(config.num_classes):
                rec[f"class_{c}"] = float(pc_arr[c])
            per_class_records.append(rec)

    # ── build experiment summary ─────────────────────────────────────────
    honest_ids = [c for c in range(config.num_clients) if c not in malicious_ids]

    def _avg_mean_sv(ids):
        vals = [
            cumulative_sv[c] / max(participation_counts[c], 1) for c in ids
        ]
        return float(np.mean(vals)) if vals else 0.0

    def _avg_cum_sv(ids):
        return float(np.mean([cumulative_sv[c] for c in ids])) if ids else 0.0

    def _avg_metric(ids, store):
        vals = [float(np.mean(store[c])) if store[c] else 0.0 for c in ids]
        return float(np.mean(vals)) if vals else 0.0

    avg_sv_h = _avg_mean_sv(honest_ids)
    avg_sv_m = _avg_mean_sv(list(malicious_ids))
    avg_cum_h = _avg_cum_sv(honest_ids)
    avg_cum_m = _avg_cum_sv(list(malicious_ids))

    summary = {
        "experiment_name": config.experiment_name,
        "attack_type": config.attack_type,
        "malicious_ratio": config.malicious_ratio if num_mal > 0 else 0.0,
        "final_global_accuracy": test_accs[-1],
        "final_global_loss": test_losses[-1],
        "avg_round_shapley_honest": avg_sv_h,
        "avg_round_shapley_malicious": avg_sv_m,
        "avg_cumulative_shapley_honest": avg_cum_h,
        "avg_cumulative_shapley_malicious": avg_cum_m,
        "shapley_gap_honest_vs_malicious": avg_sv_h - avg_sv_m,
        "avg_class_sv_variance_honest": _avg_metric(honest_ids, cumulative_class_var),
        "avg_class_sv_variance_malicious": _avg_metric(list(malicious_ids), cumulative_class_var),
        "avg_positive_class_sv_sum_honest": _avg_metric(honest_ids, cumulative_pos_sum),
        "avg_positive_class_sv_sum_malicious": _avg_metric(list(malicious_ids), cumulative_pos_sum),
        "attack_effective": "",
        "notes": "",
    }

    return (round_details, summary, test_accs, test_losses,
            per_class_records, cumulative_per_class_sv)


# ─── main ────────────────────────────────────────────────────────────────────
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Full-DistilBERT federated free-rider experiments"
    )
    parser.add_argument(
        "--gpu-profile",
        choices=("auto", "t4", "large"),
        default="auto",
        help="GPU memory profile; auto selects from detected VRAM",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail immediately unless an NVIDIA CUDA device is available",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="directory for Excel results and plots",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    require_cuda(args.require_cuda, _DEVICE)

    gpu_name = ""
    gpu_memory_gb = 0.0
    resolved_profile = None
    if _DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )
        resolved_profile = resolve_gpu_profile(args.gpu_profile, gpu_memory_gb)

    base = Config(
        model_dir=os.environ.get("MODEL_DIR", ""),
        num_rounds=50,
        local_epochs=2,
        num_clients=10,
        local_lr=2e-5,
        participation_ratio=1.0,
        batch_size=resolved_profile.train_batch_size if resolved_profile else 8,
        eval_batch_size=resolved_profile.eval_batch_size if resolved_profile else 64,
        num_workers=resolved_profile.num_workers if resolved_profile else 0,
        num_classes=20,
        val_ratio=0.1,
        iid=True,
        malicious_ratio=0.4,
        num_mc_samples=15,
        seed=42,
        results_dir=args.results_dir,
        gpu_profile=resolved_profile.name if resolved_profile else "cpu",
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )
    base.device = _DEVICE  # use the safely-detected device
    _stamp(f"Config OK — device: {base.device}")

    logger.info(f"Using device: {base.device}")
    if base.device == "cuda":
        amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        logger.info(
            f"GPU: {base.gpu_name} | VRAM: {base.gpu_memory_gb:.1f} GB | "
            f"profile={base.gpu_profile} | train_batch={base.batch_size} | "
            f"eval_batch={base.eval_batch_size} | workers={base.num_workers} | "
            f"AMP={amp_dtype}"
        )

    # ── load data once ───────────────────────────────────────────────────
    _stamp("Loading 20 Newsgroups data...")
    set_seed(base.seed)
    train_ds, val_ds, test_ds, input_dim, train_labels = load_newsgroups(
        model_name=base.model_name,
        val_ratio=base.val_ratio,
        max_seq_length=base.max_seq_length,
        model_dir=base.model_dir,
    )
    _stamp(f"Data loaded — train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    logger.info(
        f"Data loaded — train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    _stamp("Starting experiments: baseline, DFR, SDFR, AFR")

    # ── run experiments ──────────────────────────────────────────────────
    experiments = [
        ("baseline_no_attack", "none"),
        ("attack_dfr", "dfr"),
        ("attack_sdfr", "sdfr"),
        ("attack_afr", "afr"),
    ]

    all_details: list[dict] = []
    all_summaries: list[dict] = []
    all_curves: dict = {}
    all_pc_records: list[dict] = []
    all_cum_pc_sv: dict[str, dict] = {}

    for exp_name, attack_type in experiments:
        cfg = copy.deepcopy(base)
        cfg.experiment_name = exp_name
        cfg.attack_type = attack_type
        cfg.device = _DEVICE

        _stamp(f"  Running: {exp_name}")
        (details, summary, accs, losses,
         pc_records, cum_pc_sv) = run_experiment(
            cfg, train_ds, val_ds, test_ds, input_dim, train_labels
        )

        all_details.extend(details)
        all_summaries.append(summary)
        all_curves[exp_name] = {"acc": accs, "loss": losses}
        all_pc_records.extend(pc_records)
        all_cum_pc_sv[exp_name] = cum_pc_sv
        _stamp(f"  Done: {exp_name} — final acc={summary['final_global_accuracy']:.4f}")
        gc.collect()
        if base.device == "cuda":
            torch.cuda.empty_cache()

    # ── mark attack effectiveness (relative to baseline) ─────────────────
    baseline_acc = all_summaries[0]["final_global_accuracy"]
    for s in all_summaries[1:]:
        drop = baseline_acc - s["final_global_accuracy"]
        s["attack_effective"] = drop > 0.02
        s["notes"] = f"accuracy drop vs baseline: {drop:.4f}"

    # ── export ───────────────────────────────────────────────────────────
    os.makedirs(base.results_dir, exist_ok=True)
    filepath = export_results(all_details, all_summaries, base.results_dir,
                              per_class_records=all_pc_records,
                              experiment_config=asdict(base))
    logger.info(f"Excel results exported to {filepath}")
    _stamp(f"Excel exported: {filepath}")

    # ── plots ────────────────────────────────────────────────────────────
    from visualization.plots import generate_plots
    from sklearn.datasets import fetch_20newsgroups
    _raw = fetch_20newsgroups(subset="train")
    class_names = [n.split(".")[-1][:10] for n in _raw.target_names]

    plots_dir = os.path.join(base.results_dir, "plots")
    generate_plots(all_curves, all_details, all_pc_records, all_cum_pc_sv,
                   base.experiment_name, class_names, plots_dir)
    logger.info(f"Plots saved to {plots_dir}/")
    _stamp(f"Plots saved: {plots_dir}/")

    # ── print summary table ──────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)
    for s in all_summaries:
        print(
            f"  {s['experiment_name']:<22s}  "
            f"acc={s['final_global_accuracy']:.4f}  "
            f"loss={s['final_global_loss']:.4f}  "
            f"SV_h={s['avg_round_shapley_honest']:.6f}  "
            f"SV_m={s['avg_round_shapley_malicious']:.6f}  "
            f"Var_h={s.get('avg_class_sv_variance_honest',0):.2e}  "
            f"Var_m={s.get('avg_class_sv_variance_malicious',0):.2e}  "
            f"PosSum_h={s.get('avg_positive_class_sv_sum_honest',0):.6f}  "
            f"PosSum_m={s.get('avg_positive_class_sv_sum_malicious',0):.6f}"
        )
    print("=" * 100)

    _stamp("All experiments complete")


if __name__ == "__main__":
    main()
