#!/usr/bin/env python3
"""
Federated Learning on 20 Newsgroups — DistilBERT + LoRA
Free-Rider Attacks + Per-Class Shapley Detection.

Runs four experiments:
  1. baseline (no attack)
  2. DFR  (Disguised Free-Rider)
  3. SDFR (Scaled Delta Free-Rider)
  4. AFR  (Advanced Free-Rider)

Outputs:
  results/experiment_results.xlsx
  results/plots/fig_01 … fig_10 (10 chart groups)

Designed for Google Colab / public cloud — no Chinese mirrors,
no local pip cache, no cloud-specific workarounds.
"""

import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ── Minimal env tuning (safe everywhere) ────────────────────────────────
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── 抑制 HuggingFace transformers 模型加载时的 LOAD REPORT 刷屏 ──────
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import logging as _logging
_logging.getLogger("transformers").setLevel(_logging.ERROR)
_logging.getLogger("transformers.modeling_utils").setLevel(_logging.ERROR)

# ── HuggingFace token（从环境变量读取，不写死在代码里）────────────────
# Colab 用户：在 Secrets 里设置 HF_TOKEN，代码自动读取
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])
    _logging.getLogger("transformers").info("HF_TOKEN loaded from environment")

from config import Config
from utils.seed import set_seed
from utils.logger import get_logger
from data.newsgroups import load_newsgroups
from models.lora_classifier import DistilBERTWithLoRA
from fl.client import FLClient
from fl.server import FLServer
from fl.aggregation import fedavg_aggregate
from attacks.dfr import dfr_attack, estimate_dfr_sigma
from attacks.sdfr import sdfr_attack
from attacks.afr import afr_attack, AFRState
from contribution.shapley import (
    estimate_round_shapley_per_class,
    per_class_to_overall,
    compute_class_metrics,
    _class_weights_from_loader,
)
from detection.utility_score import UtilityScoreTracker
from utils.partition import iid_partition, non_iid_partition
from utils.export import export_results

logger = get_logger()


# ─── helpers ─────────────────────────────────────────────────────────────────

def create_model(input_dim, config):
    """Create a DistilBERTWithLoRA model."""
    return DistilBERTWithLoRA(
        model_name=config.model_name,
        num_classes=config.num_classes,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        model_dir=config.model_dir,
    ).to(config.device)


def _apply_attack(attack_type, global_sd, global_history, config,
                  round_num=1, dfr_sigma_est=None,
                  afr_state=None, val_loss_t=None,
                  trainable_keys=None):
    """Generate a free-rider update (trainable params only)."""
    if attack_type == "dfr":
        sigma = config.dfr_sigma
        if config.dfr_estimate_sigma and dfr_sigma_est is not None:
            sigma = dfr_sigma_est
        update = dfr_attack(global_sd, sigma=sigma,
                            round_num=round_num, gamma=config.dfr_gamma,
                            trainable_keys=trainable_keys)
        return update, {}

    elif attack_type == "sdfr":
        update = sdfr_attack(global_sd, global_history,
                              trainable_keys=trainable_keys)
        return update, {}

    elif attack_type == "afr":
        e_cos_beta = 0.0
        if config.afr_e_cos_beta_override is not None:
            e_cos_beta = config.afr_e_cos_beta_override
        elif afr_state is not None and val_loss_t is not None:
            e_cos_beta = afr_state.get_e_cos_beta(val_loss_t)
        mean_base_norm = None
        if afr_state is not None:
            mean_base_norm = afr_state.get_mean_base_norm()
        update, base_norm = afr_attack(
            global_sd, global_history,
            n_total=config.num_clients,
            e_cos_beta=e_cos_beta,
            mean_base_norm=mean_base_norm,
            noisy_frac=config.afr_noisy_frac,
            trainable_keys=trainable_keys,
        )
        return update, {"afr_base_norm": base_norm}

    raise ValueError(f"Unknown attack type: {attack_type}")


# ─── single experiment ───────────────────────────────────────────────────────

def run_experiment(config, train_dataset, val_dataset, test_dataset,
                   input_dim, train_labels):
    """Run one full FL experiment and return metrics."""
    set_seed(config.seed)
    logger.info(
        f"=== Experiment: {config.experiment_name} | "
        f"Attack: {config.attack_type} ==="
    )

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
    model_fn = lambda: create_model(input_dim, config)
    model = model_fn()
    trainable_keys = model.trainable_keys

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

    # ── attack state trackers ───────────────────────────────────────────
    dfr_sigma_est = None
    afr_state = AFRState(ema_alpha=config.afr_base_norm_ema_alpha) if config.attack_type == "afr" else None

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    eval_model = model_fn()
    # torch.compile 在 A100 上对重复推理（Shapley 240 次/轮）提效显著
    if config.device == "cuda":
        try:
            eval_model = torch.compile(eval_model, mode="reduce-overhead")
        except Exception:
            pass  # 旧版 PyTorch 可能不支持，静默降级
    class_weights = _class_weights_from_loader(val_loader, config.num_classes)

    # ── FL rounds ────────────────────────────────────────────────────────
    for round_t in tqdm(range(config.num_rounds), desc=config.experiment_name):
        config.local_lr = config.get_round_local_lr(round_t)

        selected = server.select_clients(
            config.num_clients, config.participation_ratio
        )
        global_sd = server.get_global_state_dict()
        global_history = list(server.global_history)

        # ── pre-attack: DFR sigma estimation ─────────────────────────
        if config.attack_type == "dfr" and dfr_sigma_est is None and config.dfr_estimate_sigma:
            est = estimate_dfr_sigma(global_sd, global_history, trainable_keys=trainable_keys)
            if est is not None:
                dfr_sigma_est = est
                logger.info(f"DFR sigma auto-estimated: {dfr_sigma_est:.6f}")

        # ── pre-attack: validation loss for AFR state ────────────────────
        val_loss_t = None
        if config.attack_type == "afr":
            val_loss_t, _ = server.evaluate_val()
            if afr_state is not None and afr_state.val_loss_init is None:
                afr_state.val_loss_init = val_loss_t
                logger.info(f"AFR val_loss_init set: {val_loss_t:.4f}")

        # collect updates
        updates = {}
        afr_base_norms_this_round = []
        for cid in selected:
            participation_counts[cid] += 1
            if cid in malicious_ids:
                update, meta = _apply_attack(
                    config.attack_type, global_sd, global_history, config,
                    round_num=round_t + 1,
                    dfr_sigma_est=dfr_sigma_est,
                    afr_state=afr_state,
                    val_loss_t=val_loss_t,
                    trainable_keys=trainable_keys,
                )
                updates[cid] = update
                if "afr_base_norm" in meta:
                    afr_base_norms_this_round.append(meta["afr_base_norm"])
            else:
                updates[cid] = clients[cid].train(global_sd)

        # ── post-attack: update AFR state ────────────────────────────────
        if afr_state is not None and val_loss_t is not None and afr_base_norms_this_round:
            avg_base_norm = float(np.mean(afr_base_norms_this_round))
            afr_state.update(val_loss_t, avg_base_norm)

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

        # ── aggregate (FedAvg with optional server momentum) ────────────
        new_sd, server.momentum_buffer = fedavg_aggregate(
            global_sd, updates,
            server_lr=server.current_server_lr,
            momentum=config.server_momentum,
            momentum_buffer=server.momentum_buffer,
        )
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

            if cid in per_class_sv:
                cumulative_per_class_sv[cid] += per_class_sv[cid]

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

def main():
    # ── A100 最大化利用 ──────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')     # TF32
        # A100 优先用 bf16（同 fp32 动态范围，比 fp16 稳）
        torch.cuda.amp.autocast.__default_dtype__ = torch.bfloat16
        logger.info("A100 optimizations: cudnn.benchmark + TF32 + bf16 autocast")
    # ─────────────────────────────────────────────────────────────────

    base = Config(
        num_clients=10,
        num_rounds=50,
        local_epochs=2,
        local_lr=0.0005,
        server_lr=0.5,
        participation_ratio=0.8,
        batch_size=512,
        num_classes=20,
        val_ratio=0.1,
        iid=True,
        malicious_ratio=0.4,
        num_mc_samples=30,
        seed=42,
        results_dir="results",
        # device is auto-detected via Config default_factory
    )
    logger.info(f"Using device: {base.device}")
    if base.device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── load data once ───────────────────────────────────────────────────
    set_seed(base.seed)
    logger.info("Downloading and loading 20 Newsgroups dataset...")
    train_ds, val_ds, test_ds, input_dim, train_labels = load_newsgroups(
        model_name=base.model_name,
        val_ratio=base.val_ratio,
        max_seq_length=base.max_seq_length,
    )
    logger.info(
        f"Data loaded — train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

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

        (details, summary, accs, losses,
         pc_records, cum_pc_sv) = run_experiment(
            cfg, train_ds, val_ds, test_ds, input_dim, train_labels
        )

        all_details.extend(details)
        all_summaries.append(summary)
        all_curves[exp_name] = {"acc": accs, "loss": losses}
        all_pc_records.extend(pc_records)
        all_cum_pc_sv[exp_name] = cum_pc_sv

    # ── mark attack effectiveness (relative to baseline) ─────────────────
    baseline_acc = all_summaries[0]["final_global_accuracy"]
    for s in all_summaries[1:]:
        drop = baseline_acc - s["final_global_accuracy"]
        s["attack_effective"] = drop > 0.02
        s["notes"] = f"accuracy drop vs baseline: {drop:.4f}"

    # ── export ───────────────────────────────────────────────────────────
    os.makedirs(base.results_dir, exist_ok=True)
    filepath = export_results(all_details, all_summaries, base.results_dir,
                              per_class_records=all_pc_records)
    logger.info(f"Excel results exported to {filepath}")

    # ── plots ────────────────────────────────────────────────────────────
    from visualization.plots import generate_plots
    from sklearn.datasets import fetch_20newsgroups
    _raw = fetch_20newsgroups(subset="train")
    class_names = [n.split(".")[-1][:10] for n in _raw.target_names]

    plots_dir = os.path.join(base.results_dir, "plots")
    generate_plots(all_curves, all_details, all_pc_records, all_cum_pc_sv,
                   base.experiment_name, class_names, plots_dir)
    logger.info(f"Plots saved to {plots_dir}/")

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


if __name__ == "__main__":
    main()
