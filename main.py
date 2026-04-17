#!/usr/bin/env python3
"""
Federated Learning on 20 Newsgroups with Free-Rider Attacks,
Per-Class Shapley-Value Contribution Estimation, and Detection.

Runs four experiments:
  1. baseline (no attack)
  2. DFR  (Disguised Free-Rider)
  3. SDFR (Scaled Delta Free-Rider)
  4. AFR  (Advanced Free-Rider)

Detection metrics (per-class Shapley derived):
  - class_sv_variance:      variance of per-class SV across 20 classes
  - positive_class_sv_sum:  sum of positive per-class SV values

Outputs:
  results/experiment_results.xlsx   — per-round Shapley details + summary
  results/test_accuracy.png         — accuracy curves
  results/test_loss.png             — loss curves
  results/shapley_comparison.png    — Shapley honest-vs-malicious
  results/class_metrics.png         — per-class variance & positive sum
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import Config
from utils.seed import set_seed
from utils.logger import get_logger
from data.newsgroups import load_newsgroups
from models.mlp import MLP
from fl.client import FLClient
from fl.server import FLServer
from fl.aggregation import fedavg_aggregate
from attacks.dfr import dfr_attack
from attacks.sdfr import sdfr_attack
from attacks.afr import afr_attack
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
    return MLP(input_dim, config.hidden_dim, config.num_classes).to(config.device)


def _apply_attack(attack_type, global_sd, global_history, config,
                  round_num=1, n_participants=8):
    """Generate a free-rider update according to attack_type."""
    if attack_type == "dfr":
        return dfr_attack(global_sd, sigma=config.dfr_sigma,
                          round_num=round_num, gamma=config.dfr_gamma)
    elif attack_type == "sdfr":
        return sdfr_attack(global_sd, global_history)
    elif attack_type == "afr":
        return afr_attack(global_sd, global_history,
                          n_participants=n_participants,
                          e_cos_beta=config.afr_e_cos_beta,
                          noisy_frac=config.afr_noisy_frac)
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

    # ── model / server / clients ─────────────────────────────────────────
    model_fn = lambda: create_model(input_dim, config)
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
    test_accs: list[float] = []
    test_losses: list[float] = []
    participation_counts = {cid: 0 for cid in range(config.num_clients)}
    cumulative_sv = {cid: 0.0 for cid in range(config.num_clients)}
    cumulative_class_var = {cid: [] for cid in range(config.num_clients)}
    cumulative_pos_sum = {cid: [] for cid in range(config.num_clients)}

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    eval_model = model_fn()
    class_weights = _class_weights_from_loader(val_loader, config.num_classes)

    # ── FL rounds ────────────────────────────────────────────────────────
    for round_t in tqdm(range(config.num_rounds), desc=config.experiment_name):
        selected = server.select_clients(
            config.num_clients, config.participation_ratio
        )
        global_sd = server.get_global_state_dict()
        global_history = list(server.global_history)

        # collect updates
        updates = {}
        for cid in selected:
            participation_counts[cid] += 1
            if cid in malicious_ids:
                updates[cid] = _apply_attack(
                    config.attack_type, global_sd, global_history, config,
                    round_num=round_t + 1, n_participants=len(selected),
                )
            else:
                updates[cid] = clients[cid].train(global_sd)

        # ── per-class Shapley estimation ─────────────────────────────────
        per_class_sv = estimate_round_shapley_per_class(
            eval_model, updates, global_sd, val_loader,
            num_classes=config.num_classes,
            server_lr=config.server_lr,
            num_mc_samples=config.num_mc_samples,
            device=config.device,
        )

        # derive overall Shapley and per-class metrics
        shapley_vals = per_class_to_overall(per_class_sv, class_weights)
        class_metrics = compute_class_metrics(per_class_sv)

        # ── utility scores ───────────────────────────────────────────────
        utility_tracker.update(shapley_vals)

        # ── aggregate (standard FedAvg) ──────────────────────────────────
        new_sd = fedavg_aggregate(global_sd, updates, config.server_lr)
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
                "mean_class_sv_variance": float(np.mean(cumulative_class_var[cid])),
                "mean_positive_class_sv_sum": float(np.mean(cumulative_pos_sum[cid])),
                "utility_score": utility_tracker.scores.get(cid, 0.0),
            })

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

    return round_details, summary, test_accs, test_losses


# ─── plotting ────────────────────────────────────────────────────────────────

def generate_plots(curves: dict, round_details: list, results_dir: str):
    """Generate and save all experiment plots."""
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(round_details)
    attack_names = ["attack_dfr", "attack_sdfr", "attack_afr"]

    # ── 1. test accuracy ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in curves.items():
        ax.plot(data["acc"], label=name)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Global Test Accuracy Over Rounds")
    ax.legend()
    ax.grid(True)
    fig.savefig(
        os.path.join(results_dir, "test_accuracy.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # ── 2. test loss ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in curves.items():
        ax.plot(data["loss"], label=name)
    ax.set_xlabel("Round")
    ax.set_ylabel("Test Loss")
    ax.set_title("Global Test Loss Over Rounds")
    ax.legend()
    ax.grid(True)
    fig.savefig(
        os.path.join(results_dir, "test_loss.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # ── 3. shapley comparison ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for exp_name in attack_names:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        honest = sub[~sub["is_malicious"]].groupby("round")["round_shapley_value"].mean()
        mal = sub[sub["is_malicious"]].groupby("round")["round_shapley_value"].mean()
        label_short = exp_name.replace("attack_", "").upper()
        axes[0].plot(honest.index, honest.values, label=f"{label_short} honest")
        axes[0].plot(
            mal.index, mal.values, linestyle="--", label=f"{label_short} malicious"
        )
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Mean Round Shapley Value")
    axes[0].set_title("Round-Level Shapley: Honest vs Malicious")
    axes[0].legend(fontsize=8)
    axes[0].axhline(y=0, color="k", linewidth=0.5)
    axes[0].grid(True)

    bar_labels, bar_vals, bar_colors = [], [], []
    for exp_name in attack_names:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        last_round = sub["round"].max()
        last = sub[sub["round"] == last_round]
        label_short = exp_name.replace("attack_", "").upper()

        h_cum = last[~last["is_malicious"]]["cumulative_shapley_value"].mean()
        m_cum = last[last["is_malicious"]]["cumulative_shapley_value"].mean()

        bar_labels += [f"{label_short}\nhonest", f"{label_short}\nmalicious"]
        bar_vals += [h_cum, m_cum]
        bar_colors += ["steelblue", "coral"]

    if bar_labels:
        x_pos = np.arange(len(bar_labels))
        axes[1].bar(x_pos, bar_vals, color=bar_colors, alpha=0.8)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(bar_labels, fontsize=8)
    axes[1].set_ylabel("Avg Cumulative Shapley Value")
    axes[1].set_title("Cumulative Shapley: Honest vs Malicious")
    axes[1].axhline(y=0, color="k", linewidth=0.5)
    axes[1].grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(
        os.path.join(results_dir, "shapley_comparison.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)

    # ── 4. per-class metrics: variance & positive sum ────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 4a: class_sv_variance over rounds
    for exp_name in attack_names:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        honest = sub[~sub["is_malicious"]].groupby("round")["class_sv_variance"].mean()
        mal = sub[sub["is_malicious"]].groupby("round")["class_sv_variance"].mean()
        label = exp_name.replace("attack_", "").upper()
        axes[0, 0].plot(honest.index, honest.values, label=f"{label} honest")
        axes[0, 0].plot(mal.index, mal.values, ls="--", label=f"{label} malicious")
    axes[0, 0].set_xlabel("Round")
    axes[0, 0].set_ylabel("Per-Class SV Variance")
    axes[0, 0].set_title("Per-Class Shapley Variance Over Rounds")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True)

    # 4b: positive_class_sv_sum over rounds
    for exp_name in attack_names:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        honest = sub[~sub["is_malicious"]].groupby("round")["positive_class_sv_sum"].mean()
        mal = sub[sub["is_malicious"]].groupby("round")["positive_class_sv_sum"].mean()
        label = exp_name.replace("attack_", "").upper()
        axes[0, 1].plot(honest.index, honest.values, label=f"{label} honest")
        axes[0, 1].plot(mal.index, mal.values, ls="--", label=f"{label} malicious")
    axes[0, 1].set_xlabel("Round")
    axes[0, 1].set_ylabel("Positive Per-Class SV Sum")
    axes[0, 1].set_title("Positive Per-Class SV Sum Over Rounds")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True)

    # 4c: bar chart of mean class_sv_variance
    bar_labels, bar_vals, bar_colors = [], [], []
    for exp_name in attack_names:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        last_round = sub["round"].max()
        last = sub[sub["round"] == last_round]
        label = exp_name.replace("attack_", "").upper()
        h_v = last[~last["is_malicious"]]["mean_class_sv_variance"].mean()
        m_v = last[last["is_malicious"]]["mean_class_sv_variance"].mean()
        bar_labels += [f"{label}\nhonest", f"{label}\nmalicious"]
        bar_vals += [h_v, m_v]
        bar_colors += ["steelblue", "coral"]
    if bar_labels:
        x_pos = np.arange(len(bar_labels))
        axes[1, 0].bar(x_pos, bar_vals, color=bar_colors, alpha=0.8)
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(bar_labels, fontsize=8)
    axes[1, 0].set_ylabel("Mean Per-Class SV Variance")
    axes[1, 0].set_title("Avg Per-Class SV Variance: Honest vs Malicious")
    axes[1, 0].grid(True, axis="y")

    # 4d: bar chart of mean positive_class_sv_sum
    bar_labels, bar_vals, bar_colors = [], [], []
    for exp_name in attack_names:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        last_round = sub["round"].max()
        last = sub[sub["round"] == last_round]
        label = exp_name.replace("attack_", "").upper()
        h_p = last[~last["is_malicious"]]["mean_positive_class_sv_sum"].mean()
        m_p = last[last["is_malicious"]]["mean_positive_class_sv_sum"].mean()
        bar_labels += [f"{label}\nhonest", f"{label}\nmalicious"]
        bar_vals += [h_p, m_p]
        bar_colors += ["steelblue", "coral"]
    if bar_labels:
        x_pos = np.arange(len(bar_labels))
        axes[1, 1].bar(x_pos, bar_vals, color=bar_colors, alpha=0.8)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(bar_labels, fontsize=8)
    axes[1, 1].set_ylabel("Mean Positive Per-Class SV Sum")
    axes[1, 1].set_title("Avg Positive Per-Class SV Sum: Honest vs Malicious")
    axes[1, 1].grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(
        os.path.join(results_dir, "class_metrics.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    base = Config(
        num_clients=10,
        num_rounds=30,
        local_epochs=3,
        local_lr=0.001,
        server_lr=1.0,
        participation_ratio=0.8,
        batch_size=64,
        hidden_dim=256,
        num_classes=20,
        max_features=10000,
        val_ratio=0.1,
        iid=True,
        malicious_ratio=0.4,
        num_mc_samples=30,
        seed=42,
        device="cpu",
        results_dir="results",
    )

    # ── load data once ───────────────────────────────────────────────────
    set_seed(base.seed)
    train_ds, val_ds, test_ds, input_dim, train_labels = load_newsgroups(
        max_features=base.max_features, val_ratio=base.val_ratio
    )
    logger.info(
        f"Data loaded — input_dim={input_dim}, "
        f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
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

    for exp_name, attack_type in experiments:
        cfg = copy.deepcopy(base)
        cfg.experiment_name = exp_name
        cfg.attack_type = attack_type

        details, summary, accs, losses = run_experiment(
            cfg, train_ds, val_ds, test_ds, input_dim, train_labels
        )

        all_details.extend(details)
        all_summaries.append(summary)
        all_curves[exp_name] = {"acc": accs, "loss": losses}

    # ── mark attack effectiveness (relative to baseline) ─────────────────
    baseline_acc = all_summaries[0]["final_global_accuracy"]
    for s in all_summaries[1:]:
        drop = baseline_acc - s["final_global_accuracy"]
        s["attack_effective"] = drop > 0.02
        s["notes"] = f"accuracy drop vs baseline: {drop:.4f}"

    # ── export ───────────────────────────────────────────────────────────
    os.makedirs(base.results_dir, exist_ok=True)
    filepath = export_results(all_details, all_summaries, base.results_dir)
    logger.info(f"Excel results exported to {filepath}")

    generate_plots(all_curves, all_details, base.results_dir)
    logger.info(f"Plots saved to {base.results_dir}/")

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
