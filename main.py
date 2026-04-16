#!/usr/bin/env python3
"""
Federated Learning on 20 Newsgroups with Free-Rider Attacks,
Shapley-Value Contribution Estimation (SVRFL-style), and Detection.

Runs four experiments:
  1. baseline (no attack)
  2. DFR  (Disguised Free-Rider)
  3. SDFR (Scaled Delta Free-Rider)
  4. AFR  (Advanced Free-Rider)

Outputs:
  results/experiment_results.xlsx   — per-round Shapley details + summary
  results/test_accuracy.png         — accuracy curves
  results/test_loss.png             — loss curves
  results/shapley_comparison.png    — Shapley honest-vs-malicious
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid KMeans crash on macOS

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
from contribution.shapley import estimate_round_shapley
from detection.free_rider_detection import compute_feature_values, detect_free_riders, detect_afr_by_delta_similarity
from detection.utility_score import UtilityScoreTracker
from utils.partition import iid_partition, non_iid_partition
from utils.export import export_results

logger = get_logger()


# ─── helpers ─────────────────────────────────────────────────────────────────

def create_model(input_dim, config):
    return MLP(input_dim, config.hidden_dim, config.num_classes).to(config.device)


def _apply_attack(attack_type, global_sd, prev_sd, config):
    """Generate a free-rider update according to attack_type."""
    if attack_type == "dfr":
        return dfr_attack(global_sd, config.dfr_noise_scale)
    elif attack_type == "sdfr":
        return sdfr_attack(global_sd, prev_sd, config.sdfr_scale)
    elif attack_type == "afr":
        return afr_attack(global_sd, prev_sd, config.afr_noise_scale)
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

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    eval_model = model_fn()  # re-used for Shapley evaluation

    # ── FL rounds ────────────────────────────────────────────────────────
    for round_t in tqdm(range(config.num_rounds), desc=config.experiment_name):
        selected = server.select_clients(
            config.num_clients, config.participation_ratio
        )
        global_sd = server.get_global_state_dict()
        prev_sd = server.prev_global_state_dict

        # collect updates
        updates = {}
        for cid in selected:
            participation_counts[cid] += 1
            if cid in malicious_ids:
                updates[cid] = _apply_attack(
                    config.attack_type, global_sd, prev_sd, config
                )
            else:
                updates[cid] = clients[cid].train(global_sd)

        # ── Shapley estimation ───────────────────────────────────────────
        shapley_vals = estimate_round_shapley(
            eval_model, updates, global_sd, val_loader,
            server_lr=config.server_lr,
            num_mc_samples=config.num_mc_samples,
            device=config.device,
        )

        # ── utility scores ───────────────────────────────────────────────
        utility_tracker.update(shapley_vals)

        # ── feature-value detection ──────────────────────────────────────
        feat_vals = compute_feature_values(updates, shapley_vals, global_sd)
        suspected = detect_free_riders(feat_vals, config.detection_threshold_h)

        # ── AFR detection via delta cosine similarity ────────────────────
        if prev_sd is not None:
            prev_delta = {k: global_sd[k] - prev_sd[k] for k in global_sd}
            afr_suspected = detect_afr_by_delta_similarity(
                updates, prev_delta, config.afr_cosine_threshold
            )
            for cid, flag in afr_suspected.items():
                if flag:
                    suspected[cid] = True

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
                "free_rider_flag": suspected.get(cid, False),
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
        "attack_effective": "",  # filled in post-hoc
        "notes": "",
    }

    return round_details, summary, test_accs, test_losses


# ─── plotting ────────────────────────────────────────────────────────────────

def generate_plots(curves: dict, round_details: list, results_dir: str):
    """Generate and save all experiment plots."""
    os.makedirs(results_dir, exist_ok=True)

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
    df = pd.DataFrame(round_details)
    attack_names = ["attack_dfr", "attack_sdfr", "attack_afr"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 3a: round shapley over time
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
    axes[0].grid(True)

    # 3b: cumulative shapley bar chart
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
    axes[1].grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(
        os.path.join(results_dir, "shapley_comparison.png"),
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
        iid=False,
        detection_threshold_h=20.0,
        afr_cosine_threshold=0.9,
        malicious_ratio=0.3,
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
        s["attack_effective"] = drop > 0.02  # >2pp accuracy drop
        s["notes"] = f"accuracy drop vs baseline: {drop:.4f}"

    # ── export ───────────────────────────────────────────────────────────
    os.makedirs(base.results_dir, exist_ok=True)
    filepath = export_results(all_details, all_summaries, base.results_dir)
    logger.info(f"Excel results exported to {filepath}")

    generate_plots(all_curves, all_details, base.results_dir)
    logger.info(f"Plots saved to {base.results_dir}/")

    # ── print summary table ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("EXPERIMENT SUMMARY")
    print("=" * 72)
    for s in all_summaries:
        print(
            f"  {s['experiment_name']:<22s}  "
            f"acc={s['final_global_accuracy']:.4f}  "
            f"loss={s['final_global_loss']:.4f}  "
            f"SV_h={s['avg_round_shapley_honest']:.6f}  "
            f"SV_m={s['avg_round_shapley_malicious']:.6f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
