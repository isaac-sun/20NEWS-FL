#!/usr/bin/env python3
"""
Federated Learning on 20 Newsgroups with Poisoning Attacks,
Per-Class Shapley-Value Contribution Estimation, and Detection.

Runs four experiments:
  1. baseline (no attack)
  2. SF   (Sign-Flipping)
  3. ALIE (A Little Is Enough)
  4. LF   (Label-Flipping)

Detection metrics (per-class Shapley derived):
  - class_sv_variance:      variance of per-class SV across 20 classes
  - positive_class_sv_sum:  sum of positive per-class SV values

Outputs:
  results_poisoning/poisoning_results.xlsx
  results_poisoning/*.png  (multiple plots)
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
from attacks.sign_flip import sign_flip_attack
from attacks.alie import alie_attack
from attacks.label_flip import (
    create_label_flipped_subset,
    evaluate_targeted_attack,
)
from contribution.shapley import (
    estimate_round_shapley_per_class,
    per_class_to_overall,
    compute_class_metrics,
    _class_weights_from_loader,
)
from detection.utility_score import UtilityScoreTracker
from utils.partition import iid_partition, non_iid_partition

logger = get_logger()


# ─── helpers ─────────────────────────────────────────────────────────────────

def create_model(input_dim, config):
    return MLP(input_dim, config.hidden_dim, config.num_classes).to(config.device)


# ─── single experiment ───────────────────────────────────────────────────────

def run_poisoning_experiment(
    config, train_dataset, val_dataset, test_dataset,
    input_dim, train_labels,
):
    """
    Run one full FL experiment with optional poisoning attack.

    Supports: "none", "sf", "alie", "lf"
    Returns: (round_details, summary, test_accs, test_losses, asr_list, tacc_list)
    """
    set_seed(config.seed)
    logger.info(
        f"=== Experiment: {config.experiment_name} | "
        f"Attack: {config.attack_type} "
        f"{'(mode=' + config.alie_mode + ')' if config.attack_type == 'alie' else ''}"
        f"{'(u=' + str(config.sf_scale_u) + ')' if config.attack_type == 'sf' else ''}"
        f"{'(src=' + str(config.lf_source_class) + '→tgt=' + str(config.lf_target_class) + ')' if config.attack_type == 'lf' else ''}"
        f" ==="
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

    # ── prepare label-flipped datasets for LF malicious clients ──────────
    lf_datasets = {}
    if config.attack_type == "lf":
        for cid in malicious_ids:
            lf_datasets[cid] = create_label_flipped_subset(
                train_dataset, partition[cid],
                config.lf_source_class, config.lf_target_class,
            )
        logger.info(
            f"LF attack: {len(malicious_ids)} malicious clients, "
            f"source={config.lf_source_class} → target={config.lf_target_class}"
        )

    # ── tracking structures ──────────────────────────────────────────────
    utility_tracker = UtilityScoreTracker(alpha=config.utility_alpha)
    round_details: list[dict] = []
    test_accs: list[float] = []
    test_losses: list[float] = []
    asr_list: list[float] = []
    tacc_list: list[float] = []
    participation_counts = {cid: 0 for cid in range(config.num_clients)}
    cumulative_sv = {cid: 0.0 for cid in range(config.num_clients)}
    cumulative_class_var = {cid: [] for cid in range(config.num_clients)}
    cumulative_pos_sum = {cid: [] for cid in range(config.num_clients)}

    # Per-class cumulative SV for final fingerprint analysis
    cumulative_per_class_sv = {
        cid: np.zeros(config.num_classes, dtype=np.float64)
        for cid in range(config.num_clients)
    }

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    eval_model = model_fn()
    class_weights = _class_weights_from_loader(val_loader, config.num_classes)

    # ── FL rounds ────────────────────────────────────────────────────────
    for round_t in tqdm(range(config.num_rounds), desc=config.experiment_name):
        selected = server.select_clients(
            config.num_clients, config.participation_ratio
        )
        global_sd = server.get_global_state_dict()

        # ── Phase 1: collect honest/clean updates from ALL selected ──────
        # For SF/ALIE we need honest updates first; for LF the malicious
        # clients train on poisoned data directly.
        updates = {}
        benign_selected = [c for c in selected if c not in malicious_ids]
        malicious_selected = [c for c in selected if c in malicious_ids]

        # All honest clients train normally
        for cid in benign_selected:
            participation_counts[cid] += 1
            updates[cid] = clients[cid].train(global_sd)

        # ── Phase 2: handle malicious clients per attack type ────────────
        if config.attack_type == "sf":
            for cid in malicious_selected:
                participation_counts[cid] += 1
                # Step 1: honest training
                honest_update = clients[cid].train(global_sd)
                # Step 2: sign-flip
                updates[cid] = sign_flip_attack(honest_update, u=config.sf_scale_u)

        elif config.attack_type == "alie":
            # Collect malicious clients' clean updates (needed for "original" mode)
            malicious_clean = {}
            for cid in malicious_selected:
                participation_counts[cid] += 1
                malicious_clean[cid] = clients[cid].train(global_sd)

            # Build benign-only updates dict
            benign_updates = {c: updates[c] for c in benign_selected}

            # Craft single ALIE vector
            n_part = len(selected)
            n_mal = len(malicious_selected)
            if n_mal > 0 and (len(benign_updates) > 0 or len(malicious_clean) > 0):
                p_mal = alie_attack(
                    benign_updates=benign_updates,
                    malicious_clean_updates=malicious_clean,
                    n_participants=n_part,
                    n_malicious=n_mal,
                    mode=config.alie_mode,
                )
                # All malicious clients submit the same crafted update
                for cid in malicious_selected:
                    updates[cid] = p_mal
            else:
                # No malicious selected this round; nothing to do
                pass

        elif config.attack_type == "lf":
            for cid in malicious_selected:
                participation_counts[cid] += 1
                # Train on label-flipped dataset
                updates[cid] = clients[cid].train_with_dataset(
                    global_sd, lf_datasets[cid]
                )

        elif config.attack_type == "none":
            # No malicious clients
            pass

        else:
            raise ValueError(f"Unknown attack type: {config.attack_type}")

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

        # accumulate per-class SV
        for cid in selected:
            if cid in per_class_sv:
                cumulative_per_class_sv[cid] += per_class_sv[cid]

        # ── utility scores ───────────────────────────────────────────────
        utility_tracker.update(shapley_vals)

        # ── aggregate (standard FedAvg) ──────────────────────────────────
        new_sd = fedavg_aggregate(global_sd, updates, config.server_lr)
        server.update_global_model(new_sd)

        # ── evaluate ─────────────────────────────────────────────────────
        test_loss, test_acc = server.evaluate()
        test_accs.append(test_acc)
        test_losses.append(test_loss)

        # ── LF-specific metrics ──────────────────────────────────────────
        if config.attack_type == "lf":
            eval_model.load_state_dict(server.get_global_state_dict())
            asr, tacc = evaluate_targeted_attack(
                eval_model, test_loader,
                config.lf_source_class, config.lf_target_class,
                config.device,
            )
            asr_list.append(asr)
            tacc_list.append(tacc)

        logger.info(
            f"Round {round_t:>2d}: acc={test_acc:.4f}  loss={test_loss:.4f}  "
            f"selected={selected}"
            + (f"  ASR={asr:.4f}  TACC={tacc:.4f}" if config.attack_type == "lf" else "")
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

    # Final per-class SV metrics (cumulative across all rounds)
    def _final_per_class_var(ids):
        vals = [float(np.var(cumulative_per_class_sv[c])) for c in ids]
        return float(np.mean(vals)) if vals else 0.0

    def _final_per_class_pos_sum(ids):
        sv_arr = cumulative_per_class_sv
        vals = [float(np.sum(sv_arr[c][sv_arr[c] > 0])) for c in ids]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "experiment_name": config.experiment_name,
        "attack_type": config.attack_type,
        "malicious_ratio": config.malicious_ratio if num_mal > 0 else 0.0,
        "final_global_accuracy": test_accs[-1],
        "final_global_loss": test_losses[-1],
        "avg_round_shapley_honest": _avg_mean_sv(honest_ids),
        "avg_round_shapley_malicious": _avg_mean_sv(list(malicious_ids)),
        "avg_cumulative_shapley_honest": _avg_cum_sv(honest_ids),
        "avg_cumulative_shapley_malicious": _avg_cum_sv(list(malicious_ids)),
        "avg_class_sv_variance_honest": _avg_metric(honest_ids, cumulative_class_var),
        "avg_class_sv_variance_malicious": _avg_metric(list(malicious_ids), cumulative_class_var),
        "avg_positive_class_sv_sum_honest": _avg_metric(honest_ids, cumulative_pos_sum),
        "avg_positive_class_sv_sum_malicious": _avg_metric(list(malicious_ids), cumulative_pos_sum),
        "final_per_class_sv_variance_honest": _final_per_class_var(honest_ids),
        "final_per_class_sv_variance_malicious": _final_per_class_var(list(malicious_ids)),
        "final_positive_per_class_sv_sum_honest": _final_per_class_pos_sum(honest_ids),
        "final_positive_per_class_sv_sum_malicious": _final_per_class_pos_sum(list(malicious_ids)),
        "attack_effective": "",
        "notes": "",
    }

    # LF-specific summary fields
    if config.attack_type == "lf" and asr_list:
        summary["final_asr"] = asr_list[-1]
        summary["final_tacc"] = tacc_list[-1]

    # Per-client final metrics for export
    per_client_final = []
    for cid in range(config.num_clients):
        sv_vec = cumulative_per_class_sv[cid]
        per_client_final.append({
            "experiment_name": config.experiment_name,
            "client_id": cid,
            "is_malicious": cid in malicious_ids,
            "cumulative_sv": cumulative_sv[cid],
            "final_per_class_sv_variance": float(np.var(sv_vec)),
            "final_positive_per_class_sv_sum": float(np.sum(sv_vec[sv_vec > 0])),
            "participation_count": participation_counts[cid],
        })

    return (round_details, summary, test_accs, test_losses,
            asr_list, tacc_list, per_client_final, cumulative_per_class_sv)


# ─── plotting ────────────────────────────────────────────────────────────────

def generate_poisoning_plots(
    curves: dict,
    round_details: list,
    per_client_data: dict,          # exp_name -> list of per_client_final dicts
    cumul_pc_sv: dict,              # exp_name -> {cid: np.ndarray}
    results_dir: str,
    malicious_ids: set,
    attack_names: list,
    num_classes: int = 20,
):
    """Generate all poisoning experiment visualizations."""
    os.makedirs(results_dir, exist_ok=True)
    df = pd.DataFrame(round_details)

    cmap_h = "steelblue"
    cmap_m = "coral"

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Global performance curves
    # ═══════════════════════════════════════════════════════════════════════

    # 1a. Test accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in curves.items():
        ax.plot(data["acc"], label=name)
    ax.set_xlabel("Round"); ax.set_ylabel("Test Accuracy")
    ax.set_title("Global Test Accuracy Over Rounds (Poisoning)")
    ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(results_dir, "test_accuracy.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 1b. Test loss
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in curves.items():
        ax.plot(data["loss"], label=name)
    ax.set_xlabel("Round"); ax.set_ylabel("Test Loss")
    ax.set_title("Global Test Loss Over Rounds (Poisoning)")
    ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(results_dir, "test_loss.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 1c. ASR and TACC for LF
    for name, data in curves.items():
        if data.get("asr"):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(data["asr"], color="red", label="ASR")
            axes[0].set_xlabel("Round"); axes[0].set_ylabel("ASR")
            axes[0].set_title(f"{name}: Attack Success Rate")
            axes[0].legend(); axes[0].grid(True)
            axes[0].set_ylim([-0.05, 1.05])

            axes[1].plot(data["tacc"], color="green", label="TACC")
            axes[1].set_xlabel("Round"); axes[1].set_ylabel("TACC")
            axes[1].set_title(f"{name}: Source-Class Accuracy")
            axes[1].legend(); axes[1].grid(True)
            axes[1].set_ylim([-0.05, 1.05])

            fig.tight_layout()
            fig.savefig(os.path.join(results_dir, f"{name}_asr_tacc.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Standard SV plots
    # ═══════════════════════════════════════════════════════════════════════

    # 2a. Mean round-level SV: honest vs malicious
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for exp_name in attack_names:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        honest = sub[~sub["is_malicious"]].groupby("round")["round_shapley_value"].mean()
        mal = sub[sub["is_malicious"]].groupby("round")["round_shapley_value"].mean()
        label = exp_name.replace("attack_", "").upper()
        axes[0].plot(honest.index, honest.values, label=f"{label} honest")
        axes[0].plot(mal.index, mal.values, ls="--", label=f"{label} malicious")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Mean Round SV")
    axes[0].set_title("Round-Level Shapley: Honest vs Malicious")
    axes[0].legend(fontsize=8); axes[0].axhline(y=0, color="k", lw=0.5); axes[0].grid(True)

    # 2b. Mean cumulative SV: honest vs malicious
    for exp_name in attack_names:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        honest = sub[~sub["is_malicious"]].groupby("round")["cumulative_shapley_value"].mean()
        mal = sub[sub["is_malicious"]].groupby("round")["cumulative_shapley_value"].mean()
        label = exp_name.replace("attack_", "").upper()
        axes[1].plot(honest.index, honest.values, label=f"{label} honest")
        axes[1].plot(mal.index, mal.values, ls="--", label=f"{label} malicious")
    axes[1].set_xlabel("Round"); axes[1].set_ylabel("Mean Cumulative SV")
    axes[1].set_title("Cumulative Shapley: Honest vs Malicious")
    axes[1].legend(fontsize=8); axes[1].axhline(y=0, color="k", lw=0.5); axes[1].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "sv_round_cumulative.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2c. Per-client cumulative SV bar chart (one per attack)
    for exp_name in attack_names:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        last_round = sub["round"].max()
        last = sub[sub["round"] == last_round].drop_duplicates("client_id")
        last = last.sort_values("client_id")
        colors = [cmap_m if m else cmap_h for m in last["is_malicious"]]
        label = exp_name.replace("attack_", "").upper()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(last["client_id"].values, last["cumulative_shapley_value"].values,
               color=colors, alpha=0.85)
        ax.set_xlabel("Client ID"); ax.set_ylabel("Cumulative SV")
        ax.set_title(f"{label}: Per-Client Cumulative Shapley Value")
        ax.axhline(y=0, color="k", lw=0.5); ax.grid(True, axis="y")
        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=cmap_h, label="Honest"),
                           Patch(color=cmap_m, label="Malicious")])
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"{exp_name}_cum_sv_bar.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 2d. Per-client per-round SV heatmap (one per attack)
    for exp_name in attack_names:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="client_id", columns="round",
                                values="round_shapley_value", aggfunc="first")
        pivot = pivot.fillna(0.0)

        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                        interpolation="nearest")
        ax.set_xlabel("Round"); ax.set_ylabel("Client ID")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        label = exp_name.replace("attack_", "").upper()
        ax.set_title(f"{label}: Per-Client Per-Round Shapley Value")
        plt.colorbar(im, ax=ax, label="Round SV")
        # Mark malicious clients
        for idx, cid in enumerate(pivot.index):
            if cid in malicious_ids:
                ax.text(-1.5, idx, "★", fontsize=10, color="red",
                        ha="center", va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"{exp_name}_sv_heatmap.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Final class-wise SV fingerprint
    # ═══════════════════════════════════════════════════════════════════════
    for exp_name in attack_names:
        if exp_name not in cumul_pc_sv:
            continue
        pc_data = cumul_pc_sv[exp_name]
        honest_cids = [c for c in pc_data if c not in malicious_ids]
        mal_cids = [c for c in pc_data if c in malicious_ids]

        if not honest_cids or not mal_cids:
            continue

        mean_honest = np.mean([pc_data[c] for c in honest_cids], axis=0)
        mean_mal = np.mean([pc_data[c] for c in mal_cids], axis=0)

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(num_classes)
        w = 0.35
        ax.bar(x - w/2, mean_honest, w, color=cmap_h, alpha=0.85, label="Honest (mean)")
        ax.bar(x + w/2, mean_mal, w, color=cmap_m, alpha=0.85, label="Malicious (mean)")
        ax.set_xlabel("Class"); ax.set_ylabel("Cumulative Per-Class SV")
        label = exp_name.replace("attack_", "").upper()
        ax.set_title(f"{label}: Final Per-Class Shapley Fingerprint")
        ax.set_xticks(x); ax.legend(); ax.grid(True, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"{exp_name}_class_sv_fingerprint.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Final detection metrics
    # ═══════════════════════════════════════════════════════════════════════
    for exp_name in attack_names:
        if exp_name not in per_client_data:
            continue
        pcf = pd.DataFrame(per_client_data[exp_name])
        label = exp_name.replace("attack_", "").upper()

        # 4a. Box/violin plots of final per-class SV variance
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        h_var = pcf[~pcf["is_malicious"]]["final_per_class_sv_variance"]
        m_var = pcf[pcf["is_malicious"]]["final_per_class_sv_variance"]
        bp = axes[0].boxplot([h_var, m_var], labels=["Honest", "Malicious"],
                             patch_artist=True)
        bp["boxes"][0].set_facecolor(cmap_h); bp["boxes"][1].set_facecolor(cmap_m)
        axes[0].set_ylabel("Final Per-Class SV Variance")
        axes[0].set_title(f"{label}: Per-Class SV Variance")
        axes[0].grid(True, axis="y")

        # 4b. Box plot of final positive per-class SV sum
        h_pos = pcf[~pcf["is_malicious"]]["final_positive_per_class_sv_sum"]
        m_pos = pcf[pcf["is_malicious"]]["final_positive_per_class_sv_sum"]
        bp2 = axes[1].boxplot([h_pos, m_pos], labels=["Honest", "Malicious"],
                              patch_artist=True)
        bp2["boxes"][0].set_facecolor(cmap_h); bp2["boxes"][1].set_facecolor(cmap_m)
        axes[1].set_ylabel("Final Positive Per-Class SV Sum")
        axes[1].set_title(f"{label}: Positive Per-Class SV Sum")
        axes[1].grid(True, axis="y")

        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"{exp_name}_detection_boxes.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 4c. Two-metric scatter
        fig, ax = plt.subplots(figsize=(8, 6))
        h_data = pcf[~pcf["is_malicious"]]
        m_data = pcf[pcf["is_malicious"]]
        ax.scatter(h_data["final_per_class_sv_variance"],
                   h_data["final_positive_per_class_sv_sum"],
                   c=cmap_h, marker="o", s=80, label="Honest", zorder=3)
        ax.scatter(m_data["final_per_class_sv_variance"],
                   m_data["final_positive_per_class_sv_sum"],
                   c=cmap_m, marker="x", s=100, label="Malicious", zorder=3)
        ax.set_xlabel("Final Per-Class SV Variance")
        ax.set_ylabel("Final Positive Per-Class SV Sum")
        ax.set_title(f"{label}: Two-Metric Detection Scatter")
        ax.legend(); ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, f"{exp_name}_scatter_2metric.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Multi-attack summary
    # ═══════════════════════════════════════════════════════════════════════
    attack_only = [a for a in attack_names if "baseline" not in a]
    if attack_only:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        labels_list, h_vars, m_vars, h_pos_sums, m_pos_sums = [], [], [], [], []
        for exp_name in attack_only:
            if exp_name not in per_client_data:
                continue
            pcf = pd.DataFrame(per_client_data[exp_name])
            label = exp_name.replace("attack_", "").upper()
            labels_list.append(label)
            h_vars.append(pcf[~pcf["is_malicious"]]["final_per_class_sv_variance"].mean())
            m_vars.append(pcf[pcf["is_malicious"]]["final_per_class_sv_variance"].mean())
            h_pos_sums.append(pcf[~pcf["is_malicious"]]["final_positive_per_class_sv_sum"].mean())
            m_pos_sums.append(pcf[pcf["is_malicious"]]["final_positive_per_class_sv_sum"].mean())

        if labels_list:
            x = np.arange(len(labels_list))
            w = 0.35
            axes[0].bar(x - w/2, h_vars, w, color=cmap_h, label="Honest")
            axes[0].bar(x + w/2, m_vars, w, color=cmap_m, label="Malicious")
            axes[0].set_xticks(x); axes[0].set_xticklabels(labels_list)
            axes[0].set_ylabel("Mean Final Per-Class SV Variance")
            axes[0].set_title("Multi-Attack: Per-Class SV Variance")
            axes[0].legend(); axes[0].grid(True, axis="y")

            axes[1].bar(x - w/2, h_pos_sums, w, color=cmap_h, label="Honest")
            axes[1].bar(x + w/2, m_pos_sums, w, color=cmap_m, label="Malicious")
            axes[1].set_xticks(x); axes[1].set_xticklabels(labels_list)
            axes[1].set_ylabel("Mean Final Positive Per-Class SV Sum")
            axes[1].set_title("Multi-Attack: Positive Per-Class SV Sum")
            axes[1].legend(); axes[1].grid(True, axis="y")

        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "multi_attack_summary.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # Bonus: per-class per-round SV heatmap for representative clients
    # ═══════════════════════════════════════════════════════════════════════
    # (one honest, one malicious per attack)
    for exp_name in attack_names:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue

        # Pick representative honest and malicious client
        honest_cids = sorted([c for c in sub["client_id"].unique() if c not in malicious_ids])
        mal_cids = sorted([c for c in sub["client_id"].unique() if c in malicious_ids])
        if not honest_cids or not mal_cids:
            continue

        rep_h = honest_cids[0]
        rep_m = mal_cids[0]

        # Get per-class SV data from round_details
        sub_h = sub[sub["client_id"] == rep_h].sort_values("round")
        sub_m = sub[sub["client_id"] == rep_m].sort_values("round")

        # We only have aggregate class metrics in round_details, not per-class vectors.
        # For the heatmap, we'll use the cumulative per-class SV (final snapshot).
        # This gives a single fingerprint per client, not per-round per-class.
        # Skip this if data isn't rich enough.

    logger.info(f"All plots saved to {results_dir}/")


# ─── export ──────────────────────────────────────────────────────────────────

def export_poisoning_results(
    round_details: list,
    summaries: list,
    per_client_all: list,
    output_dir: str,
) -> str:
    """Export poisoning experiment results to Excel."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "poisoning_results.xlsx")

    df_details = pd.DataFrame(round_details)
    df_summary = pd.DataFrame(summaries)
    df_clients = pd.DataFrame(per_client_all)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        df_details.to_excel(writer, sheet_name="round_details", index=False)
        df_summary.to_excel(writer, sheet_name="experiment_summary", index=False)
        df_clients.to_excel(writer, sheet_name="per_client_final", index=False)

    return filepath


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    base = Config(
        num_clients=20,
        num_rounds=30,
        local_epochs=3,
        local_lr=0.001,
        server_lr=1.0,
        participation_ratio=0.5,       # 10 out of 20
        batch_size=64,
        hidden_dim=256,
        num_classes=20,
        max_features=10000,
        val_ratio=0.1,
        iid=True,
        malicious_ratio=0.3,           # 6 out of 20
        num_mc_samples=30,
        seed=42,
        device="cpu",
        results_dir="results_poisoning",

        # Poisoning defaults
        sf_scale_u=-1.0,
        alie_mode="svrfl",
        lf_source_class=0,             # alt.atheism
        lf_target_class=10,            # sci.crypt
    )

    # ── load data once ───────────────────────────────────────────────────
    set_seed(base.seed)
    train_ds, val_ds, test_ds, input_dim, train_labels = load_newsgroups(
        max_features=base.max_features, val_ratio=base.val_ratio,
    )
    logger.info(
        f"Data loaded — input_dim={input_dim}, "
        f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    # ── run experiments ──────────────────────────────────────────────────
    experiments = [
        ("baseline_no_attack", "none"),
        ("attack_sf", "sf"),
        ("attack_alie", "alie"),
        ("attack_lf", "lf"),
    ]

    all_details: list[dict] = []
    all_summaries: list[dict] = []
    all_curves: dict = {}
    all_per_client: dict = {}           # exp_name -> list of per_client dicts
    all_cumul_pc_sv: dict = {}          # exp_name -> {cid: ndarray}
    all_per_client_flat: list[dict] = []

    for exp_name, attack_type in experiments:
        cfg = copy.deepcopy(base)
        cfg.experiment_name = exp_name
        cfg.attack_type = attack_type

        (details, summary, accs, losses,
         asr_list, tacc_list, per_client_final,
         cumul_pc_sv) = run_poisoning_experiment(
            cfg, train_ds, val_ds, test_ds, input_dim, train_labels,
        )

        all_details.extend(details)
        all_summaries.append(summary)
        all_curves[exp_name] = {
            "acc": accs, "loss": losses,
            "asr": asr_list, "tacc": tacc_list,
        }
        all_per_client[exp_name] = per_client_final
        all_cumul_pc_sv[exp_name] = cumul_pc_sv
        all_per_client_flat.extend(per_client_final)

    # ── mark attack effectiveness (relative to baseline) ─────────────────
    baseline_acc = all_summaries[0]["final_global_accuracy"]
    for s in all_summaries[1:]:
        drop = baseline_acc - s["final_global_accuracy"]
        s["attack_effective"] = drop > 0.02
        s["notes"] = f"accuracy drop vs baseline: {drop:.4f}"

    # ── malicious ids (consistent: first 6 clients) ─────────────────────
    num_mal = int(base.num_clients * base.malicious_ratio)
    malicious_ids = set(range(num_mal))
    attack_names = [name for name, _ in experiments if name != "baseline_no_attack"]

    # ── export ───────────────────────────────────────────────────────────
    os.makedirs(base.results_dir, exist_ok=True)
    filepath = export_poisoning_results(
        all_details, all_summaries, all_per_client_flat, base.results_dir,
    )
    logger.info(f"Excel results exported to {filepath}")

    # ── plots ────────────────────────────────────────────────────────────
    generate_poisoning_plots(
        all_curves, all_details,
        all_per_client, all_cumul_pc_sv,
        base.results_dir,
        malicious_ids,
        attack_names + ["baseline_no_attack"],
        num_classes=base.num_classes,
    )

    # ── print summary table ──────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("POISONING EXPERIMENT SUMMARY")
    print("=" * 120)
    for s in all_summaries:
        line = (
            f"  {s['experiment_name']:<22s}  "
            f"acc={s['final_global_accuracy']:.4f}  "
            f"loss={s['final_global_loss']:.4f}  "
            f"SV_h={s['avg_round_shapley_honest']:.6f}  "
            f"SV_m={s['avg_round_shapley_malicious']:.6f}  "
            f"FVar_h={s['final_per_class_sv_variance_honest']:.2e}  "
            f"FVar_m={s['final_per_class_sv_variance_malicious']:.2e}  "
            f"FPos_h={s['final_positive_per_class_sv_sum_honest']:.6f}  "
            f"FPos_m={s['final_positive_per_class_sv_sum_malicious']:.6f}"
        )
        if "final_asr" in s:
            line += f"  ASR={s['final_asr']:.4f}  TACC={s['final_tacc']:.4f}"
        print(line)
    print("=" * 120)

    # ── detection analysis ───────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("DETECTION ANALYSIS")
    print("=" * 120)
    for exp_name in attack_names:
        s = next(x for x in all_summaries if x["experiment_name"] == exp_name)
        label = exp_name.replace("attack_", "").upper()
        print(f"\n{'─'*60}")
        print(f"  Attack: {label}")
        print(f"{'─'*60}")

        sv_h = s["avg_round_shapley_honest"]
        sv_m = s["avg_round_shapley_malicious"]
        cum_h = s["avg_cumulative_shapley_honest"]
        cum_m = s["avg_cumulative_shapley_malicious"]
        fvar_h = s["final_per_class_sv_variance_honest"]
        fvar_m = s["final_per_class_sv_variance_malicious"]
        fpos_h = s["final_positive_per_class_sv_sum_honest"]
        fpos_m = s["final_positive_per_class_sv_sum_malicious"]

        round_sv_detect = "YES" if abs(sv_h - sv_m) / max(abs(sv_h), 1e-9) > 0.3 else "WEAK/NO"
        cum_sv_detect = "YES" if abs(cum_h - cum_m) / max(abs(cum_h), 1e-9) > 0.3 else "WEAK/NO"

        # For variance: higher variance for malicious = detectable
        var_ratio = fvar_m / max(fvar_h, 1e-12)
        var_detect = "YES" if var_ratio > 2.0 or var_ratio < 0.5 else "WEAK/NO"

        # For positive sum: lower positive sum for malicious = detectable
        pos_ratio = fpos_m / max(fpos_h, 1e-9) if fpos_h > 0 else 0.0
        pos_detect = "YES" if pos_ratio < 0.5 or pos_ratio > 2.0 else "WEAK/NO"

        print(f"  Round-level SV detects malicious?     {round_sv_detect}  (H={sv_h:.6f}, M={sv_m:.6f})")
        print(f"  Cumulative SV detects malicious?      {cum_sv_detect}  (H={cum_h:.6f}, M={cum_m:.6f})")
        print(f"  Final per-class SV var detects?        {var_detect}  (H={fvar_h:.2e}, M={fvar_m:.2e}, ratio={var_ratio:.2f})")
        print(f"  Final positive per-class SV sum?       {pos_detect}  (H={fpos_h:.6f}, M={fpos_m:.6f}, ratio={pos_ratio:.2f})")

        joint = "YES" if (var_detect == "YES" or pos_detect == "YES") else "WEAK/NO"
        print(f"  Joint two-metric separates?            {joint}")

    print("\n" + "=" * 120)


if __name__ == "__main__":
    main()
