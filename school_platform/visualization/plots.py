"""
Publication-quality visualizations for 20 Newsgroups FL experiments.

Generates 10 chart groups with consistent seaborn styling,
honest (steelblue) vs malicious (coral) color coding, and
baseline experiment reference lines.

Charts:
  01 – Accuracy & Loss curves
  02 – Round-level Shapley comparison + cumulative SV bars
  03 – Per-client per-round Shapley heatmap (all experiments)
  04 – Per-class SV fingerprint (honest vs malicious bar chart)
  05 – Two-metric scatter (variance vs positive sum)
  06 – Metric distribution boxplots
  07 – Per-class SV deep-dive heatmap (single honest vs malicious)
  08 – Class metrics overview (variance/pos-sum over rounds + bars)
  09 – Per-client cumulative SV bar chart
  10 – Multi-attack summary comparison
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import seaborn as sns

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Global Style
# ═══════════════════════════════════════════════════════════════════════════════

sns.set_theme(
    style="whitegrid",
    font="sans-serif",
    font_scale=1.0,
    rc={
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    },
)

# ── Color palette ────────────────────────────────────────────────────────────

COLOR_HONEST = "#4682B4"      # steelblue
COLOR_MALICIOUS = "#FF7F50"   # coral
COLOR_BASELINE = "#2E8B57"    # seagreen
COLOR_BASELINE_LIGHT = "#8FBC8F"
COLOR_EDGE_HONEST = "#2B5F8A"
COLOR_EDGE_MALICIOUS = "#CC5533"

DASH_MALICIOUS = (0, (3.5, 1.5))
DASH_BASELINE = (0, (4, 2, 1.5, 2))
LW = 1.6
ALPHA_BAR = 0.82
ALPHA_GRID = 0.25

HEATMAP_CMAP = "RdBu_r"

ALL_EXPS = ["baseline_no_attack", "attack_dfr", "attack_sdfr", "attack_afr"]
ATTACK_EXPS = ["attack_dfr", "attack_sdfr", "attack_afr"]


# ═══════════════════════════════════════════════════════════════════════════════
# Helper Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _label(exp_name: str) -> str:
    """Short readable label from experiment name."""
    return exp_name.replace("attack_", "").replace("baseline_no_attack", "BASELINE").upper()


def _is_baseline(name: str) -> bool:
    return "baseline" in name


def _save(fig, plots_dir: str, num: int, name: str):
    """Save figure with consistent naming and close."""
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"fig_{num:02d}_{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved: %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared Plot Components
# ═══════════════════════════════════════════════════════════════════════════════

def _line_honest_vs_mal(ax, df, exp_names, metric, *,
                        baseline_df=None,
                        ylabel="", title="",
                        y_zero_line=True):
    """
    Line plot: honest (solid) vs malicious (dashed) per round,
    one line pair per experiment.  Optionally adds baseline line.
    """
    for exp_name in exp_names:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        # honest mean
        h = sub[~sub["is_malicious"]].groupby("round")[metric].mean()
        # malicious mean
        m = sub[sub["is_malicious"]].groupby("round")[metric].mean()
        lb = _label(exp_name)
        ax.plot(h.index, h.values, color=COLOR_HONEST, linewidth=LW,
                label=f"{lb} honest")
        if len(m) > 0:
            ax.plot(m.index, m.values, color=COLOR_MALICIOUS,
                    linewidth=LW, linestyle=DASH_MALICIOUS,
                    label=f"{lb} malicious")

    if baseline_df is not None:
        bl = baseline_df.groupby("round")[metric].mean()
        ax.plot(bl.index, bl.values, color=COLOR_BASELINE,
                linewidth=LW + 0.4, linestyle=DASH_BASELINE,
                label="Baseline (no attack)")

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=6.5, ncol=1)
    if y_zero_line:
        ax.axhline(y=0, color="grey", linewidth=0.5, zorder=0)
    ax.grid(True, alpha=ALPHA_GRID)


def _bar_honest_vs_mal(ax, df, exp_names, metric, *,
                       agg="last_round",
                       ylabel="", title="",
                       baseline_df=None):
    """
    Grouped bar chart: honest vs malicious per experiment.
    agg can be "last_round" (default) or "mean".
    """
    bar_labels, bar_vals, bar_colors = [], [], []

    if baseline_df is not None:
        val = (baseline_df.groupby("round")[metric].mean().iloc[-1]
               if agg == "last_round"
               else baseline_df[metric].mean())
        bar_labels.append("Baseline")
        bar_vals.append(val)
        bar_colors.append(COLOR_BASELINE)

    for exp_name in exp_names:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        lb = _label(exp_name)

        if agg == "last_round":
            lr = sub["round"].max()
            last = sub[sub["round"] == lr]
            h_val = last[~last["is_malicious"]][metric].mean()
            m_val = last[last["is_malicious"]][metric].mean()
        else:
            h_val = sub[~sub["is_malicious"]][metric].mean()
            m_val = sub[sub["is_malicious"]][metric].mean()

        bar_labels += [f"{lb}\nhonest", f"{lb}\nmalicious"]
        bar_vals += [h_val, m_val]
        bar_colors += [COLOR_HONEST, COLOR_MALICIOUS]

    if not bar_labels:
        return

    x = np.arange(len(bar_labels))
    ax.bar(x, bar_vals, color=bar_colors, alpha=ALPHA_BAR, edgecolor="white",
           linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=7, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(y=0, color="grey", linewidth=0.5, zorder=0)
    ax.grid(True, axis="y", alpha=ALPHA_GRID)


def _heatmap_sv(ax, pivot, title, highlight_ids=None, xtick_every=5):
    """
    Single Shapley heatmap: rows = clients, cols = rounds.
    Malicious client rows get a red outline.
    """
    if pivot.empty:
        ax.set_title(title)
        return

    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 1e-6)
    sns.heatmap(
        pivot, ax=ax, cmap=HEATMAP_CMAP, center=0,
        vmin=-vmax, vmax=vmax,
        xticklabels=xtick_every,
        cbar_kws={"shrink": 0.75, "label": "Round SV"},
        linewidths=0,
    )
    ax.set_xlabel("Round")
    ax.set_ylabel("Client ID")
    ax.set_title(title)

    # Red rectangle around malicious client rows
    if highlight_ids:
        for i, cid in enumerate(pivot.index):
            if cid in highlight_ids:
                ax.add_patch(Rectangle(
                    (0, i), pivot.shape[1], 1,
                    fill=False, edgecolor="red", linewidth=1.5, zorder=10,
                ))


# ═══════════════════════════════════════════════════════════════════════════════
# Individual Chart Groups
# ═══════════════════════════════════════════════════════════════════════════════

def _chart_01_accuracy_loss(curves: dict, plots_dir: str):
    """Accuracy & Loss curves for all experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    colors_cycle = sns.color_palette("muted", n_colors=len(curves))
    for idx, (name, data) in enumerate(curves.items()):
        color = COLOR_BASELINE if _is_baseline(name) else colors_cycle[idx]
        lw = LW + 0.6 if _is_baseline(name) else LW
        ls = DASH_BASELINE if _is_baseline(name) else "solid"
        ax1.plot(data["acc"], color=color, linewidth=lw, linestyle=ls,
                 label=_label(name))
        ax2.plot(data["loss"], color=color, linewidth=lw, linestyle=ls,
                 label=_label(name))

    ax1.set_xlabel("Round"); ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Global Test Accuracy")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=ALPHA_GRID)

    ax2.set_xlabel("Round"); ax2.set_ylabel("Test Loss")
    ax2.set_title("Global Test Loss")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=ALPHA_GRID)

    fig.tight_layout()
    _save(fig, plots_dir, 1, "accuracy_loss")


def _chart_02_shapley_comparison(df, plots_dir):
    """Round-level SV lines + cumulative SV bars."""
    fig, (ax_line, ax_bar) = plt.subplots(1, 2, figsize=(16, 6))

    base_df = df[df["experiment_name"] == "baseline_no_attack"]

    _line_honest_vs_mal(
        ax_line, df, ATTACK_EXPS, "round_shapley_value",
        baseline_df=base_df,
        ylabel="Mean Round Shapley Value",
        title="Round-Level Shapley: Honest vs Malicious",
    )

    _bar_honest_vs_mal(
        ax_bar, df, ATTACK_EXPS, "cumulative_shapley_value",
        agg="last_round", baseline_df=base_df,
        ylabel="Final Cumulative Shapley Value",
        title="Cumulative Shapley: Honest vs Malicious",
    )

    fig.tight_layout()
    _save(fig, plots_dir, 2, "shapley_round_cumulative")


def _chart_03_shapley_heatmap(df, plots_dir):
    """Per-client per-round SV heatmap (all 4 experiments)."""
    fig, axes = plt.subplots(2, 2, figsize=(22, 14))
    axes = axes.flatten()

    for idx, exp_name in enumerate(ALL_EXPS):
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            axes[idx].set_title(_label(exp_name))
            continue
        pivot = sub.pivot_table(
            index="client_id", columns="round",
            values="round_shapley_value", aggfunc="mean",
        ).fillna(0.0)
        mal_ids = set(sub[sub["is_malicious"]]["client_id"].unique())
        _heatmap_sv(axes[idx], pivot, _label(exp_name), highlight_ids=mal_ids)

    # Legend for red outline
    fig.legend(
        handles=[Patch(edgecolor="red", facecolor="none", linewidth=1.5,
                       label="Malicious client (red outline)")],
        loc="lower center", fontsize=9, ncol=1,
    )
    fig.suptitle("Per-Client Per-Round Shapley Value", fontsize=15, y=1.01)
    fig.tight_layout()
    _save(fig, plots_dir, 3, "shapley_heatmap")


def _chart_04_per_class_fingerprint(df_pc, class_names, plots_dir):
    """Per-class SV fingerprint: honest vs malicious bars per attack."""
    class_cols = [f"class_{c}" for c in range(len(class_names))]
    fig, axes = plt.subplots(1, 3, figsize=(24, 6.5))

    for idx, exp_name in enumerate(ATTACK_EXPS):
        ax = axes[idx]
        sub = df_pc[df_pc["experiment_name"] == exp_name]
        if sub.empty:
            continue
        honest = sub[~sub["is_malicious"]][class_cols].mean().values
        mal = sub[sub["is_malicious"]][class_cols].mean().values

        x = np.arange(len(class_names))
        w = 0.35
        ax.bar(x - w / 2, honest, w, color=COLOR_HONEST, alpha=ALPHA_BAR,
               edgecolor=COLOR_EDGE_HONEST, linewidth=0.3, label="Honest")
        ax.bar(x + w / 2, mal, w, color=COLOR_MALICIOUS, alpha=ALPHA_BAR,
               edgecolor=COLOR_EDGE_MALICIOUS, linewidth=0.3, label="Malicious")
        ax.axhline(y=0, color="grey", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=75, fontsize=5.5)
        ax.set_ylabel("Mean Per-Class Shapley Value")
        ax.set_title(_label(exp_name))
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=ALPHA_GRID)

    fig.suptitle("Per-Class SV Fingerprint: Honest (peaked) vs Free-Rider (flat)",
                 fontsize=13)
    fig.tight_layout()
    _save(fig, plots_dir, 4, "per_class_fingerprint")


def _chart_05_two_metric_scatter(df, plots_dir):
    """Scatter: class_sv_variance vs positive_class_sv_sum (last 5 rounds avg)."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    for idx, exp_name in enumerate(ATTACK_EXPS):
        ax = axes[idx]
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        last_rounds = sub[sub["round"] >= sub["round"].max() - 4]
        stats = last_rounds.groupby(["client_id", "is_malicious"]).agg({
            "class_sv_variance": "mean",
            "positive_class_sv_sum": "mean",
        }).reset_index()

        honest = stats[~stats["is_malicious"]]
        mal = stats[stats["is_malicious"]]

        ax.scatter(honest["class_sv_variance"], honest["positive_class_sv_sum"],
                   c=COLOR_HONEST, s=80, marker="o",
                   edgecolors=COLOR_EDGE_HONEST, linewidth=0.4,
                   label="Honest", zorder=5)
        ax.scatter(mal["class_sv_variance"], mal["positive_class_sv_sum"],
                   c=COLOR_MALICIOUS, s=90, marker="X",
                   edgecolors=COLOR_EDGE_MALICIOUS, linewidth=0.4,
                   label="Malicious", zorder=5)
        ax.set_xlabel("Class SV Variance")
        ax.set_ylabel("Positive Class SV Sum")
        ax.set_title(_label(exp_name))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=ALPHA_GRID)

    fig.suptitle("Two-Metric Scatter: Honest (circle) vs Malicious (X)", fontsize=13)
    fig.tight_layout()
    _save(fig, plots_dir, 5, "two_metric_scatter")


def _chart_06_metric_boxplots(df, plots_dir):
    """Box plots: class_sv_variance (top) and positive_class_sv_sum (bottom)."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    for idx, exp_name in enumerate(ATTACK_EXPS):
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        h_data = sub[~sub["is_malicious"]]
        m_data = sub[sub["is_malicious"]]
        lb = _label(exp_name)

        # Variance
        bp1 = axes[0, idx].boxplot(
            [h_data["class_sv_variance"], m_data["class_sv_variance"]],
            labels=["Honest", "Malicious"], patch_artist=True, widths=0.55,
        )
        bp1["boxes"][0].set_facecolor(COLOR_HONEST)
        bp1["boxes"][1].set_facecolor(COLOR_MALICIOUS)
        for b in bp1["boxes"]:
            b.set_alpha(0.65)
        axes[0, idx].set_ylabel("Class SV Variance")
        axes[0, idx].set_title(f"{lb}: Variance")
        axes[0, idx].grid(True, axis="y", alpha=ALPHA_GRID)

        # Positive sum
        bp2 = axes[1, idx].boxplot(
            [h_data["positive_class_sv_sum"], m_data["positive_class_sv_sum"]],
            labels=["Honest", "Malicious"], patch_artist=True, widths=0.55,
        )
        bp2["boxes"][0].set_facecolor(COLOR_HONEST)
        bp2["boxes"][1].set_facecolor(COLOR_MALICIOUS)
        for b in bp2["boxes"]:
            b.set_alpha(0.65)
        axes[1, idx].set_ylabel("Positive Class SV Sum")
        axes[1, idx].set_title(f"{lb}: Positive Sum")
        axes[1, idx].grid(True, axis="y", alpha=ALPHA_GRID)

    fig.suptitle("Metric Distributions: Honest vs Malicious", fontsize=13)
    fig.tight_layout()
    _save(fig, plots_dir, 6, "metric_boxplots")


def _chart_07_per_class_sv_deepdive(df_pc, class_names, plots_dir):
    """Deep-dive: per-class per-round SV heatmap for one honest + one malicious client."""
    # Use DFR experiment (strongest detection signal)
    sub = df_pc[df_pc["experiment_name"] == "attack_dfr"]
    if sub.empty:
        logger.warning("No DFR data for deep-dive heatmap; skipping.")
        return

    honest_cids = sorted(sub[~sub["is_malicious"]]["client_id"].unique())
    mal_cids = sorted(sub[sub["is_malicious"]]["client_id"].unique())
    if not honest_cids or not mal_cids:
        return

    h_cid, m_cid = honest_cids[0], mal_cids[0]
    class_cols = [f"class_{c}" for c in range(len(class_names))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 7))

    for ax, cid, ctype in [(ax1, h_cid, "Honest"), (ax2, m_cid, "Malicious")]:
        cdata = sub[sub["client_id"] == cid]
        pivot = cdata.pivot_table(index="round", values=class_cols).T
        pivot.index = class_names

        vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 1e-6)
        sns.heatmap(
            pivot, ax=ax, cmap=HEATMAP_CMAP, center=0,
            vmin=-vmax, vmax=vmax,
            xticklabels=5,
            cbar_kws={"shrink": 0.7, "label": "Round SV"},
        )
        ax.set_xlabel("Round")
        ax.set_ylabel("Class")
        ax.set_title(f"DFR: {ctype} (Client {cid})")

    fig.suptitle("Per-Class Per-Round SV: Honest (varied) vs Free-Rider (blank)",
                 fontsize=13)
    fig.tight_layout()
    _save(fig, plots_dir, 7, "per_class_sv_deepdive")


def _chart_08_class_metrics_overview(df, plots_dir):
    """2×2: variance/pos-sum over rounds (top) + bars of mean values (bottom)."""
    fig, axes = plt.subplots(2, 2, figsize=(17, 11))

    base_df = df[df["experiment_name"] == "baseline_no_attack"]

    _line_honest_vs_mal(
        axes[0, 0], df, ATTACK_EXPS, "class_sv_variance",
        baseline_df=base_df,
        ylabel="Per-Class SV Variance",
        title="Per-Class SV Variance Over Rounds",
    )

    _line_honest_vs_mal(
        axes[0, 1], df, ATTACK_EXPS, "positive_class_sv_sum",
        baseline_df=base_df,
        ylabel="Positive Per-Class SV Sum",
        title="Positive Per-Class SV Sum Over Rounds",
    )

    _bar_honest_vs_mal(
        axes[1, 0], df, ATTACK_EXPS, "mean_class_sv_variance",
        agg="last_round", baseline_df=base_df,
        ylabel="Mean Class SV Variance",
        title="Avg Per-Class SV Variance: Honest vs Malicious",
    )

    _bar_honest_vs_mal(
        axes[1, 1], df, ATTACK_EXPS, "mean_positive_class_sv_sum",
        agg="last_round", baseline_df=base_df,
        ylabel="Mean Positive Class SV Sum",
        title="Avg Positive Per-Class SV Sum: Honest vs Malicious",
    )

    fig.tight_layout()
    _save(fig, plots_dir, 8, "class_metrics_overview")


def _chart_09_cumulative_sv_bar(df, plots_dir):
    """Per-client cumulative SV bar chart (all 4 experiments)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, exp_name in enumerate(ALL_EXPS):
        ax = axes[idx]
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            ax.set_title(_label(exp_name))
            continue
        lr = sub["round"].max()
        last = sub[sub["round"] == lr].drop_duplicates("client_id").sort_values("client_id")
        colors = [COLOR_MALICIOUS if r["is_malicious"] else COLOR_HONEST
                  for _, r in last.iterrows()]
        ax.bar(last["client_id"].values, last["cumulative_shapley_value"].values,
               color=colors, alpha=ALPHA_BAR, edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Client ID")
        ax.set_ylabel("Cumulative SV")
        ax.set_title(_label(exp_name))
        ax.axhline(y=0, color="grey", linewidth=0.5)
        ax.grid(True, axis="y", alpha=ALPHA_GRID)
        # Legend
        ax.legend(handles=[
            Patch(color=COLOR_HONEST, label="Honest"),
            Patch(color=COLOR_MALICIOUS, label="Malicious"),
        ], fontsize=7.5, loc="best")

    fig.suptitle("Per-Client Cumulative Shapley Value", fontsize=14)
    fig.tight_layout()
    _save(fig, plots_dir, 9, "cumulative_sv_bar")


def _chart_10_multi_attack_summary(df, plots_dir):
    """Multi-attack summary: variance and positive-sum bars across attacks."""
    fig, (ax_var, ax_pos) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Use last-round data for honest vs malicious averages per attack
    labels, h_vars, m_vars, h_poss, m_poss = [], [], [], [], []
    for exp_name in ATTACK_EXPS:
        sub = df[df["experiment_name"] == exp_name]
        if sub.empty:
            continue
        lr = sub["round"].max()
        last = sub[sub["round"] == lr]
        lb = _label(exp_name)
        labels.append(lb)
        h_vars.append(last[~last["is_malicious"]]["class_sv_variance"].mean())
        m_vars.append(last[last["is_malicious"]]["class_sv_variance"].mean())
        h_poss.append(last[~last["is_malicious"]]["positive_class_sv_sum"].mean())
        m_poss.append(last[last["is_malicious"]]["positive_class_sv_sum"].mean())

    if not labels:
        return

    x = np.arange(len(labels))
    w = 0.32

    for ax, h_vals, m_vals, ylabel, title in [
        (ax_var, h_vars, m_vars, "Mean Class SV Variance", "Variance by Attack"),
        (ax_pos, h_poss, m_poss, "Mean Positive Class SV Sum", "Pos Sum by Attack"),
    ]:
        ax.bar(x - w / 2, h_vals, w, color=COLOR_HONEST, alpha=ALPHA_BAR,
               label="Honest")
        ax.bar(x + w / 2, m_vals, w, color=COLOR_MALICIOUS, alpha=ALPHA_BAR,
               label="Malicious")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=ALPHA_GRID)

    fig.tight_layout()
    _save(fig, plots_dir, 10, "multi_attack_summary")


# ═══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def generate_plots(
    curves: dict,
    round_details: list,
    per_class_records: list,
    cumulative_per_class_sv: dict,
    experiment_name: str,
    class_names: list,
    plots_dir: str = "results/plots",
):
    """
    Generate all 10 publication-quality chart groups.

    Parameters
    ----------
    curves : dict of exp_name -> {"acc": list, "loss": list}
        Per-round accuracy and loss for each experiment.
    round_details : list of dict
        Per-round, per-client aggregated Shapley metrics.
    per_class_records : list of dict
        Raw per-class Shapley values (class_0 .. class_19) per (exp, round, client).
    cumulative_per_class_sv : dict of exp_name -> {cid: np.ndarray}
        Cumulative per-class SV per client per experiment.
    experiment_name : str
        Unused; kept for signature compatibility.
    class_names : list of str
        20 Newsgroups category name abbreviations.
    plots_dir : str
        Output directory for PNG files.
    """
    df = pd.DataFrame(round_details)
    df_pc = pd.DataFrame(per_class_records)

    logger.info("Generating publication-quality plots → %s/", plots_dir)

    _chart_01_accuracy_loss(curves, plots_dir)
    _chart_02_shapley_comparison(df, plots_dir)
    _chart_03_shapley_heatmap(df, plots_dir)
    _chart_04_per_class_fingerprint(df_pc, class_names, plots_dir)
    _chart_05_two_metric_scatter(df, plots_dir)
    _chart_06_metric_boxplots(df, plots_dir)
    _chart_07_per_class_sv_deepdive(df_pc, class_names, plots_dir)
    _chart_08_class_metrics_overview(df, plots_dir)
    _chart_09_cumulative_sv_bar(df, plots_dir)
    _chart_10_multi_attack_summary(df, plots_dir)

    logger.info("All %d chart groups saved.", 10)
