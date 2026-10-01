"""Generate visual dashboard plots from hyperparameter sweep runs."""

import glob
import json
import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np


def load_sweep_runs(runs_dir: str = "runs") -> List[Dict[str, Any]]:
    runs = []
    for filepath in sorted(glob.glob(os.path.join(runs_dir, "sweep_trial_*.json"))):
        with open(filepath, "r", encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def plot_training_curves(runs: List[Dict[str, Any]], output_path: str = "assets/training_curves.png") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    run_dict = {r["run_name"]: r for r in runs}

    key_runs = [
        ("sweep_trial_014", "Trial 14 (AdamW, lr=0.001) - Best", "#1f77b4", "o-"),
        ("sweep_trial_002", "Trial 02 (Adam, lr=0.001) - Runner-up", "#2ca02c", "s-"),
        ("sweep_trial_007", "Trial 07 (SGD, lr=0.0003) - Underfitting", "#ff7f0e", "^-"),
        ("sweep_trial_008", "Trial 08 (Adam, lr=0.02) - High LR Divergence", "#d62728", "x-"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # Panel 1: Loss curves
    for run_name, label, color, marker in key_runs:
        if run_name in run_dict:
            history = run_dict[run_name]["history"]
            epochs = [h["epoch"] for h in history]
            val_losses = [h["val_loss"] for h in history]
            ax1.plot(epochs, val_losses, marker, label=f"{label} (Val)", color=color, linewidth=2.2, markersize=7)

    # Also show train loss for best trial
    if "sweep_trial_014" in run_dict:
        history = run_dict["sweep_trial_014"]["history"]
        epochs = [h["epoch"] for h in history]
        train_losses = [h["train_loss"] for h in history]
        ax1.plot(epochs, train_losses, "--", label="Trial 14 (Train Loss)", color="#1f77b4", alpha=0.65, linewidth=1.8)

    ax1.set_title("Loss Trajectories across Regimes", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.set_xticks([1, 2, 3])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, fontsize=9.5)

    # Panel 2: Accuracy curves
    for run_name, label, color, marker in key_runs:
        if run_name in run_dict:
            history = run_dict[run_name]["history"]
            epochs = [h["epoch"] for h in history]
            val_accs = [h["val_acc"] for h in history]
            ax2.plot(epochs, val_accs, marker, label=label, color=color, linewidth=2.2, markersize=7)

    # Random guessing reference line
    ax2.axhline(y=10.0, color="#888888", linestyle=":", label="Random Baseline (10%)", alpha=0.7)

    ax2.set_title("Validation Accuracy Dynamics", fontsize=13, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax2.set_xticks([1, 2, 3])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved training curves plot to {output_path}")


def plot_hyperparameter_analysis(runs: List[Dict[str, Any]], output_path: str = "assets/hyperparameter_analysis.png") -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    optimizers = []
    lrs = []
    depths = []
    val_accs = []

    for r in runs:
        cfg = r["config"]
        summary = r["summary"]
        optimizers.append(cfg["optimizer"])
        lrs.append(cfg["lr"])
        depths.append(cfg["num_blocks"])
        val_accs.append(summary["best_val_acc"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    # Subplot 1: Optimizer Comparison
    opt_labels = ["adamw", "adam", "sgd"]
    opt_data = {opt: [acc for opt_val, acc in zip(optimizers, val_accs) if opt_val == opt] for opt in opt_labels}
    positions = [1, 2, 3]

    for pos, opt in zip(positions, opt_labels):
        accs = opt_data[opt]
        ax1.scatter([pos] * len(accs), accs, alpha=0.7, s=60, color="#2b5c8f", zorder=3)
        mean_acc = np.mean(accs) if accs else 0
        ax1.hlines(mean_acc, pos - 0.25, pos + 0.25, colors="#d62728", linestyles="-", linewidth=2.5, zorder=4)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(["AdamW", "Adam", "SGD"], fontsize=11)
    ax1.set_title("Validation Accuracy by Optimizer", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Subplot 2: Learning Rate vs Val Acc
    unique_lrs = sorted(list(set(lrs)))
    ax2.scatter(lrs, val_accs, alpha=0.8, s=70, c=depths, cmap="viridis", edgecolors="black", linewidth=0.5, zorder=3)
    ax2.set_xscale("log")
    ax2.set_xticks(unique_lrs)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.set_title("Validation Accuracy vs. Learning Rate", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Learning Rate (log scale)", fontsize=11)
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Subplot 3: Network Depth vs Val Acc
    depth_labels = [2, 3, 4]
    depth_data = {d: [acc for depth_val, acc in zip(depths, val_accs) if depth_val == d] for d in depth_labels}

    for d in depth_labels:
        accs = depth_data[d]
        ax3.scatter([d] * len(accs), accs, alpha=0.7, s=60, color="#2b5c8f", zorder=3)
        mean_acc = np.mean(accs) if accs else 0
        ax3.hlines(mean_acc, d - 0.25, d + 0.25, colors="#d62728", linestyles="-", linewidth=2.5, zorder=4)

    ax3.set_xticks(depth_labels)
    ax3.set_xticklabels(["2 Blocks", "3 Blocks", "4 Blocks"], fontsize=11)
    ax3.set_title("Validation Accuracy by Depth", fontsize=12, fontweight="bold", pad=10)
    ax3.set_xlabel("Convolutional Blocks", fontsize=11)
    ax3.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.4, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved hyperparameter analysis plot to {output_path}")


if __name__ == "__main__":
    runs = load_sweep_runs("runs")
    print(f"Loaded {len(runs)} sweep trials.")
    plot_training_curves(runs)
    plot_hyperparameter_analysis(runs)
