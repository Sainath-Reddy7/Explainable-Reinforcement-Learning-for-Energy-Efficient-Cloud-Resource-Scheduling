"""
plots.py — Generate all report figures from the saved results JSON files.

Reads results/{comparison_results,dqn_training_history,trust_metrics,decision_log}.json
and writes PNGs into results/.

Figures
-------
  scheduler_comparison.png  — grouped bars: makespan/cost/energy/miss across loads
  dqn_learning_curve.png    — reward + makespan vs episode (learning curves)
  xai_method_comparison.png — bar chart of all 4 XAI methods across trust metrics
  deletion_insertion_curves.png — deletion + insertion curves for SHAP vs IG
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
R = HERE / "results"

# palette
NAVY = "#1B365D"
BLUE = "#0066CC"
TEAL = "#17A2B8"
ORANGE = "#E27D60"
GREEN = "#5C946E"
RED = "#C75450"
PURPLE = "#7D6B91"
GREY = "#888"
SCHED_COLORS = {
    "FCFS": GREY, "RoundRobin": "#AAA", "GreedyLeastLoaded": TEAL,
    "Min-Min": GREEN, "Max-Min": ORANGE, "PSO": PURPLE,
    "Q-learning": RED, "DQN (ours)": BLUE,
}
XAI_COLORS = {
    "kernelshap": BLUE, "grad_x_input": ORANGE,
    "occlusion": TEAL, "integrated_gradients": GREEN,
}


def plot_scheduler_comparison(comp, out):
    metrics = [("makespan_s", "Makespan (s)", False),
               ("cost", "Total Cost ($)", False),
               ("energy_wh", "Energy (Wh)", False),
               ("deadline_miss_rate", "Deadline Miss Rate", True)]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    loads = sorted(int(l) for l in comp.keys())
    for ax, (key, title, pct) in zip(axes, metrics):
        names = list(next(iter(comp.values())).keys())
        x = np.arange(len(loads))
        w = 0.8 / len(names)
        for i, name in enumerate(names):
            vals = [comp[str(l)][name][key] if str(l) in comp else comp[l][name][key]
                    for l in loads]
            ax.bar(x + i * w, vals, w, label=name, color=SCHED_COLORS.get(name, GREY))
        ax.set_title(title, fontweight="bold", color=NAVY)
        ax.set_xlabel("Number of Tasks")
        ax.set_xticks(x + w * (len(names) - 1) / 2)
        ax.set_xticklabels(loads)
        if pct:
            ax.set_ylim(0, max(0.05, ax.get_ylim()[1] * 1.1))
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2, loc="upper left")
    fig.suptitle("Scheduler Performance Comparison (Borg Workloads)",
                 fontsize=14, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def plot_learning_curve(history, out):
    eps = [h["episode"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(eps, [h["reward"] for h in history], color=BLUE, alpha=0.3, linewidth=1)
    # smoothed
    if len(eps) > 10:
        window = max(3, len(eps) // 10)
        smooth = np.convolve([h["reward"] for h in history], np.ones(window) / window, mode="valid")
        ax1.plot(eps[window - 1:], smooth, color=NAVY, linewidth=2, label=f"{window}-ep MA")
    ax1.set_title("Episode Reward (smoothed)", fontweight="bold", color=NAVY)
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Total Reward")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(eps, [h["makespan_s"] for h in history], color=ORANGE, alpha=0.4, linewidth=1)
    if len(eps) > 10:
        window = max(3, len(eps) // 10)
        smooth = np.convolve([h["makespan_s"] for h in history], np.ones(window) / window, mode="valid")
        ax2.plot(eps[window - 1:], smooth, color=RED, linewidth=2, label=f"{window}-ep MA")
    ax2.set_title("Episode Makespan (smoothed)", fontweight="bold", color=NAVY)
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Makespan (s)")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("DQN Training Curves (Double DQN + PER)",
                 fontsize=13, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def plot_xai_comparison(trust, out):
    methods = ["kernelshap", "grad_x_input", "occlusion", "integrated_gradients"]
    labels = ["KernelSHAP", "Grad×Input", "Occlusion", "Integ. Grad."]
    # pick comparable metrics
    metrics_specs = [
        ("deletion_aopc", "Deletion AOPC\n(↑ faithful)"),
        ("insertion_aopc", "Insertion AOPC\n(↑ faithful)"),
        ("infidelity", "Infidelity\n(↓ faithful)"),
        ("consistency", "Consistency\n(↑ stable)"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, (key, title) in zip(axes, metrics_specs):
        vals = [trust[m].get(key, 0) or 0 for m in methods]
        colors = [XAI_COLORS[m] for m in methods]
        bars = ax.bar(labels, vals, color=colors)
        ax.set_title(title, fontweight="bold", color=NAVY, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.suptitle("XAI Method Trust Benchmark (4 methods, 60 decisions)",
                 fontsize=13, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def plot_latency(trust, out):
    methods = ["kernelshap", "grad_x_input", "occlusion", "integrated_gradients"]
    labels = ["KernelSHAP", "Grad×Input", "Occlusion", "Integ. Grad."]
    vals = [trust[m]["mean_latency_sec"] * 1000 for m in methods]
    colors = [XAI_COLORS[m] for m in methods]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_title("Explanation Latency per Decision (ms, log scale)",
                 fontweight="bold", color=NAVY)
    ax.set_ylabel("Latency (ms)"); ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, which="both")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}ms",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main():
    print("Generating plots ...")
    comp = json.load(open(R / "comparison_results.json"))
    plot_scheduler_comparison(comp, R / "scheduler_comparison.png")

    hist = json.load(open(R / "dqn_training_history.json"))
    plot_learning_curve(hist, R / "dqn_learning_curve.png")

    trust = json.load(open(R / "trust_metrics.json"))
    plot_xai_comparison(trust, R / "xai_method_comparison.png")
    plot_latency(trust, R / "xai_latency.png")
    print("Done.")


if __name__ == "__main__":
    main()
