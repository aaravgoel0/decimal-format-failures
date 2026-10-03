#!/usr/bin/env python3
"""Build the fresh realistic-task mitigation confirmation figure."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("llama", "qwen")
LABELS = {"llama": "Llama 3.1 8B", "qwen": "Qwen3 4B"}
POLICIES = ("original", "selective", "blanket")
COLORS = {"original": "#777777", "selective": "#2A9D8F", "blanket": "#D1495B"}


def main():
    report = json.loads((ROOT / "results/downstream_mitigation_analysis_postcrash.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(MODELS))
    width = 0.24
    for offset, policy in enumerate(POLICIES):
        values = [
            report["models"][model]["domains"]["pooled"]["policies"][policy]["accuracy"]
            for model in MODELS
        ]
        positions = x + (offset - 1) * width
        ax.bar(positions, values, width, color=COLORS[policy], label=policy)
    ax.set_xticks(x, [LABELS[model] for model in MODELS])
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("Exact-answer accuracy")
    ax.set_title("A. Fresh FinQA and TAT-QA confirmation")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    domains = ("pooled", "finqa", "tatqa")
    domain_labels = ("Pooled", "FinQA", "TAT-QA")
    offsets = (-0.22, 0.0, 0.22)
    model_colors = ("#4C78A8", "#F58518")
    y = np.arange(len(domains))
    for model, offset, color in zip(MODELS, offsets, model_colors):
        values, lower, upper = [], [], []
        for domain in domains:
            result = report["models"][model]["domains"][domain]["selective_vs_original"]
            value = result["delta_accuracy"]
            interval = result["cluster_bootstrap95"]
            values.append(value)
            lower.append(value - interval[0])
            upper.append(interval[1] - value)
        ax.errorbar(
            values,
            y + offset,
            xerr=np.array([lower, upper]),
            fmt="o",
            capsize=3,
            color=color,
            label=LABELS[model],
            markersize=5,
        )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y, domain_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Selective minus original accuracy")
    ax.set_title("B. Base-clustered paired effects")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    output = ROOT / "figures/downstream_mitigation_confirmation.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
