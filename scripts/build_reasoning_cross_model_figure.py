#!/usr/bin/env python3
"""Build the three-model reasoning-enabled comparison figure."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
KEYS = ("llama", "qwen", "gemma27")
LABELS = ("Llama 3.1 8B", "Qwen3 4B", "Gemma 3 27B (4-bit)")


def main():
    report = json.loads((ROOT / "results/reasoning_cross_model_analysis.json").read_text())
    summary = report["descriptive_three_model_summary"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(KEYS))
    width = 0.34
    canonical = [summary[key]["canonical_accuracy"] for key in KEYS]
    scientific = [summary[key]["scientific_accuracy"] for key in KEYS]
    ax.bar(x - width / 2, canonical, width, color="#4C78A8", label="Canonical")
    ax.bar(x + width / 2, scientific, width, color="#E45756", label="Scientific")
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Exact-answer accuracy")
    ax.set_title("A. Reasoning-enabled accuracy")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    effects, lower, upper = [], [], []
    for key in KEYS:
        value = summary[key]["selective_minus_original"]
        low, high = summary[key]["selective_minus_original_bootstrap95"]
        effects.append(value)
        lower.append(value - low)
        upper.append(high - value)
    y = np.arange(len(KEYS))
    ax.errorbar(
        effects,
        y,
        xerr=np.array([lower, upper]),
        fmt="o",
        color="#2A9D8F",
        capsize=4,
        markersize=6,
    )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y, LABELS)
    ax.invert_yaxis()
    ax.set_xlabel("Selective minus original accuracy")
    ax.set_title("B. Frozen policy effect")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    output = ROOT / "figures/reasoning_cross_model_confirmation.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
