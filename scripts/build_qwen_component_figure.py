#!/usr/bin/env python3
"""Build the Qwen component-localization figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/qwen_component_causal_analysis.json"


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, zorder=0)


def main():
    data = json.loads(RESULT.read_text())
    selected = set(data["selected_heads"])
    ranking = sorted(data["discovery_ranking"], key=lambda row: row["head"])
    fig = plt.figure(figsize=(10.8, 6.5))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1), hspace=0.42, wspace=0.30)
    head_ax = fig.add_subplot(grid[0, :])
    effect_ax = fig.add_subplot(grid[1, 0])
    contrast_ax = fig.add_subplot(grid[1, 1])

    heads = np.array([row["head"] for row in ranking])
    scores = np.array([row["discovery_centered_mean"] for row in ranking])
    colors = ["#0072B2" if head in selected else "#B8B8B8" for head in heads]
    head_ax.bar(heads, scores, color=colors, width=0.82, zorder=2)
    head_ax.axhline(0, color="#555555", linewidth=1, linestyle=":")
    head_ax.set_xlim(-0.8, 31.8)
    head_ax.set_xticks(range(0, 32, 2))
    head_ax.set_xlabel("Attention head (zero-based)")
    head_ax.set_ylabel("Discovery aligned-minus-random effect")
    head_ax.set_title("Discovery ranking at Qwen layer 2; blue heads were frozen for testing")
    clean(head_ax)

    keys = ("selected_heads", "attention", "mlp", "whole_block")
    labels = ("Selected\nheads", "All\nattention", "MLP", "Whole\nblock")
    aligned = np.array([
        data["interventions"][key]["aligned_mean_effect"] for key in keys
    ])
    random = np.array([
        data["interventions"][key]["random_case_mean_effect"] for key in keys
    ])
    x = np.arange(len(keys))
    width = 0.35
    effect_ax.bar(x - width / 2, aligned, width, color="#0072B2", label="Aligned")
    effect_ax.bar(x + width / 2, random, width, color="#999999", label="Random mean")
    effect_ax.axhline(0, color="#555555", linewidth=1, linestyle=":")
    effect_ax.set_xticks(x, labels)
    effect_ax.set_ylabel("Correct-label margin effect")
    effect_ax.set_title("Raw held-out effects")
    effect_ax.legend(frameon=False, fontsize=8)
    clean(effect_ax)

    centered = np.array([
        data["interventions"][key]["aligned_minus_random"] for key in keys
    ])
    lows = np.array([
        data["interventions"][key]["aligned_minus_random_bootstrap_95_ci"][0]
        for key in keys
    ])
    highs = np.array([
        data["interventions"][key]["aligned_minus_random_bootstrap_95_ci"][1]
        for key in keys
    ])
    contrast_ax.errorbar(
        x, centered, yerr=np.vstack([centered - lows, highs - centered]),
        marker="o", linestyle="none", capsize=4, color="#0072B2",
    )
    contrast_ax.axhline(0, color="#555555", linewidth=1, linestyle=":")
    contrast_ax.set_xticks(x, labels)
    contrast_ax.set_ylabel("Aligned minus random effect")
    contrast_ax.set_title("Held-out contrasts with 95% intervals")
    clean(contrast_ax)

    fig.suptitle("Component localization of Qwen's confirmed layer-2 effect", fontsize=14)
    fig.subplots_adjust(top=0.90)
    output = ROOT / "figures/qwen_component_causal.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
