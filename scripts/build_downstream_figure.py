#!/usr/bin/env python3
"""Build the paired downstream format-invariance figure."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLORS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "#D55E00",
    "Qwen/Qwen3-4B-Instruct-2507": "#0072B2",
    "google/gemma-2-9b-it": "#009E73",
}
LABELS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3 4B",
    "google/gemma-2-9b-it": "Gemma 2 9B",
}


def main():
    rows = json.loads(
        (ROOT / "results/downstream_format_invariance_analysis.json").read_text()
    )["models"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    offsets = (-0.18, 0, 0.18)
    for offset, row in zip(offsets, rows):
        x = np.arange(2, dtype=float) + offset
        cells = row["domains"]
        effects = np.array([100 * cell["padded_minus_canonical"] for cell in cells])
        lows = np.array([100 * cell["paired_bootstrap_95_ci"][0] for cell in cells])
        highs = np.array([100 * cell["paired_bootstrap_95_ci"][1] for cell in cells])
        axes[0].errorbar(
            x, effects, yerr=np.vstack([effects - lows, highs - effects]),
            marker="o", linestyle="none", capsize=3, color=COLORS[row["model"]],
            label=LABELS[row["model"]],
        )
        disagreements = np.array([
            100 * cell["prediction_disagreement_rate"] for cell in cells
        ])
        d_lows = np.array([
            100 * cell["prediction_disagreement_bootstrap_95_ci"][0] for cell in cells
        ])
        d_highs = np.array([
            100 * cell["prediction_disagreement_bootstrap_95_ci"][1] for cell in cells
        ])
        axes[1].errorbar(
            x, disagreements,
            yerr=np.vstack([disagreements - d_lows, d_highs - disagreements]),
            marker="o", linestyle="none", capsize=3, color=COLORS[row["model"]],
            label=LABELS[row["model"]],
        )

    axes[0].axhline(0, color="#555555", linewidth=1, linestyle=":")
    axes[0].set_ylabel("Padded minus canonical accuracy (points)")
    axes[0].set_title("Paired accuracy effect")
    axes[1].set_ylabel("Pairs with different numeric answers (%)")
    axes[1].set_title("Prediction instability")
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.set_xticks(range(2), ["GSM8K", "FinQA"])
        ax.grid(axis="y", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Equivalent decimal formatting in downstream numerical problems", fontsize=13)
    fig.tight_layout()
    output = ROOT / "figures/downstream_format_invariance.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
