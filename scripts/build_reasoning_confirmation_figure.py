#!/usr/bin/env python3
"""Build the reasoning-enabled downstream confirmation figure."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FORMS = ("canonical", "padded", "leading_zero", "scientific")
FORM_LABELS = ("Canonical", "Padded", "Leading zero", "Scientific")
DOMAINS = ("finqa", "tatqa", "pooled")
DOMAIN_LABELS = ("FinQA", "TAT-QA", "Pooled")
POLICIES = ("original", "selective", "blanket")
POLICY_LABELS = ("Original", "Selective", "Blanket")
COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756")


def main():
    report = json.loads((ROOT / "results/reasoning_confirmation_analysis.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.15), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(DOMAINS))
    width = 0.18
    for index, (form, label, color) in enumerate(zip(FORMS, FORM_LABELS, COLORS)):
        values = [report["domains"][domain]["forms"][form]["accuracy"] for domain in DOMAINS]
        positions = x + (index - 1.5) * width
        ax.bar(positions, values, width, color=color, label=label)
    ax.set_xticks(x, DOMAIN_LABELS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Exact-answer accuracy")
    ax.set_title("A. Reasoning-enabled accuracy by form")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    pooled = report["domains"]["pooled"]
    values = [pooled["policies"][policy]["accuracy"] for policy in POLICIES]
    bars = ax.bar(POLICY_LABELS, values, color=("#777777", "#2A9D8F", "#D1495B"), width=0.62)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Exact-answer accuracy")
    ax.set_title("B. Frozen normalization policy")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018,
                f"{100 * value:.1f}%", ha="center", va="bottom", fontsize=9)
    effect = pooled["selective_vs_original"]
    low, high = effect["cluster_bootstrap95"]
    ax.text(0.5, 0.94,
            f"Selective - original: {100 * effect['delta_accuracy']:.1f} points\n"
            f"95% base-clustered CI [{100 * low:.1f}, {100 * high:.1f}]",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    output = ROOT / "figures/reasoning_enabled_confirmation.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
