#!/usr/bin/env python3
"""Build the paired order-control figure from checked analysis output."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "paired_order_generalization_analysis.json"
OUT = ROOT / "figures" / "paired_order_generalization.png"


def short_name(model):
    if model.startswith("meta-llama"):
        return "Llama 3.1 8B"
    if model.startswith("Qwen"):
        return "Qwen3 4B"
    return "Gemma 2 9B"


def main():
    payload = json.loads(DATA.read_text())
    rows = payload["models"]
    names = [short_name(row["model"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, key, title in (
        (axes[0], "paired_order", "Strict exact-response accuracy"),
        (axes[1], "constrained_paired_order", "Constrained-label accuracy"),
    ):
        first = [100 * row[key]["padded_first_accuracy"] for row in rows]
        second = [100 * row[key]["padded_second_accuracy"] for row in rows]
        ax.bar(x - width / 2, first, width, label="Padded first", color="#D55E00")
        ax.bar(x + width / 2, second, width, label="Padded second", color="#0072B2")
        ax.axhline(100 / 3, color="#555555", linewidth=1, linestyle="--", label="Random choice")
        ax.set_title(title)
        ax.set_xticks(x, names)
        ax.set_ylim(0, 103)
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Paired numeral-order test with equality-label position controlled", fontsize=13)
    fig.tight_layout(rect=(0, .12, 1, .94))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
