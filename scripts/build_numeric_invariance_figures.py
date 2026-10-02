#!/usr/bin/env python3
"""Build final figures for downstream invariance and selective normalization."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("llama", "qwen", "gemma")
LABELS = {"llama": "Llama 3.1 8B", "qwen": "Qwen3 4B", "gemma": "Gemma 2 9B"}
COLORS = {
    "canonical": "#2F6690",
    "padded": "#D1495B",
    "leading_zero": "#EDAE49",
    "scientific": "#6A4C93",
    "original": "#777777",
    "full_canonical": "#D1495B",
    "selective": "#2A9D8F",
}


def errorbar(ax, x, value, interval, **kwargs):
    lower = max(0.0, value - interval[0])
    upper = max(0.0, interval[1] - value)
    ax.errorbar(x, value, yerr=np.array([[lower], [upper]]), capsize=3,
                linewidth=1.1, **kwargs)


def downstream_figure():
    report = json.loads((ROOT / "results/numeric_invariance_analysis.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(MODELS))
    forms = ("canonical", "padded", "leading_zero", "scientific")
    width = 0.19
    for offset, form in enumerate(forms):
        values = [report["models"][model]["domains"]["pooled"]["forms"][form]["accuracy"]
                  for model in MODELS]
        positions = x + (offset - 1.5) * width
        ax.bar(positions, values, width, color=COLORS[form], label=form.replace("_", " "))
        for position, model, value in zip(positions, MODELS, values):
            interval = report["models"][model]["domains"]["pooled"]["forms"][form]["wilson95"]
            errorbar(ax, position, value, interval, fmt="none", color="#202020")
    ax.set_xticks(x, [LABELS[model] for model in MODELS])
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("Exact-answer accuracy")
    ax.set_title("A. Accuracy under equivalent rewrites")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    domains = ("gsm8k", "finqa", "tatqa")
    width = 0.24
    domain_colors = ("#4C78A8", "#F58518", "#54A24B")
    for offset, (domain, color) in enumerate(zip(domains, domain_colors)):
        values = [report["models"][model]["domains"][domain]["any_prediction_disagreement"]
                  for model in MODELS]
        positions = x + (offset - 1) * width
        ax.bar(positions, values, width, color=color, label=domain.upper() if domain != "tatqa" else "TAT-QA")
        for position, model, value in zip(positions, MODELS, values):
            interval = report["models"][model]["domains"][domain]["disagreement_wilson95"]
            errorbar(ax, position, value, interval, fmt="none", color="#202020")
    ax.set_xticks(x, [LABELS[model] for model in MODELS])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Four-form prediction disagreement")
    ax.set_title("B. Instability across task domains")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    output = ROOT / "figures/numeric_invariance_downstream.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output)


def mitigation_figure():
    report = json.loads((ROOT / "results/selective_normalization_analysis.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(MODELS))
    conditions = ("original", "full_canonical", "selective")
    width = 0.24
    for offset, condition in enumerate(conditions):
        values = [report["models"][model]["families"]["pooled"]["conditions"][condition]["accuracy"]
                  for model in MODELS]
        positions = x + (offset - 1) * width
        ax.bar(positions, values, width, color=COLORS[condition],
               label=condition.replace("_", " "))
        for position, model, value in zip(positions, MODELS, values):
            interval = report["models"][model]["families"]["pooled"]["conditions"][condition]["wilson95"]
            errorbar(ax, position, value, interval, fmt="none", color="#202020")
    ax.set_xticks(x, [LABELS[model] for model in MODELS])
    ax.set_ylim(0.2, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("A. Fresh 600-case confirmation")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)

    ax = axes[1]
    families = ("negative", "leading_zero", "long_fraction", "scientific", "signed_zero")
    y = np.arange(len(families))
    offsets = (-0.22, 0, 0.22)
    model_colors = ("#4C78A8", "#F58518", "#54A24B")
    for model, offset, color in zip(MODELS, offsets, model_colors):
        values, lows, highs = [], [], []
        for family in families:
            result = report["models"][model]["families"][family]["paired_changes"]["selective"]
            value = result["delta_accuracy"]
            values.append(value)
            lows.append(value - result["bootstrap95"][0])
            highs.append(result["bootstrap95"][1] - value)
        ax.errorbar(values, y + offset, xerr=np.array([lows, highs]), fmt="o", capsize=3,
                    color=color, label=LABELS[model], markersize=5)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y, [family.replace("_", " ") for family in families])
    ax.invert_yaxis()
    ax.set_xlim(-0.08, 0.62)
    ax.set_xlabel("Selective minus original accuracy")
    ax.set_title("B. Paired effects by format family")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    output = ROOT / "figures/selective_normalization_confirmation.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    downstream_figure()
    mitigation_figure()
