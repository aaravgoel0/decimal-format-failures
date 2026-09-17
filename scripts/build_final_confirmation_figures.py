#!/usr/bin/env python3
"""Build the prompt-robustness and many-random-site confirmation figures."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
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
ORDER = list(LABELS)


def ordered(rows):
    by_model = {row["model"]: row for row in rows}
    return [by_model[model] for model in ORDER]


def prompt_figure():
    rows = ordered(json.loads(
        (ROOT / "results/prompt_robustness_confirmation_analysis.json").read_text()))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    x = np.arange(3)
    width = 0.34
    first = [100 * row["primary_constrained_paired_order"]["padded_first_accuracy"]
             for row in rows]
    second = [100 * row["primary_constrained_paired_order"]["padded_second_accuracy"]
              for row in rows]
    axes[0].bar(x - width / 2, first, width, color="#CC6677", label="Padded first")
    axes[0].bar(x + width / 2, second, width, color="#4477AA", label="Padded second")
    axes[0].axhline(100 / 3, color="#555555", linewidth=1, linestyle=":")
    axes[0].set_xticks(x, [LABELS[row["model"]] for row in rows])
    axes[0].set_ylim(0, 103)
    axes[0].set_ylabel("Constrained-label accuracy (%)")
    axes[0].set_title("Pooled across 10 prompt templates")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=.18)

    for row in rows:
        effects = [100 * value["paired_first_minus_second"]
                   for value in row["prompt_specific"]]
        lows = [100 * value["paired_bootstrap_95_ci"][0]
                for value in row["prompt_specific"]]
        highs = [100 * value["paired_bootstrap_95_ci"][1]
                 for value in row["prompt_specific"]]
        yerr = np.vstack([np.asarray(effects) - lows, np.asarray(highs) - effects])
        axes[1].errorbar(np.arange(1, 11), effects, yerr=yerr, marker="o",
                         markersize=3.5, linewidth=1.2, capsize=2,
                         color=COLORS[row["model"]], label=LABELS[row["model"]])
    axes[1].axhline(0, color="#555555", linewidth=1, linestyle=":")
    axes[1].set_xticks(range(1, 11))
    axes[1].set_xlabel("Prompt template")
    axes[1].set_ylabel("Padded-first minus padded-second (points)")
    axes[1].set_title("Paired effect by prompt template")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=.18)
    fig.suptitle("Fresh prompt and label robustness confirmation", fontsize=13)
    fig.tight_layout()
    path = FIGURES / "prompt_robustness_confirmation.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def causal_figure():
    payload = json.loads(
        (ROOT / "results/causal_random_site_confirmation_analysis.json").read_text())
    rows = ordered(payload["models"])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    components = ("short", "padded", "joint")
    component_labels = ("Canonical only", "Padded only", "Joint")
    offsets = (-.24, 0, .24)
    for component, label, offset in zip(components, component_labels, offsets):
        means, low, high = [], [], []
        for row in rows:
            value = row["components"][component]
            means.append(value["aligned_minus_random"])
            low.append(value["aligned_minus_random_bootstrap_95_ci"][0])
            high.append(value["aligned_minus_random_bootstrap_95_ci"][1])
        means = np.asarray(means)
        yerr = np.vstack([means - low, np.asarray(high) - means])
        axes[0].errorbar(np.arange(3) + offset, means, yerr=yerr, marker="o",
                         linestyle="none", capsize=3, label=label)
    axes[0].axhline(0, color="#555555", linewidth=1, linestyle=":")
    axes[0].set_xticks(np.arange(3), [LABELS[row["model"]] for row in rows])
    axes[0].set_ylabel("Aligned minus random margin effect")
    axes[0].set_title("Fixed-site effects with 95% bootstrap intervals")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=.18)

    aligned = [row["components"]["joint"]["aligned_incorrect_to_correct_flips"]
               for row in rows]
    random = [row["components"]["joint"]["random_incorrect_to_correct_flips_mean_per_draw"]
              for row in rows]
    x = np.arange(3)
    width = .34
    axes[1].bar(x - width / 2, aligned, width, color="#CC6677", label="Aligned joint patch")
    axes[1].bar(x + width / 2, random, width, color="#999999", label="Mean random draw")
    axes[1].set_xticks(x, [LABELS[row["model"]] for row in rows])
    axes[1].set_ylabel("Incorrect-to-correct cases (of 150)")
    axes[1].set_title("Behavior-changing patches")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=.18)
    fig.suptitle("Fresh many-random-site causal confirmation", fontsize=13)
    fig.tight_layout()
    path = FIGURES / "causal_random_site_confirmation.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    FIGURES.mkdir(exist_ok=True)
    paths = [prompt_figure()]
    causal_path = ROOT / "results/causal_random_site_confirmation_analysis.json"
    if causal_path.exists():
        paths.append(causal_figure())
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
