#!/usr/bin/env python3
"""Build publication figures directly from controlled mechanistic outputs."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = {"meta-llama-Meta-Llama-3.1-8B-Instruct": "#d95f02",
          "Qwen-Qwen3-4B-Instruct-2507": "#1b9e77", "google-gemma-2-9b-it": "#7570b3"}
LABELS = {"meta-llama-Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B",
          "Qwen-Qwen3-4B-Instruct-2507": "Qwen3 4B", "google-gemma-2-9b-it": "Gemma 2 9B"}


def probe_figure():
    data = json.load(open("results/cross_format_probes.json"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=False)
    for row, position in enumerate(("numeral_final", "answer")):
        for col, task in enumerate(("value", "equality")):
            ax = axes[row, col]
            for item in data:
                if item["position"] != position or item["task"] != task:
                    continue
                ax.plot([v["layer"] for v in item["layers"]], [v["metric"] for v in item["layers"]],
                        color=COLORS[item["model"]], linestyle="-" if item["direction"] == "canonical_to_padded" else "--",
                        linewidth=1.5, label=f"{LABELS[item['model']]} · {'C→P' if item['direction'].startswith('canonical') else 'P→C'}")
            ax.set_title(f"{position.replace('_', ' ')} · {task}")
            ax.set_xlabel("Layer (zero-based)")
            ax.set_ylabel("Spearman ρ" if task == "value" else "Balanced accuracy")
            if task == "equality": ax.axhline(.5, color="#888", linewidth=.8, linestyle=":")
            else: ax.axhline(0, color="#888", linewidth=.8, linestyle=":")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Cross-format probe transfer on strictly held-out numerical values")
    fig.tight_layout()
    fig.savefig("figures/mechanistic_probe_layers.png", dpi=200)
    plt.close(fig)


def geometry_figure():
    data = json.load(open("results/representation_geometry.json"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row, position in enumerate(("numeral_final", "answer")):
        for col, target in enumerate((1, 2)):
            ax = axes[row, col]
            for item in data:
                if item["activation_position"] != position or item["target_position"] != target: continue
                ax.plot([v["layer"] for v in item["layers"]], [v["equivalence_minus_nearby_cosine"] for v in item["layers"]],
                        color=COLORS[item["model"]], label=LABELS[item["model"]])
            ax.axhline(0, color="#888", linewidth=.8, linestyle=":")
            ax.set_title(f"{'Numeral-final' if position == 'numeral_final' else 'Answer'} · target {'first' if target == 1 else 'second'}")
            if row == 1: ax.set_xlabel("Layer (zero-based)")
            if col == 0: ax.set_ylabel("Equivalent − nearby cosine")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Held-out residualized representation geometry")
    fig.savefig("figures/mechanistic_geometry_layers.png", dpi=200)
    plt.close(fig)


def causal_figure():
    paths = [Path("results/causal_patch_llama.csv"), Path("results/causal_patch_Qwen-Qwen3-4B-Instruct-2507.csv"),
             Path("results/causal_patch_google-gemma-2-9b-it.csv")]
    summaries = {s["model"]: s for s in json.load(open("results/causal_patch_cross_model.json"))}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, path in zip(axes, paths):
        rows = list(csv.DictReader(open(path))); model = rows[0]["model"]
        for control, color, label in (("number_tokens", "#d95f02", "Aligned numerals"),
                                      ("random_positions", "#7570b3", "Random positions"),
                                      ("answer_position", "#1b9e77", "Answer position")):
            values = defaultdict(list)
            for row in rows:
                if row["split"] == "heldout" and row["control"] == control:
                    values[int(row["layer"])].append(float(row["margin_effect"]))
            layers = sorted(values)
            ax.plot(layers, [sum(values[layer]) / len(values[layer]) for layer in layers], color=color, label=label)
        ax.axhline(0, color="#888", linewidth=.8, linestyle=":")
        ax.axvline(summaries[model]["chosen_layer_zero_based"], color="#222", linewidth=.9, linestyle="--")
        ax.set_title(LABELS[model.replace("/", "-")])
        ax.set_xlabel("Layer (zero-based)")
    axes[0].set_ylabel("Held-out equality-margin change")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Causal residual interchange with equal-size random controls")
    fig.tight_layout()
    fig.savefig("figures/causal_patch_cross_model.png", dpi=200)
    plt.close(fig)


def main():
    probe_figure(); geometry_figure(); causal_figure()
    print("wrote mechanistic figures")


if __name__ == "__main__": main()
