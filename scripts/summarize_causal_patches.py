#!/usr/bin/env python3
"""Recompute discovery-selected, held-out causal summaries from raw CSVs."""
import csv
import json
import random
from pathlib import Path

FILES = [
    "results/causal_patch_llama.csv",
    "results/causal_patch_Qwen-Qwen3-4B-Instruct-2507.csv",
    "results/causal_patch_google-gemma-2-9b-it.csv",
]


def mean(values):
    return sum(values) / len(values)


def bootstrap(values, seed=7401, repetitions=10_000):
    rng = random.Random(seed)
    draws = sorted(mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(repetitions))
    return [draws[int(.025 * repetitions)], draws[int(.975 * repetitions)]]


def summarize(path):
    rows = list(csv.DictReader(open(path)))
    for row in rows:
        row["case"], row["layer"] = int(row["case"]), int(row["layer"])
        row["margin_effect"] = float(row["margin_effect"])
        row["target_correct"] = row["target_correct"].lower() == "true"
        row["patched_correct"] = row["patched_correct"].lower() == "true"
    model, revision = rows[0]["model"], rows[0]["revision"]
    layers = sorted({row["layer"] for row in rows})
    discovery = [row for row in rows if row["split"] == "discovery"]
    scores = {}
    for layer in layers:
        number = [r["margin_effect"] for r in discovery if r["layer"] == layer and r["control"] == "number_tokens"]
        random_control = [r["margin_effect"] for r in discovery if r["layer"] == layer and r["control"] == "random_positions"]
        scores[layer] = mean(number) - mean(random_control)
    chosen = max(scores, key=scores.get)
    heldout = [row for row in rows if row["split"] == "heldout" and row["layer"] == chosen]
    by_case = {(r["case"], r["control"]): r for r in heldout}
    cases = sorted({r["case"] for r in heldout})
    contrasts = [by_case[c, "number_tokens"]["margin_effect"] - by_case[c, "random_positions"]["margin_effect"] for c in cases]
    number = [by_case[c, "number_tokens"]["margin_effect"] for c in cases]
    random_control = [by_case[c, "random_positions"]["margin_effect"] for c in cases]
    answer = [by_case[c, "answer_position"]["margin_effect"] for c in cases]
    flips = [by_case[c, "number_tokens"]["patched_correct"] and not by_case[c, "number_tokens"]["target_correct"] for c in cases]
    return {"model": model, "revision": revision, "n_layers": len(layers),
            "n_discovery_cases": 20, "n_heldout_cases": 25,
            "chosen_layer_zero_based": chosen, "discovery_selection_contrast": scores[chosen],
            "heldout_number_patch_mean_effect": mean(number),
            "heldout_random_patch_mean_effect": mean(random_control),
            "heldout_answer_patch_mean_effect": mean(answer),
            "heldout_difference_in_effects": mean(contrasts),
            "heldout_difference_bootstrap_95_ci": bootstrap(contrasts),
            "heldout_number_patch_flip_rate": mean(flips),
            "claim_criterion_passed": bootstrap(contrasts)[0] > 0}


def main():
    summaries = [summarize(Path(path)) for path in FILES if Path(path).exists()]
    Path("results/causal_patch_cross_model.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
