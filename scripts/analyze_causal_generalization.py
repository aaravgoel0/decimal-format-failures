#!/usr/bin/env python3
"""Inference for fixed-site causal-generalization experiments."""
import csv
import json
from pathlib import Path

import numpy as np

FILES = [
    Path("results/causal_generalization_Qwen-Qwen3-4B-Instruct-2507.csv"),
    Path("results/causal_generalization_google-gemma-2-9b-it.csv"),
]


def interval(values, seed, repetitions=10_000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, float)
    means = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def summarize(rows, subset, seed):
    selected = [row for row in rows if subset is None or row["template"] == subset]
    by_case = {(int(row["case"]), row["control"]): row for row in selected}
    cases = sorted({int(row["case"]) for row in selected})
    easy = [float(by_case[case, "easy_number_tokens"]["margin_effect"]) for case in cases]
    easy_random = [float(by_case[case, "easy_random_positions"]["margin_effect"]) for case in cases]
    donor = [float(by_case[case, "donor_number_tokens"]["margin_effect"]) for case in cases]
    donor_random = [float(by_case[case, "donor_random_positions"]["margin_effect"]) for case in cases]
    easy_contrast = np.asarray(easy) - np.asarray(easy_random)
    donor_contrast = np.asarray(donor) - np.asarray(donor_random)
    return {
        "template": subset or "pooled", "n_cases": len(cases),
        "easy_number_mean_effect": float(np.mean(easy)),
        "easy_random_mean_effect": float(np.mean(easy_random)),
        "easy_aligned_minus_random": float(np.mean(easy_contrast)),
        "easy_contrast_bootstrap_95_ci": interval(easy_contrast, seed),
        "easy_incorrect_to_correct_flip_rate": float(np.mean([
            by_case[case, "easy_number_tokens"]["hard_correct"].lower() == "false" and
            by_case[case, "easy_number_tokens"]["patched_correct"].lower() == "true" for case in cases])),
        "donor_number_mean_effect": float(np.mean(donor)),
        "donor_random_mean_effect": float(np.mean(donor_random)),
        "donor_aligned_minus_random": float(np.mean(donor_contrast)),
        "donor_contrast_bootstrap_95_ci": interval(donor_contrast, seed + 1),
        "donor_correct_to_incorrect_flip_rate": float(np.mean([
            by_case[case, "donor_number_tokens"]["hard_correct"].lower() == "true" and
            by_case[case, "donor_number_tokens"]["patched_correct"].lower() == "false" for case in cases])),
    }


def main():
    output = []
    for path in FILES:
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 600
        model = rows[0]["model"]
        analyses = [summarize(rows, None, 81201),
                    summarize(rows, "relation_statements", 81211),
                    summarize(rows, "direct_choice", 81221)]
        primary_pass = all(item["easy_contrast_bootstrap_95_ci"][0] > 0 for item in analyses)
        donor_pass = all(item["donor_contrast_bootstrap_95_ci"][1] < 0 for item in analyses)
        output.append({"model": model, "revision": rows[0]["revision"],
                       "fixed_layer_zero_based": int(rows[0]["layer"]),
                       "primary_generalization_passed": primary_pass,
                       "selective_donor_corruption_passed": donor_pass,
                       "analyses": analyses})
    Path("results/causal_generalization_analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
