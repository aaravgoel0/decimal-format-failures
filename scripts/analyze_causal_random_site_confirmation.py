#!/usr/bin/env python3
"""Analyze fresh fixed-layer interventions against many random-site controls."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "results").glob("causal_random_site_confirmation_*.jsonl"))
COMPONENTS = ("short", "padded", "joint")
NBOOT = 10_000


def interval(values, rng):
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(0, len(values), size=(NBOOT, len(values)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(rows, component, rng):
    aligned = np.asarray([row["interventions"][component]["aligned_effect"] for row in rows],
                         dtype=float)
    random_effects = np.asarray([row["interventions"][component]["random_effects"]
                                 for row in rows], dtype=float)
    if random_effects.shape != (len(rows), 50):
        raise RuntimeError(f"wrong random-effect shape {random_effects.shape}")
    random_case_mean = random_effects.mean(axis=1)
    centered = aligned - random_case_mean
    null_indices = rng.integers(0, 50, size=(NBOOT, len(rows)))
    null_means = random_effects[np.arange(len(rows))[None, :], null_indices].mean(axis=1)
    aligned_flips = np.asarray([row["interventions"][component]["aligned_flip"]
                                for row in rows], dtype=float)
    random_flips = np.asarray([row["interventions"][component]["random_flips"]
                               for row in rows], dtype=float)
    upper = np.quantile(random_effects, 0.95, axis=1)
    result = {
        "n_cases": len(rows),
        "random_sets_per_case": 50,
        "aligned_mean_effect": float(aligned.mean()),
        "aligned_bootstrap_95_ci": interval(aligned, rng),
        "random_case_mean_effect": float(random_case_mean.mean()),
        "random_mean_bootstrap_95_ci": interval(random_case_mean, rng),
        "aligned_minus_random": float(centered.mean()),
        "aligned_minus_random_bootstrap_95_ci": interval(centered, rng),
        "randomization_p_one_sided": float((1 + np.sum(null_means >= aligned.mean())) /
                                            (NBOOT + 1)),
        "aligned_incorrect_to_correct_flips": int(aligned_flips.sum()),
        "random_incorrect_to_correct_flips_mean_per_draw": float(random_flips.sum(axis=0).mean()),
        "cases_aligned_at_or_above_random_95th_percentile": int(np.sum(aligned >= upper)),
        "fraction_cases_aligned_at_or_above_random_95th_percentile": float(np.mean(aligned >= upper)),
    }
    low, high = result["aligned_minus_random_bootstrap_95_ci"]
    result["prespecified_primary_passed"] = (component == "joint" and low > 0 and
                                              result["randomization_p_one_sided"] <= 0.05)
    return result


def main():
    if len(FILES) != 3:
        raise RuntimeError(f"expected three complete model files, found {len(FILES)}")
    output = []
    centered_by_model = {}
    for model_index, path in enumerate(FILES):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 150 or len({row["id"] for row in rows}) != 150:
            raise RuntimeError(f"invalid result count in {path}")
        if any(row.get("status") != "complete" for row in rows):
            raise RuntimeError(f"incomplete row in {path}")
        rng = np.random.default_rng(2026091500 + model_index)
        components = {component: summarize(rows, component, rng) for component in COMPONENTS}
        component_comparisons = []
        for left, right in (("short", "padded"), ("joint", "short"), ("joint", "padded")):
            aligned_difference = np.asarray([
                row["interventions"][left]["aligned_effect"] -
                row["interventions"][right]["aligned_effect"] for row in rows
            ])
            centered_difference = np.asarray([
                (row["interventions"][left]["aligned_effect"] -
                 np.mean(row["interventions"][left]["random_effects"])) -
                (row["interventions"][right]["aligned_effect"] -
                 np.mean(row["interventions"][right]["random_effects"])) for row in rows
            ])
            component_comparisons.append({
                "left": left,
                "right": right,
                "aligned_effect_difference": float(aligned_difference.mean()),
                "aligned_effect_difference_bootstrap_95_ci": interval(aligned_difference, rng),
                "centered_difference": float(centered_difference.mean()),
                "centered_difference_bootstrap_95_ci": interval(centered_difference, rng),
            })
        templates = []
        for template in range(3):
            selected = [row for row in rows if row["template_index"] == template]
            for component in COMPONENTS:
                value = summarize(selected, component, rng)
                value.update({"template_index": template, "component": component})
                templates.append(value)
        joint_centered = np.asarray([
            row["interventions"]["joint"]["aligned_effect"] -
            np.mean(row["interventions"]["joint"]["random_effects"]) for row in rows
        ])
        centered_by_model[rows[0]["model"]] = joint_centered
        output.append({
            "model": rows[0]["model"],
            "revision": rows[0]["revision"],
            "fixed_layer_zero_based": rows[0]["layer"],
            "hard_accuracy": float(np.mean([row["hard_correct"] for row in rows])),
            "easy_accuracy": float(np.mean([row["easy_correct"] for row in rows])),
            "components": components,
            "component_comparisons": component_comparisons,
            "template_strata": templates,
        })
    rng = np.random.default_rng(2026091599)
    cross_model = []
    models = list(centered_by_model)
    for index, left in enumerate(models):
        for right in models[index + 1:]:
            difference = centered_by_model[left] - centered_by_model[right]
            cross_model.append({
                "left_model": left,
                "right_model": right,
                "joint_centered_difference": float(difference.mean()),
                "bootstrap_95_ci": interval(difference, rng),
            })
    payload = {"models": output, "cross_model_joint_comparisons": cross_model}
    path = ROOT / "results/causal_random_site_confirmation_analysis.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

