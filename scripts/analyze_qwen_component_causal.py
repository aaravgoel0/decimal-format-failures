#!/usr/bin/env python3
"""Analyze discovery-selected Qwen attention and MLP interventions."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INTERVENTIONS = ("selected_heads", "attention", "mlp", "whole_block")


def interval(values, rng, repetitions=10_000):
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), size=(repetitions, len(values)))
    return [float(x) for x in np.quantile(values[draws].mean(axis=1), [0.025, 0.975])]


def summarize(rows, intervention, seed):
    rng = np.random.default_rng(seed)
    aligned = np.array([row["interventions"][intervention]["aligned_effect"] for row in rows])
    random_effects = np.array([row["interventions"][intervention]["random_effects"] for row in rows])
    random_means = random_effects.mean(axis=1)
    centered = aligned - random_means
    null_draws = np.empty(10_000)
    for draw in range(10_000):
        choices = rng.integers(0, random_effects.shape[1], size=len(rows))
        null_draws[draw] = random_effects[np.arange(len(rows)), choices].mean()
    return {
        "n_cases": len(rows),
        "random_sets_per_case": random_effects.shape[1],
        "aligned_mean_effect": float(aligned.mean()),
        "random_case_mean_effect": float(random_means.mean()),
        "aligned_minus_random": float(centered.mean()),
        "aligned_minus_random_bootstrap_95_ci": interval(centered, rng),
        "randomization_p_one_sided": float((1 + np.sum(null_draws >= aligned.mean())) / 10_001),
        "aligned_incorrect_to_correct_flips": int(sum(
            (not row["hard_correct"]) and row["interventions"][intervention]["aligned_flip"]
            for row in rows)),
        "random_incorrect_to_correct_flips_mean_per_draw": float(np.mean([
            np.mean(row["interventions"][intervention]["random_flips"]) for row in rows])),
    }


def main():
    selection = json.loads((ROOT / "results/qwen_component_head_selection.json").read_text())
    rows = [json.loads(line) for line in (ROOT / "results/qwen_component_heldout.jsonl").read_text().splitlines()]
    assert len(rows) == 90 and len({row["id"] for row in rows}) == 90
    assert {row["model"] for row in rows} == {"Qwen/Qwen3-4B-Instruct-2507"}
    assert {row["revision"] for row in rows} == {
        "cdbee75f17c01a7cc42f958dc650907174af0554"
    }
    assert {row["layer"] for row in rows} == {2}
    assert {tuple(row["selected_heads"]) for row in rows} == {
        tuple(selection["selected_heads"])
    }
    assert {template: sum(row["template_index"] == template for row in rows)
            for template in range(3)} == {0: 30, 1: 30, 2: 30}
    for row in rows:
        random_sets = [tuple(positions) for positions in row["random_position_sets"]]
        assert len(random_sets) == 20 and len(set(random_sets)) == 20
        aligned = set(row["aligned_target_positions"])
        assert all(len(positions) == len(aligned) for positions in random_sets)
        assert all(not (aligned & set(positions)) for positions in random_sets)
        assert all((row["sequence_length"] - 1) not in positions for positions in random_sets)
        for intervention in INTERVENTIONS:
            values = row["interventions"][intervention]
            assert len(values["random_effects"]) == 20
            assert np.isfinite(values["aligned_effect"])
            assert np.isfinite(values["random_effects"]).all()
    result = {
        "model": rows[0]["model"],
        "revision": rows[0]["revision"],
        "layer": rows[0]["layer"],
        "selected_heads": selection["selected_heads"],
        "discovery_ranking": selection["head_ranking"],
        "heldout_hard_accuracy": sum(row["hard_correct"] for row in rows) / len(rows),
        "heldout_easy_accuracy": sum(row["easy_correct"] for row in rows) / len(rows),
        "interventions": {},
    }
    for index, intervention in enumerate(INTERVENTIONS):
        summary = summarize(rows, intervention, 2026091780 + index)
        summary["template_strata"] = []
        for template in range(3):
            selected = [row for row in rows if row["template_index"] == template]
            cell = summarize(selected, intervention, 2026091800 + index * 10 + template)
            cell["template_index"] = template
            summary["template_strata"].append(cell)
        result["interventions"][intervention] = summary
    primary = result["interventions"]["selected_heads"]
    result["prespecified_selected_head_passed"] = (
        primary["aligned_minus_random_bootstrap_95_ci"][0] > 0 and
        primary["randomization_p_one_sided"] <= 0.05 and
        all(cell["aligned_minus_random_bootstrap_95_ci"][0] > 0
            for cell in primary["template_strata"]))
    attention = np.array([
        row["interventions"]["attention"]["aligned_effect"] -
        np.mean(row["interventions"]["attention"]["random_effects"]) for row in rows])
    mlp = np.array([
        row["interventions"]["mlp"]["aligned_effect"] -
        np.mean(row["interventions"]["mlp"]["random_effects"]) for row in rows])
    result["attention_minus_mlp_centered"] = float((attention - mlp).mean())
    result["attention_minus_mlp_bootstrap_95_ci"] = interval(
        attention - mlp, np.random.default_rng(2026091899))
    output = ROOT / "results/qwen_component_causal_analysis.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
