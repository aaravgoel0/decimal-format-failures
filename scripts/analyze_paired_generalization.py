#!/usr/bin/env python3
"""Paired inference for the frozen order and label-position experiment."""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "results").glob("paired_order_generalization_*.jsonl"))


def exact_mcnemar(left_only, right_only):
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def interval(values, rng, repetitions=10_000):
    values = np.asarray(values, float)
    draws = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def summarize(rows, subset, rng, outcome="correct"):
    chosen = [row for row in rows if subset(row)]
    by = {(row["base_id"], row["padded_position"]): row for row in chosen}
    base_ids = sorted({row["base_id"] for row in chosen})
    if any((base, 1) not in by or (base, 2) not in by for base in base_ids):
        raise RuntimeError("incomplete order pair")
    first = np.asarray([int(by[base, 1][outcome]) for base in base_ids])
    second = np.asarray([int(by[base, 2][outcome]) for base in base_ids])
    diff = first - second
    first_only = int(((first == 1) & (second == 0)).sum())
    second_only = int(((first == 0) & (second == 1)).sum())
    return {
        "n_pairs": len(base_ids),
        "padded_first_accuracy": float(first.mean()),
        "padded_second_accuracy": float(second.mean()),
        "paired_first_minus_second": float(diff.mean()),
        "paired_bootstrap_95_ci": interval(diff, rng),
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "mcnemar_p_two_sided": exact_mcnemar(first_only, second_only),
    }


def main():
    if len(FILES) != 3:
        raise RuntimeError(f"expected three completed model files, found {len(FILES)}")
    output = []
    pair_effects = {"strict_exact": {}, "constrained_label": {}}
    for model_index, path in enumerate(FILES):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 600 or len({row["id"] for row in rows}) != 600:
            raise RuntimeError(f"invalid result count: {path}")
        if any(row["parse_status"] == "error" for row in rows):
            raise RuntimeError(f"execution errors remain: {path}")
        rng = np.random.default_rng(91_121 + model_index * 100)
        overall = summarize(rows, lambda row: True, rng, "correct")
        constrained_overall = summarize(rows, lambda row: True, rng, "constrained_correct")
        strata = []
        constrained_strata = []
        for field, values in (("template_index", (0, 1)), ("answer", (1, 2, 3)),
                              ("digit", range(10)), ("zeros", range(1, 6))):
            for value in values:
                selector = lambda row, f=field, v=value: row[f] == v
                cell = summarize(rows, selector, rng, "correct")
                cell.update({"field": field, "value": value})
                strata.append(cell)
                constrained_cell = summarize(rows, selector, rng, "constrained_correct")
                constrained_cell.update({"field": field, "value": value})
                constrained_strata.append(constrained_cell)
        base_ids = sorted({row["base_id"] for row in rows})
        by = {(row["base_id"], row["padded_position"]): row for row in rows}
        pair_effects["strict_exact"][rows[0]["model"]] = np.asarray([
            int(by[base, 1]["correct"]) - int(by[base, 2]["correct"]) for base in base_ids
        ])
        pair_effects["constrained_label"][rows[0]["model"]] = np.asarray([
            int(by[base, 1]["constrained_correct"]) - int(by[base, 2]["constrained_correct"])
            for base in base_ids
        ])
        output.append({
            "model": rows[0]["model"],
            "revision": rows[0]["model_revision"],
            "overall_accuracy": float(np.mean([row["correct"] for row in rows])),
            "invalid_or_nonexact_responses": int(sum(row["parse_status"] != "exact" for row in rows)),
            "constrained_accuracy": float(np.mean([row["constrained_correct"] for row in rows])),
            "exact_constrained_agreement": float(np.mean([row["prediction"] == row["constrained_prediction"] for row in rows])),
            "paired_order": overall,
            "constrained_paired_order": constrained_overall,
            "strata": strata,
            "constrained_strata": constrained_strata,
        })
    rng = np.random.default_rng(91_999)
    comparisons = []
    for outcome, effects in pair_effects.items():
        models = list(effects)
        for i, left in enumerate(models):
            for right in models[i + 1:]:
                values = effects[left] - effects[right]
                comparisons.append({
                    "outcome": outcome,
                    "left_model": left,
                    "right_model": right,
                    "difference_in_order_effects": float(values.mean()),
                    "paired_bootstrap_95_ci": interval(values, rng),
                })
    payload = {"models": output, "cross_model_order_effects": comparisons}
    (ROOT / "results/paired_order_generalization_analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
