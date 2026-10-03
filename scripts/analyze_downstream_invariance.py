#!/usr/bin/env python3
"""Paired analysis for the downstream format-invariance benchmark."""

import json
import math
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODELS = (
    "meta-llama-Meta-Llama-3.1-8B-Instruct",
    "Qwen-Qwen3-4B-Instruct-2507",
    "google-gemma-2-9b-it",
)


def interval(values, rng, repetitions=10_000):
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[draws].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def wilson(successes, n, z=1.959963984540054):
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def mcnemar(first_only, second_only):
    n = first_only + second_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(first_only, second_only) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def normalized_prediction(row):
    if row.get("parse_status") != "ok":
        return None
    try:
        value = Decimal(str(row.get("prediction_value")))
    except (InvalidOperation, TypeError, ValueError):
        return None
    percent = bool(row.get("prediction_is_percent"))
    return value, percent


def row_correct(row):
    prediction = normalized_prediction(row)
    if prediction is None:
        return False
    value, is_percent = prediction
    try:
        equal = value == Decimal(str(row["answer_value"]))
    except (InvalidOperation, TypeError, ValueError):
        equal = False
    return equal and bool(is_percent) == bool(row["answer_is_percent"])


def summarize(rows, rng):
    by_base = defaultdict(dict)
    for row in rows:
        by_base[row["base_id"]][row["condition"]] = row
    assert all(set(pair) == {"canonical", "padded"} for pair in by_base.values())
    pairs = list(by_base.values())
    canonical = np.array([row_correct(pair["canonical"]) for pair in pairs], dtype=float)
    padded = np.array([row_correct(pair["padded"]) for pair in pairs], dtype=float)
    effect = padded - canonical
    damage = int(np.sum((canonical == 1) & (padded == 0)))
    gain = int(np.sum((canonical == 0) & (padded == 1)))
    disagreements = []
    for pair in pairs:
        left, right = pair["canonical"], pair["padded"]
        left_prediction = normalized_prediction(left)
        right_prediction = normalized_prediction(right)
        disagreements.append(left_prediction != right_prediction)
    disagreements = np.asarray(disagreements, dtype=float)
    return {
        "n_pairs": len(pairs),
        "canonical_accuracy": float(canonical.mean()),
        "canonical_wilson_95_ci": wilson(int(canonical.sum()), len(pairs)),
        "padded_accuracy": float(padded.mean()),
        "padded_wilson_95_ci": wilson(int(padded.sum()), len(pairs)),
        "padded_minus_canonical": float(effect.mean()),
        "paired_bootstrap_95_ci": interval(effect, rng),
        "canonical_correct_padded_incorrect": damage,
        "canonical_incorrect_padded_correct": gain,
        "mcnemar_p_two_sided": mcnemar(damage, gain),
        "prediction_disagreement_rate": float(disagreements.mean()),
        "prediction_disagreement_bootstrap_95_ci": interval(disagreements, rng),
        "parse_failures_canonical": sum(pair["canonical"]["parse_status"] != "ok" for pair in pairs),
        "parse_failures_padded": sum(pair["padded"]["parse_status"] != "ok" for pair in pairs),
    }


def holm(rows):
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["mcnemar_p_two_sided"])
    running = 0.0
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (len(rows) - rank) * row["mcnemar_p_two_sided"])
        running = max(running, adjusted)
        rows[index]["mcnemar_holm_p"] = running
        rows[index]["primary_accuracy_effect_passed"] = (
            running < 0.05 and not (row["paired_bootstrap_95_ci"][0] <= 0 <= row["paired_bootstrap_95_ci"][1]))


def main():
    output = []
    primary = []
    for model_index, safe in enumerate(MODELS):
        path = ROOT / "results" / f"downstream_format_invariance_{safe}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 400 and len({row["id"] for row in rows}) == 400
        assert all(row["checkpoint_precision"] == "native-unquantized" for row in rows)
        failures = [row for row in rows if row["error"] is not None]
        successful = [row for row in rows if row["error"] is None]
        failed_base_ids = {row["base_id"] for row in failures}
        rows = [row for row in successful if row["base_id"] not in failed_base_ids]
        model_result = {
            "model": successful[0]["model"],
            "revision": successful[0]["model_revision"],
            "execution_failures": [
                {"id": row["id"], "error": row["error"]} for row in failures
            ],
            "excluded_incomplete_base_ids": sorted(failed_base_ids),
            "domains": [],
        }
        for domain_index, domain in enumerate(("gsm8k", "finqa")):
            selected = [row for row in rows if row["domain"] == domain]
            result = summarize(selected, np.random.default_rng(2026091700 + model_index * 10 + domain_index))
            result["domain"] = domain
            model_result["domains"].append(result)
            primary.append(result)
        model_result["pooled"] = summarize(rows, np.random.default_rng(2026091790 + model_index))
        output.append(model_result)
    holm(primary)
    passed = sum(row["primary_accuracy_effect_passed"] for row in primary)
    result = {
        "models": output,
        "primary_family_size": 6,
        "primary_cells_passed": passed,
        "any_downstream_accuracy_effect_passed": passed > 0,
        "bootstrap_repetitions": 10_000,
    }
    path = ROOT / "results/downstream_format_invariance_analysis.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(path)


if __name__ == "__main__":
    main()
