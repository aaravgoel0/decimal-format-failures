#!/usr/bin/env python3
"""Analyze held-out broad-format and paired canonicalization results."""
import json
import math
from pathlib import Path

import numpy as np

FILES = sorted(Path("results").glob("format_robustness_*.jsonl"))
FAMILIES = ["negative", "leading_zero", "long_fraction", "scientific", "signed_zero"]


def wilson(successes, n, z=1.959963984540054):
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def paired(rows, family, seed):
    selected = [row for row in rows if family is None or row["family"] == family]
    by_key = {(row["id"], row["condition"]): row for row in selected}
    ids = sorted({row["id"] for row in selected})
    differences = np.asarray([int(by_key[id_, "canonicalized"]["correct"]) - int(by_key[id_, "original"]["correct"]) for id_ in ids])
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(0, len(ids), size=(10_000, len(ids)))].mean(1)
    return {"family": family or "overall", "n": len(ids), "accuracy_change": float(differences.mean()),
            "bootstrap_95_ci": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "incorrect_to_correct": int((differences == 1).sum()),
            "correct_to_incorrect": int((differences == -1).sum())}


def subgroup_cells(rows):
    """Report prespecified design dimensions to expose prompt/label artifacts."""
    output = []
    dimensions = {
        "template_index": [0, 1],
        "equal": [True, False],
        "answer_position": [1, 2, 3],
    }
    for condition in ("original", "canonicalized"):
        for dimension, values in dimensions.items():
            for value in values:
                if dimension == "answer_position":
                    selected = [r for r in rows if r["condition"] == condition and r["answer"] == value]
                else:
                    selected = [r for r in rows if r["condition"] == condition and r[dimension] == value]
                correct = sum(r["correct"] for r in selected)
                output.append({"condition": condition, "dimension": dimension, "value": value,
                               "n": len(selected), "correct": correct,
                               "accuracy": correct / len(selected),
                               "wilson_95_ci": wilson(correct, len(selected))})
    return output


def main():
    output = []
    for model_index, path in enumerate(FILES):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 1000 and len({(row["id"], row["condition"]) for row in rows}) == 1000
        cells = []
        for condition in ("original", "canonicalized"):
            for family in [None] + FAMILIES:
                selected = [row for row in rows if row["condition"] == condition and (family is None or row["family"] == family)]
                correct = sum(row["correct"] for row in selected)
                cells.append({"condition": condition, "family": family or "overall", "n": len(selected),
                              "correct": correct, "accuracy": correct / len(selected),
                              "wilson_95_ci": wilson(correct, len(selected)),
                              "parse_failures": sum(row["prediction"] is None for row in selected),
                              "recovered_parses": sum(row["parse_status"] == "recovered" for row in selected)})
        changes = [paired(rows, None, 82301 + model_index * 100)] + [paired(rows, family, 82311 + model_index * 100 + i) for i, family in enumerate(FAMILIES)]
        overall = changes[0]
        benefit = overall["bootstrap_95_ci"][0] > 0 and all(change["accuracy_change"] >= 0 for change in changes[1:])
        output.append({"model": rows[0]["model"], "revision": rows[0]["revision"],
                       "canonicalization_benefit_criterion_passed": benefit,
                       "cells": cells, "paired_changes": changes,
                       "subgroups": subgroup_cells(rows)})
    Path("results/format_robustness_analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
