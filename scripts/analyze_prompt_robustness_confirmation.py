#!/usr/bin/env python3
"""Analyze the frozen 10-template prompt-robustness confirmation."""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "results").glob("prompt_robustness_confirmation_*.jsonl"))
NBOOT = 10_000


def exact_mcnemar(left_only, right_only):
    total = left_only + right_only
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(left_only, right_only) + 1)) / 2 ** total
    return min(1.0, 2 * tail)


def interval(values, rng):
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(0, len(values), size=(NBOOT, len(values)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired(rows, selector, field, rng):
    chosen = [row for row in rows if selector(row)]
    by = {(row["base_id"], row["padded_position"]): row for row in chosen}
    bases = sorted({row["base_id"] for row in chosen})
    if any((base, 1) not in by or (base, 2) not in by for base in bases):
        raise RuntimeError("incomplete pair")
    first = np.asarray([int(by[base, 1][field]) for base in bases])
    second = np.asarray([int(by[base, 2][field]) for base in bases])
    effects = first - second
    first_only = int(((first == 1) & (second == 0)).sum())
    second_only = int(((first == 0) & (second == 1)).sum())
    return {
        "n_pairs": len(bases),
        "padded_first_accuracy": float(first.mean()),
        "padded_second_accuracy": float(second.mean()),
        "paired_first_minus_second": float(effects.mean()),
        "paired_bootstrap_95_ci": interval(effects, rng),
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "mcnemar_p_two_sided": exact_mcnemar(first_only, second_only),
    }


def logistic_clustered(rows):
    import pandas as pd
    import statsmodels.api as sm

    frame = pd.DataFrame({
        "correct": [int(row["constrained_correct"]) for row in rows],
        "padded_first": [int(row["padded_position"] == 1) for row in rows],
        "template": [str(row["template_index"]) for row in rows],
        "label": [str(row["equal_label"]) for row in rows],
        "digit": [str(row["digit"]) for row in rows],
        "zeros": [str(row["zeros"]) for row in rows],
        "base_id": [row["base_id"] for row in rows],
    })
    design = pd.get_dummies(frame[["padded_first", "template", "label", "digit", "zeros"]],
                            columns=["template", "label", "digit", "zeros"], drop_first=True,
                            dtype=float)
    design = sm.add_constant(design, has_constant="add")
    fit = sm.GLM(frame["correct"], design, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": frame["base_id"]})
    name = "padded_first"
    return {
        "coefficient": float(fit.params[name]),
        "cluster_robust_se": float(fit.bse[name]),
        "z": float(fit.tvalues[name]),
        "p_two_sided": float(fit.pvalues[name]),
        "odds_ratio": float(np.exp(fit.params[name])),
        "odds_ratio_95_ci": [float(np.exp(fit.conf_int().loc[name, 0])),
                              float(np.exp(fit.conf_int().loc[name, 1]))],
        "n_rows": int(fit.nobs),
        "n_base_clusters": int(frame["base_id"].nunique()),
    }


def main():
    if len(FILES) != 3:
        raise RuntimeError(f"expected three complete result files, found {len(FILES)}")
    output = []
    for model_index, path in enumerate(FILES):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 3000 or len({row["id"] for row in rows}) != 3000:
            raise RuntimeError(f"invalid result count in {path}")
        if any(row.get("error") is not None or row.get("constrained_prediction") is None
               for row in rows):
            raise RuntimeError(f"execution error remains in {path}")
        rng = np.random.default_rng(2026091400 + model_index)
        pooled = paired(rows, lambda row: True, "constrained_correct", rng)
        templates = []
        for template in range(10):
            value = paired(rows, lambda row, template=template:
                           row["template_index"] == template,
                           "constrained_correct", rng)
            value["template_index"] = template
            templates.append(value)
        pooled_direction = np.sign(pooled["paired_first_minus_second"])
        consistent = 0
        for value in templates:
            low, high = value["paired_bootstrap_95_ci"]
            if pooled_direction < 0 and high < 0:
                consistent += 1
            elif pooled_direction > 0 and low > 0:
                consistent += 1
        strata = []
        for field, values in (("equal_label", (1, 2, 3)), ("digit", range(10)),
                              ("zeros", range(1, 6))):
            for value in values:
                cell = paired(rows, lambda row, field=field, value=value:
                              row[field] == value, "constrained_correct", rng)
                cell.update({"field": field, "value": value})
                strata.append(cell)
        token_strata = []
        token_fields = ("canonical_token_count", "padded_token_count",
                        "token_count_difference", "token_lcs_ratio")
        for field in token_fields:
            for value in sorted({row[field] for row in rows}):
                selected = [row for row in rows if row[field] == value]
                if len({row["base_id"] for row in selected}) < 20:
                    continue
                cell = paired(rows, lambda row, field=field, value=value:
                              row[field] == value, "constrained_correct", rng)
                cell.update({"field": field, "value": value})
                token_strata.append(cell)
        effects = [value["paired_first_minus_second"] for value in templates]
        output.append({
            "model": rows[0]["model"],
            "revision": rows[0]["model_revision"],
            "n_prompts": len(rows),
            "n_base_pairs": 1500,
            "primary_constrained_paired_order": pooled,
            "prompt_specific": templates,
            "prompt_effect_median": float(np.median(effects)),
            "prompt_effect_minimum": float(np.min(effects)),
            "prompt_effect_maximum": float(np.max(effects)),
            "prompt_intervals_same_direction": consistent,
            "prespecified_prompt_generalization_passed": consistent >= 8 and
                ((pooled_direction < 0 and pooled["paired_bootstrap_95_ci"][1] < 0) or
                 (pooled_direction > 0 and pooled["paired_bootstrap_95_ci"][0] > 0)),
            "outcome_definition": "argmax over exact logits for single-token labels 1, 2, and 3",
            "free_text_generation": "not run by prespecified operational amendment",
            "strata": strata,
            "tokenization_strata": token_strata,
            "clustered_logistic_padded_first": logistic_clustered(rows),
        })
    path = ROOT / "results/prompt_robustness_confirmation_analysis.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
