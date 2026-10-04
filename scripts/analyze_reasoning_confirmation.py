#!/usr/bin/env python3
"""Analyze the frozen Qwen reasoning-enabled downstream confirmation."""

import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
FORMS = ("canonical", "padded", "leading_zero", "scientific")
NONCANONICAL = FORMS[1:]
POLICIES = {
    "original": {"padded": "padded", "leading_zero": "leading_zero", "scientific": "scientific"},
    "selective": {"padded": "padded", "leading_zero": "canonical", "scientific": "canonical"},
    "blanket": {"padded": "canonical", "leading_zero": "canonical", "scientific": "canonical"},
}


def wilson(successes, total, z=1.959963984540054):
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def bootstrap(values, rng, draws=10000):
    values = np.asarray(values, dtype=float)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(sampled, [0.025, 0.975])]


def mcnemar(first, second):
    improve = sum((not a) and b for a, b in zip(first, second))
    regress = sum(a and (not b) for a, b in zip(first, second))
    discordant = improve + regress
    p = 1.0 if not discordant else float(binomtest(min(improve, regress), discordant, 0.5).pvalue)
    return {"improve": improve, "regress": regress, "discordant": discordant, "p_raw": p}


def holm(records):
    ranked = sorted(enumerate(records), key=lambda pair: pair[1]["mcnemar"]["p_raw"])
    running = 0.0
    for rank, (index, record) in enumerate(ranked):
        adjusted = min(1.0, record["mcnemar"]["p_raw"] * (len(records) - rank))
        running = max(running, adjusted)
        records[index]["mcnemar"]["p_holm"] = running


def prediction(row):
    if row["parse_status"] != "ok":
        return None
    return Decimal(str(row["prediction_value"])), bool(row["prediction_is_percent"])


def policy_summary(selected, policy):
    rows = [[forms[POLICIES[policy][form]] for form in NONCANONICAL] for forms in selected]
    outcomes = [bool(row["correct"]) for group in rows for row in group]
    predictions = [[prediction(row) for row in group] for group in rows]
    return {
        "accuracy": float(np.mean(outcomes)),
        "correct": int(sum(outcomes)),
        "total": len(outcomes),
        "robust_accuracy": float(np.mean([all(row["correct"] for row in group) for group in rows])),
        "prediction_disagreement": float(np.mean([len(set(items)) > 1 for items in predictions])),
    }


def policy_comparison(selected, old_policy, new_policy, rng):
    deltas = []
    old_rows, new_rows = [], []
    for forms in selected:
        old = [bool(forms[POLICIES[old_policy][form]]["correct"]) for form in NONCANONICAL]
        new = [bool(forms[POLICIES[new_policy][form]]["correct"]) for form in NONCANONICAL]
        old_rows.extend(old)
        new_rows.extend(new)
        deltas.append(float(np.mean(new) - np.mean(old)))
    return {
        "delta_accuracy": float(np.mean(deltas)),
        "cluster_bootstrap95": bootstrap(deltas, rng),
        "mcnemar_row_sensitivity": mcnemar(old_rows, new_rows),
    }


def main():
    path = ROOT / "results/reasoning_confirmation_Qwen-Qwen3-4B-Instruct-2507.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 800 or len({row["id"] for row in rows}) != 800:
        raise RuntimeError("expected 800 unique reasoning-confirmation rows")
    if any(row.get("error") is not None for row in rows):
        raise RuntimeError("reasoning-confirmation execution error")
    groups = defaultdict(dict)
    for row in rows:
        groups[row["base_id"]][row["form"]] = row
    if len(groups) != 200 or any(set(forms) != set(FORMS) for forms in groups.values()):
        raise RuntimeError("invalid reasoning-confirmation grouping")

    rng = np.random.default_rng(2026100301)
    secondary_rng = np.random.default_rng(2026100302)
    report = {"model": rows[0]["model"], "revision": rows[0]["model_revision"], "n_bases": 200,
              "bootstrap_repetitions": 10000, "domains": {}}
    primary = []
    for domain in ("finqa", "tatqa", "pooled"):
        selected = [forms for forms in groups.values()
                    if domain == "pooled" or next(iter(forms.values()))["domain"] == domain]
        domain_report = {"n_bases": len(selected), "forms": {}, "paired_vs_canonical": {}}
        for form in FORMS:
            outcomes = [bool(forms[form]["correct"]) for forms in selected]
            domain_report["forms"][form] = {
                "accuracy": float(np.mean(outcomes)), "correct": int(sum(outcomes)),
                "total": len(outcomes), "wilson95": wilson(sum(outcomes), len(outcomes)),
                "parse_failures": sum(forms[form]["parse_status"] != "ok" for forms in selected),
            }
        canonical = [bool(forms["canonical"]["correct"]) for forms in selected]
        for form in NONCANONICAL:
            comparison = [bool(forms[form]["correct"]) for forms in selected]
            record = {"domain": domain, "form": form,
                      "delta_accuracy": float(np.mean(comparison) - np.mean(canonical)),
                      "cluster_bootstrap95": bootstrap([int(b)-int(a) for a,b in zip(canonical, comparison)], rng),
                      "mcnemar": mcnemar(canonical, comparison)}
            domain_report["paired_vs_canonical"][form] = record
            if domain != "pooled": primary.append(record)
        predictions = [[prediction(forms[form]) for form in FORMS] for forms in selected]
        domain_report["four_form_prediction_disagreement"] = float(np.mean([len(set(x)) > 1 for x in predictions]))
        domain_report["four_form_robust_accuracy"] = float(np.mean([all(forms[f]["correct"] for f in FORMS) for forms in selected]))
        domain_report["policies"] = {name: policy_summary(selected, name) for name in POLICIES}
        domain_report["selective_vs_original"] = policy_comparison(selected, "original", "selective", rng)
        domain_report["blanket_vs_selective_secondary"] = policy_comparison(selected, "selective", "blanket", secondary_rng)
        report["domains"][domain] = domain_report
    holm(primary)
    report["primary_holm_family"] = primary
    report["primary_significant_contrasts"] = sum(row["mcnemar"]["p_holm"] < 0.05 for row in primary)
    output = ROOT / "results/reasoning_confirmation_analysis.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
