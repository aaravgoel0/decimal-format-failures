#!/usr/bin/env python3
"""Analyze the frozen Llama and hosted Gemma reasoning confirmations."""

import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_reasoning_confirmation import (
    FORMS,
    NONCANONICAL,
    bootstrap,
    holm,
    mcnemar,
    policy_comparison,
    policy_summary,
    prediction,
    wilson,
)


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "llama": "reasoning_confirmation_meta-llama-Meta-Llama-3.1-8B-Instruct.jsonl",
    "gemma27": "reasoning_confirmation_mlx-community-gemma-3-text-27b-it-4bit.jsonl",
}
SEEDS = {"llama": 2026100401, "gemma27": 2026100402}


def analyze_model(path, seed):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 800 or len({row["id"] for row in rows}) != 800:
        raise RuntimeError(f"expected 800 unique rows in {path}")
    if any(row.get("error") is not None for row in rows):
        raise RuntimeError(f"execution error in {path}")
    groups = defaultdict(dict)
    for row in rows:
        groups[row["base_id"]][row["form"]] = row
    if len(groups) != 200 or any(set(forms) != set(FORMS) for forms in groups.values()):
        raise RuntimeError(f"invalid grouping in {path}")

    rng = np.random.default_rng(seed)
    secondary_rng = np.random.default_rng(seed + 100)
    report = {
        "model": rows[0]["model"],
        "revision": rows[0].get("model_revision"),
        "provider": rows[0].get("provider"),
        "provider_revision_pin_available": rows[0].get(
            "provider_revision_pin_available", True
        ),
        "source_model": rows[0].get("source_model"),
        "checkpoint_precision": rows[0].get("checkpoint_precision"),
        "n_bases": 200,
        "bootstrap_repetitions": 10000,
        "domains": {},
    }
    primary = []
    for domain in ("finqa", "tatqa", "pooled"):
        selected = [
            forms
            for forms in groups.values()
            if domain == "pooled" or next(iter(forms.values()))["domain"] == domain
        ]
        domain_report = {"n_bases": len(selected), "forms": {}, "paired_vs_canonical": {}}
        for form in FORMS:
            outcomes = [bool(forms[form]["correct"]) for forms in selected]
            domain_report["forms"][form] = {
                "accuracy": float(np.mean(outcomes)),
                "correct": int(sum(outcomes)),
                "total": len(outcomes),
                "wilson95": wilson(sum(outcomes), len(outcomes)),
                "parse_failures": sum(
                    forms[form]["parse_status"] != "ok" for forms in selected
                ),
            }
        canonical = [bool(forms["canonical"]["correct"]) for forms in selected]
        for form in NONCANONICAL:
            comparison = [bool(forms[form]["correct"]) for forms in selected]
            record = {
                "domain": domain,
                "form": form,
                "delta_accuracy": float(np.mean(comparison) - np.mean(canonical)),
                "cluster_bootstrap95": bootstrap(
                    [int(b) - int(a) for a, b in zip(canonical, comparison)], rng
                ),
                "mcnemar": mcnemar(canonical, comparison),
            }
            domain_report["paired_vs_canonical"][form] = record
            if domain != "pooled":
                primary.append(record)
        predictions = [[prediction(forms[form]) for form in FORMS] for forms in selected]
        domain_report["four_form_prediction_disagreement"] = float(
            np.mean([len(set(items)) > 1 for items in predictions])
        )
        domain_report["four_form_robust_accuracy"] = float(
            np.mean([all(forms[form]["correct"] for form in FORMS) for forms in selected])
        )
        domain_report["policies"] = {
            name: policy_summary(selected, name)
            for name in ("original", "selective", "blanket")
        }
        domain_report["selective_vs_original"] = policy_comparison(
            selected, "original", "selective", rng
        )
        domain_report["blanket_vs_selective_secondary"] = policy_comparison(
            selected, "selective", "blanket", secondary_rng
        )
        report["domains"][domain] = domain_report
    holm(primary)
    report["primary_holm_family"] = primary
    report["primary_significant_contrasts"] = sum(
        row["mcnemar"]["p_holm"] < 0.05 for row in primary
    )
    return report


def main():
    reports = {
        key: analyze_model(ROOT / "results" / filename, SEEDS[key])
        for key, filename in FILES.items()
    }

    combined = []
    for key, report in reports.items():
        for record in report["primary_holm_family"]:
            item = copy.deepcopy(record)
            item["model_key"] = key
            combined.append(item)
    holm(combined)
    new_model_lookup = {
        (row["model_key"], row["domain"], row["form"]): row["mcnemar"]["p_holm"]
        for row in combined
    }
    for key, report in reports.items():
        for row in report["primary_holm_family"]:
            row["mcnemar"]["p_holm_new_model_family"] = new_model_lookup[
                (key, row["domain"], row["form"])
            ]

    qwen = json.loads((ROOT / "results/reasoning_confirmation_analysis.json").read_text())
    all_reports = {"qwen": qwen, **reports}
    summary = {}
    for key, report in all_reports.items():
        pooled = report["domains"]["pooled"]
        summary[key] = {
            "model": report["model"],
            "canonical_accuracy": pooled["forms"]["canonical"]["accuracy"],
            "scientific_accuracy": pooled["forms"]["scientific"]["accuracy"],
            "scientific_minus_canonical": pooled["paired_vs_canonical"]["scientific"]["delta_accuracy"],
            "scientific_minus_canonical_bootstrap95": pooled["paired_vs_canonical"]["scientific"]["cluster_bootstrap95"],
            "four_form_prediction_disagreement": pooled["four_form_prediction_disagreement"],
            "four_form_robust_accuracy": pooled["four_form_robust_accuracy"],
            "selective_minus_original": pooled["selective_vs_original"]["delta_accuracy"],
            "selective_minus_original_bootstrap95": pooled["selective_vs_original"]["cluster_bootstrap95"],
            "within_model_primary_significant_contrasts": report["primary_significant_contrasts"],
        }
    output = {
        "models": reports,
        "secondary_new_model_holm_family": combined,
        "secondary_new_model_significant_contrasts": sum(
            row["mcnemar"]["p_holm"] < 0.05 for row in combined
        ),
        "descriptive_three_model_summary": summary,
    }
    path = ROOT / "results/reasoning_cross_model_analysis.json"
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
