#!/usr/bin/env python3
"""Analyze the frozen fresh downstream mitigation confirmation."""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FORMS = ("canonical", "padded", "leading_zero", "scientific")
NONCANONICAL = FORMS[1:]
MODELS = {
    "llama": (
        "meta-llama/Llama-3.1-8B-Instruct",
        "0e9e39f249a16976918f6564b8830bc894c89659",
    ),
    "qwen": (
        "Qwen/Qwen3-4B-Instruct-2507",
        "cdbee75f17c01a7cc42f958dc650907174af0554",
    ),
    "gemma": (
        "google/gemma-2-9b-it",
        "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    ),
}
POLICY_SOURCE = {
    "original": {
        "padded": "padded",
        "leading_zero": "leading_zero",
        "scientific": "scientific",
    },
    "selective": {
        "padded": "padded",
        "leading_zero": "canonical",
        "scientific": "canonical",
    },
    "blanket": {
        "padded": "canonical",
        "leading_zero": "canonical",
        "scientific": "canonical",
    },
}


def safe_name(model):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model)


def wilson(successes, total, z=1.959963984540054):
    if not total:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def clustered_bootstrap(base_deltas, rng, draws):
    values = np.asarray(base_deltas, dtype=float)
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    estimates = values[samples].mean(axis=1)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def sign_flip_test(base_deltas, rng, draws):
    values = np.asarray(base_deltas, dtype=float)
    observed = abs(values.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (draws + 1))


def mcnemar(original, comparison):
    improve = sum((not old) and new for old, new in zip(original, comparison))
    regress = sum(old and (not new) for old, new in zip(original, comparison))
    return {"improve": improve, "regress": regress, "discordant": improve + regress}


def holm(records):
    ranked = sorted(enumerate(records), key=lambda pair: pair[1]["p_raw"])
    running = 0.0
    total = len(records)
    for rank, (index, record) in enumerate(ranked):
        adjusted = min(1.0, record["p_raw"] * (total - rank))
        running = max(running, adjusted)
        records[index]["p_holm"] = running


def prediction(row):
    if row.get("parse_status") != "ok":
        return None
    percent = bool(row.get("prediction_is_percent")) if row["answer_is_percent"] else False
    return row.get("prediction_value"), percent


def load_primary(model, revision):
    path = ROOT / "results" / f"downstream_mitigation_{safe_name(model)}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 800 or len({row["id"] for row in rows}) != 800:
        raise RuntimeError(f"{path}: expected 800 unique rows")
    if {row["model"] for row in rows} != {model}:
        raise RuntimeError(f"{path}: model mismatch")
    if {row["model_revision"] for row in rows} != {revision}:
        raise RuntimeError(f"{path}: revision mismatch")
    if {row["checkpoint_precision"] for row in rows} != {"native-unquantized"}:
        raise RuntimeError(f"{path}: primary precision mismatch")
    if any(row.get("error") is not None for row in rows):
        raise RuntimeError(f"{path}: execution error")
    return rows


def summarize_policy(selected, policy):
    source_map = POLICY_SOURCE[policy]
    row_correct = []
    base_robust = []
    base_disagreement = []
    base_unique = []
    for forms in selected:
        rows = [forms[source_map[form]] for form in NONCANONICAL]
        correctness = [bool(row["correct"]) for row in rows]
        predictions = [prediction(row) for row in rows]
        row_correct.extend(correctness)
        base_robust.append(all(correctness))
        base_disagreement.append(len(set(predictions)) > 1)
        base_unique.append(len(set(predictions)))
    successes = sum(row_correct)
    return {
        "correct": successes,
        "total": len(row_correct),
        "accuracy": successes / len(row_correct),
        "wilson95_row_descriptive": wilson(successes, len(row_correct)),
        "robust_accuracy": float(np.mean(base_robust)),
        "prediction_disagreement": float(np.mean(base_disagreement)),
        "mean_unique_predictions": float(np.mean(base_unique)),
    }


def comparison(selected, rng, bootstraps, randomizations):
    original_rows = []
    selective_rows = []
    base_deltas = []
    for forms in selected:
        old = [bool(forms[POLICY_SOURCE["original"][form]]["correct"]) for form in NONCANONICAL]
        new = [bool(forms[POLICY_SOURCE["selective"][form]]["correct"]) for form in NONCANONICAL]
        original_rows.extend(old)
        selective_rows.extend(new)
        base_deltas.append(float(np.mean(new) - np.mean(old)))
    return {
        "delta_accuracy": float(np.mean(base_deltas)),
        "cluster_bootstrap95": clustered_bootstrap(base_deltas, rng, bootstraps),
        "base_sign_flip_p_raw": sign_flip_test(base_deltas, rng, randomizations),
        "mcnemar_row_sensitivity": mcnemar(original_rows, selective_rows),
        "base_deltas": base_deltas,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--randomizations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=842019)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    report = {
        "seed": args.seed,
        "bootstraps": args.bootstraps,
        "randomizations": args.randomizations,
        "policy_mapping": POLICY_SOURCE,
        "models": {},
    }

    for key, (model, revision) in MODELS.items():
        rows = load_primary(model, revision)
        groups = defaultdict(dict)
        for row in rows:
            groups[row["base_id"]][row["form"]] = row
        if len(groups) != 200 or any(set(forms) != set(FORMS) for forms in groups.values()):
            raise RuntimeError(f"{key}: invalid base grouping")
        model_report = {
            "model": model,
            "revision": revision,
            "n_bases": len(groups),
            "form_accuracy": {},
            "domains": {},
        }
        for form in FORMS:
            form_rows = [forms[form] for forms in groups.values()]
            successes = sum(bool(row["correct"]) for row in form_rows)
            model_report["form_accuracy"][form] = {
                "correct": successes,
                "total": len(form_rows),
                "accuracy": successes / len(form_rows),
                "wilson95": wilson(successes, len(form_rows)),
                "parse_failures": sum(row["parse_status"] != "ok" for row in form_rows),
            }

        source_tests = []
        for domain in ("finqa", "tatqa", "pooled"):
            selected = [
                forms for forms in groups.values()
                if domain == "pooled" or next(iter(forms.values()))["domain"] == domain
            ]
            domain_report = {
                "n_bases": len(selected),
                "policies": {
                    policy: summarize_policy(selected, policy) for policy in POLICY_SOURCE
                },
                "selective_vs_original": comparison(
                    selected, rng, args.bootstraps, args.randomizations
                ),
                "by_input_form": {},
            }
            for form in NONCANONICAL:
                old = [bool(forms[form]["correct"]) for forms in selected]
                source_form = POLICY_SOURCE["selective"][form]
                new = [bool(forms[source_form]["correct"]) for forms in selected]
                deltas = [int(after) - int(before) for before, after in zip(old, new)]
                domain_report["by_input_form"][form] = {
                    "delta_accuracy": float(np.mean(deltas)),
                    "cluster_bootstrap95": clustered_bootstrap(deltas, rng, args.bootstraps),
                    "mcnemar": mcnemar(old, new),
                }
            if domain != "pooled":
                source_tests.append({
                    "domain": domain,
                    "p_raw": domain_report["selective_vs_original"]["base_sign_flip_p_raw"],
                })
            model_report["domains"][domain] = domain_report
        holm(source_tests)
        model_report["source_sign_flip_holm"] = source_tests
        pooled = model_report["domains"]["pooled"]["selective_vs_original"]
        source_deltas = [
            model_report["domains"][domain]["selective_vs_original"]["delta_accuracy"]
            for domain in ("finqa", "tatqa")
        ]
        model_report["primary_confirmation_passed"] = (
            pooled["cluster_bootstrap95"][0] > 0 and min(source_deltas) >= 0
        )
        for domain_report in model_report["domains"].values():
            domain_report["selective_vs_original"].pop("base_deltas")
        report["models"][key] = model_report

    output = ROOT / "results/downstream_mitigation_analysis.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
