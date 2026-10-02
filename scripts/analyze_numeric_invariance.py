#!/usr/bin/env python3
"""Analyze the frozen downstream numerical-invariance benchmark."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
FORMS = ("canonical", "padded", "leading_zero", "scientific")
MODELS = {
    "llama": ("meta-llama/Llama-3.1-8B-Instruct",
              "0e9e39f249a16976918f6564b8830bc894c89659"),
    "qwen": ("Qwen/Qwen3-4B-Instruct-2507",
             "cdbee75f17c01a7cc42f958dc650907174af0554"),
    "gemma": ("google/gemma-2-9b-it",
              "11c9b309abf73637e4b6f9a3fa1e92e615547819"),
}


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def paired_bootstrap(delta, rng, draws=10000):
    values = np.asarray(delta, dtype=float)
    if len(values) == 0:
        return [None, None]
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    estimates = values[samples].mean(axis=1)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def mcnemar(a, b):
    improve = sum((not x) and y for x, y in zip(a, b))
    regress = sum(x and (not y) for x, y in zip(a, b))
    discordant = improve + regress
    p = 1.0 if discordant == 0 else float(binomtest(min(improve, regress), discordant, 0.5).pvalue)
    return {"improve": improve, "regress": regress, "discordant": discordant, "p_raw": p}


def holm(records):
    ranked = sorted(enumerate(records), key=lambda pair: pair[1]["mcnemar"]["p_raw"])
    running = 0.0
    for rank, (index, record) in enumerate(ranked):
        adjusted = min(1.0, record["mcnemar"]["p_raw"] * (len(records) - rank))
        running = max(running, adjusted)
        records[index]["mcnemar"]["p_holm"] = running


def normalized_prediction(row):
    if row.get("parse_status") != "ok":
        return None
    percent = bool(row.get("prediction_is_percent")) if row["answer_is_percent"] else False
    return row.get("prediction_value"), percent


def tuple_correct(prediction, reference):
    if prediction is None:
        return False
    value, is_percent = prediction
    return value == reference["answer_value"] and (not reference["answer_is_percent"] or is_percent)


def load_model(key, model, revision):
    safe = "".join(character if character.isalnum() or character in "_.-" else "-" for character in model)
    path = ROOT / "results" / f"numeric_invariance_{safe}.jsonl"
    attempts = [json.loads(line) for line in path.read_text().splitlines()]
    if {row["model"] for row in attempts} != {model} or {row["model_revision"] for row in attempts} != {revision}:
        raise RuntimeError(f"provenance mismatch: {key}")
    if {row["checkpoint_precision"] for row in attempts} != {"native-unquantized"}:
        raise RuntimeError(f"non-native primary checkpoint: {key}")
    rows = {}
    for row in attempts:
        previous = rows.get(row["id"])
        if previous is None or (previous.get("error") is not None and row.get("error") is None):
            rows[row["id"]] = row
    if len(rows) != 852:
        raise RuntimeError(f"{key}: expected 852 unique ids, got {len(rows)}")
    return list(rows.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=731947)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    report = {"seed": args.seed, "bootstraps": args.bootstraps, "models": {}}
    tests = []

    for key, (model, revision) in MODELS.items():
        rows = load_model(key, model, revision)
        groups = defaultdict(dict)
        for row in rows:
            groups[row["base_id"]][row["form"]] = row
        usable = {base: forms for base, forms in groups.items()
                  if set(forms) == set(FORMS) and not any(forms[form].get("error") for form in FORMS)}
        if len(groups) != 213:
            raise RuntimeError(f"{key}: expected 213 bases, got {len(groups)}")

        model_report = {
            "model": model,
            "revision": revision,
            "usable_bases": len(usable),
            "excluded_bases": len(groups) - len(usable),
            "excluded_by_domain": {
                domain: sum(next(iter(forms.values()))["domain"] == domain for base, forms in groups.items()
                            if base not in usable)
                for domain in ("gsm8k", "finqa", "tatqa")
            },
            "domains": {},
        }
        for domain in ("gsm8k", "finqa", "tatqa", "pooled"):
            selected = [forms for forms in usable.values()
                        if domain == "pooled" or next(iter(forms.values()))["domain"] == domain]
            domain_report = {"n": len(selected), "forms": {}, "paired_vs_canonical": {}}
            for form in FORMS:
                correct = [bool(forms[form]["correct"]) for forms in selected]
                successes = sum(correct)
                domain_report["forms"][form] = {
                    "correct": successes,
                    "accuracy": successes / len(correct),
                    "wilson95": wilson(successes, len(correct)),
                    "parse_failures": sum(forms[form]["parse_status"] != "ok" for forms in selected),
                }
            canonical = [bool(forms["canonical"]["correct"]) for forms in selected]
            for form in FORMS[1:]:
                comparison = [bool(forms[form]["correct"]) for forms in selected]
                record = {
                    "model": key,
                    "domain": domain,
                    "form": form,
                    "delta_accuracy": float(np.mean(comparison) - np.mean(canonical)),
                    "bootstrap95": paired_bootstrap(
                        [int(y) - int(x) for x, y in zip(canonical, comparison)], rng, args.bootstraps),
                    "mcnemar": mcnemar(canonical, comparison),
                }
                domain_report["paired_vs_canonical"][form] = record
                if domain != "pooled":
                    tests.append(record)

            predictions = [[normalized_prediction(forms[form]) for form in FORMS]
                           for forms in selected]
            disagreement_count = sum(len(set(items)) > 1 for items in predictions)
            domain_report["any_prediction_disagreement"] = disagreement_count / len(predictions)
            domain_report["disagreement_wilson95"] = wilson(disagreement_count, len(predictions))
            domain_report["robust_accuracy"] = sum(
                all(forms[form]["correct"] for form in FORMS) for forms in selected) / len(selected)
            domain_report["mean_unique_predictions"] = float(np.mean([len(set(items)) for items in predictions]))

            plurality_correct = []
            for forms, items in zip(selected, predictions):
                votes = Counter(item for item in items if item is not None)
                if not votes:
                    choice = normalized_prediction(forms["canonical"])
                else:
                    top = max(votes.values())
                    leaders = [item for item, count in votes.items() if count == top]
                    canonical_prediction = normalized_prediction(forms["canonical"])
                    choice = canonical_prediction if canonical_prediction in leaders or len(leaders) > 1 else leaders[0]
                plurality_correct.append(tuple_correct(choice, forms["canonical"]))
            plurality_successes = sum(plurality_correct)
            domain_report["plurality_mitigation"] = {
                "correct": plurality_successes,
                "accuracy": plurality_successes / len(selected),
                "wilson95": wilson(plurality_successes, len(selected)),
                "delta_vs_canonical": float(np.mean(plurality_correct) - np.mean(canonical)),
                "bootstrap95": paired_bootstrap(
                    [int(y) - int(x) for x, y in zip(canonical, plurality_correct)], rng, args.bootstraps),
                "mcnemar": mcnemar(canonical, plurality_correct),
            }
            model_report["domains"][domain] = domain_report
        report["models"][key] = model_report

    if len(tests) != 27:
        raise RuntimeError(f"expected 27 primary tests, got {len(tests)}")
    holm(tests)
    report["primary_holm_family"] = [
        {"model": record["model"], "domain": record["domain"], "form": record["form"],
         **record["mcnemar"]} for record in tests
    ]
    output = ROOT / "results/numeric_invariance_analysis.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
