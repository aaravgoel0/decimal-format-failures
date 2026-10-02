#!/usr/bin/env python3
"""Analyze the fresh selective-normalization confirmation study."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("negative", "leading_zero", "long_fraction", "scientific", "signed_zero")
CONDITIONS = ("original", "full_canonical", "selective")
MODELS = {
    "llama": ("meta-llama/Llama-3.1-8B-Instruct",
              "0e9e39f249a16976918f6564b8830bc894c89659"),
    "qwen": ("Qwen/Qwen3-4B-Instruct-2507",
             "cdbee75f17c01a7cc42f958dc650907174af0554"),
    "gemma": ("google/gemma-2-9b-it",
              "11c9b309abf73637e4b6f9a3fa1e92e615547819"),
}


def wilson(successes, total, z=1.959963984540054):
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def paired_bootstrap(delta, rng, draws):
    values = np.asarray(delta, dtype=float)
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    return [float(x) for x in np.quantile(values[samples].mean(axis=1), [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=731948)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    report = {"seed": args.seed, "bootstraps": args.bootstraps, "models": {}}

    for key, (model, revision) in MODELS.items():
        safe = "".join(character if character.isalnum() or character in "_.-" else "-" for character in model)
        path = ROOT / "results" / f"selective_normalization_{safe}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if {row["model"] for row in rows} != {model} or {row["model_revision"] for row in rows} != {revision}:
            raise RuntimeError(f"provenance mismatch: {key}")
        if {row["checkpoint_precision"] for row in rows} != {"native-unquantized"}:
            raise RuntimeError(f"non-native checkpoint: {key}")
        grouped = defaultdict(dict)
        for row in rows:
            grouped[row["id"]][row["condition"]] = row
        usable = {item: conditions for item, conditions in grouped.items()
                  if set(conditions) == set(CONDITIONS) and
                  not any(conditions[condition].get("error") for condition in CONDITIONS)}
        if len(usable) != 600:
            raise RuntimeError(f"{key}: expected 600 complete items, got {len(usable)}")

        model_report = {"model": model, "revision": revision, "families": {}}
        pooled = []
        for family in (*FAMILIES, "pooled"):
            selected = [conditions for conditions in usable.values()
                        if family == "pooled" or conditions["original"]["family"] == family]
            result = {"n": len(selected), "conditions": {}, "paired_changes": {}}
            for condition in CONDITIONS:
                correct = [bool(conditions[condition]["correct"]) for conditions in selected]
                successes = sum(correct)
                result["conditions"][condition] = {
                    "correct": successes,
                    "accuracy": successes / len(correct),
                    "wilson95": wilson(successes, len(correct)),
                    "parse_failures": sum(
                        conditions[condition]["parse_status"] != "ok" for conditions in selected),
                }
            original = [bool(conditions["original"]["correct"]) for conditions in selected]
            for condition in ("full_canonical", "selective"):
                changed = [bool(conditions[condition]["correct"]) for conditions in selected]
                delta = [int(y) - int(x) for x, y in zip(original, changed)]
                result["paired_changes"][condition] = {
                    "delta_accuracy": float(np.mean(delta)),
                    "bootstrap95": paired_bootstrap(delta, rng, args.bootstraps),
                    "improve": sum(value == 1 for value in delta),
                    "regress": sum(value == -1 for value in delta),
                }
            if family != "pooled" and family not in {"leading_zero", "scientific"}:
                mismatches = sum(
                    conditions["original"]["raw_response"] != conditions["selective"]["raw_response"]
                    for conditions in selected)
                if mismatches:
                    raise RuntimeError(f"determinism failure in unchanged {family}: {mismatches}")
            if family == "pooled":
                pooled = result
            model_report["families"][family] = result

        changed_effects = [model_report["families"][family]["paired_changes"]["selective"]["delta_accuracy"]
                           for family in ("leading_zero", "scientific")]
        pooled_interval = pooled["paired_changes"]["selective"]["bootstrap95"]
        model_report["selective_pass"] = pooled_interval[0] > 0 and all(effect >= 0 for effect in changed_effects)
        report["models"][key] = model_report

    output = ROOT / "results/selective_normalization_analysis.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
