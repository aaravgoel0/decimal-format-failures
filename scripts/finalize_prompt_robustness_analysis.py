#!/usr/bin/env python3
"""Make separated auxiliary fits explicit while preserving frozen primary results."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "results/prompt_robustness_confirmation_analysis.json"


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main():
    payload = json.loads(PATH.read_text())
    corrected = 0
    for model in payload:
        fit = model["clustered_logistic_padded_first"]
        if finite_tree(fit):
            fit["status"] = "estimated"
            continue
        model["clustered_logistic_padded_first"] = {
            "status": "not_estimable_complete_separation",
            "reason": "padded-second accuracy was 1.0, producing complete separation",
            "n_rows": fit["n_rows"],
            "n_base_clusters": fit["n_base_clusters"],
        }
        corrected += 1
    PATH.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print({"corrected_nonfinite_auxiliary_fits": corrected, "models": len(payload)})


if __name__ == "__main__":
    main()
