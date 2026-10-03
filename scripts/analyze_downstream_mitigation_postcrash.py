#!/usr/bin/env python3
"""Apply the frozen analysis to complete runs and label partial Gemma data."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import analyze_downstream_mitigation as frozen


ROOT = Path(__file__).resolve().parents[1]


def main():
    complete_models = {
        key: value for key, value in frozen.MODELS.items() if key in {"llama", "qwen"}
    }
    frozen.MODELS = complete_models
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], "--bootstraps", "10000", "--randomizations", "100000", "--seed", "842019"]
        frozen.main()
    finally:
        sys.argv = original_argv

    frozen_output = ROOT / "results/downstream_mitigation_analysis.json"
    report = json.loads(frozen_output.read_text())
    frozen_output.unlink()
    report["operational_deviation"] = {
        "path": "DOWNSTREAM_MITIGATION_CRASH_DEVIATION.md",
        "complete_confirmatory_models": ["llama", "qwen"],
        "incomplete_model": "gemma",
        "local_large_model_abandoned": True,
        "hosted_72b_study_rows_sent": 0,
    }

    model, revision = frozen.MODELS.get("gemma", (
        "google/gemma-2-9b-it",
        "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    ))
    path = ROOT / "results/downstream_mitigation_google-gemma-2-9b-it.jsonl"
    attempts = [json.loads(line) for line in path.read_text().splitlines()]
    if len(attempts) != 752:
        raise RuntimeError("unexpected partial Gemma attempt count")
    if {row["model"] for row in attempts} != {model} or {row["model_revision"] for row in attempts} != {revision}:
        raise RuntimeError("partial Gemma provenance mismatch")
    error_attempts = sum(row.get("error") is not None for row in attempts)
    final = {}
    repeated = defaultdict(list)
    for row in attempts:
        repeated[row["id"]].append(row)
        previous = final.get(row["id"])
        if previous is None or (previous.get("error") is not None and row.get("error") is None):
            final[row["id"]] = row
    rows = list(final.values())
    if len(rows) != 637 or error_attempts != 169:
        raise RuntimeError("unexpected partial Gemma unique-id count")
    duplicate_groups = [group for group in repeated.values() if len(group) > 1]
    raw_conflicts = sum(len({row["raw_response"] for row in group}) > 1 for group in duplicate_groups)
    prediction_conflicts = sum(
        len({(row["prediction_value"], row["prediction_is_percent"]) for row in group}) > 1
        for group in duplicate_groups
    )

    dataset = [
        json.loads(line)
        for line in (ROOT / "data/downstream_mitigation_confirmation.jsonl").read_text().splitlines()
    ]
    all_base_ids = {row["base_id"] for row in dataset}
    groups = defaultdict(dict)
    for row in rows:
        if row.get("error") is None:
            groups[row["base_id"]][row["form"]] = row
    complete = {
        base_id: forms for base_id, forms in groups.items()
        if set(forms) == set(frozen.FORMS)
    }
    missing = sorted(all_base_ids - set(complete))
    if len(complete) != 142 or len(missing) != 58:
        raise RuntimeError("unexpected partial Gemma base structure")

    partial = {
        "model": model,
        "revision": revision,
        "outcomes_used": False,
        "primary_confirmation_status": "not_evaluable_memory_failure",
        "preserved_attempt_lines": len(attempts),
        "error_attempts": error_attempts,
        "unique_prompt_ids": len(rows),
        "successful_unique_rows": sum(row.get("error") is None for row in rows),
        "error_only_prompt_ids": sum(row.get("error") is not None for row in rows),
        "expected_rows": len(dataset),
        "complete_successful_bases": len(complete),
        "complete_successful_bases_by_domain": {
            domain: sum(
                next(iter(forms.values()))["domain"] == domain
                for forms in complete.values()
            )
            for domain in ("finqa", "tatqa")
        },
        "partial_successful_bases": sum(0 < len(forms) < 4 for forms in groups.values()),
        "duplicate_prompt_ids": len(duplicate_groups),
        "duplicate_raw_response_conflicts": raw_conflicts,
        "duplicate_prediction_conflicts": prediction_conflicts,
        "missing_bases": missing,
        "missing_domains": {
            domain: sum(
                next(row for row in dataset if row["base_id"] == base_id)["domain"] == domain
                for base_id in missing
            )
            for domain in ("finqa", "tatqa")
        },
        "note": "Accuracy outcomes are not analyzed because failures and truncation are nonrandom.",
    }
    report["incomplete_gemma_execution"] = partial

    output = ROOT / "results/downstream_mitigation_analysis_postcrash.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
