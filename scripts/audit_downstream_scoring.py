#!/usr/bin/env python3
"""Audit downstream answer extraction and exact-value scoring."""

import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "four_format_llama": ROOT / "results/numeric_invariance_meta-llama-Llama-3.1-8B-Instruct.jsonl",
    "four_format_qwen": ROOT / "results/numeric_invariance_Qwen-Qwen3-4B-Instruct-2507.jsonl",
    "four_format_gemma": ROOT / "results/numeric_invariance_google-gemma-2-9b-it.jsonl",
    "mitigation_llama": ROOT / "results/downstream_mitigation_meta-llama-Llama-3.1-8B-Instruct.jsonl",
    "mitigation_qwen": ROOT / "results/downstream_mitigation_Qwen-Qwen3-4B-Instruct-2507.jsonl",
}
SCIENTIFIC = re.compile(r"[-+]?\d+(?:\.\d+)?[eE][-+]?\d+")


def final_rows(path):
    chosen = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        previous = chosen.get(row["id"])
        if previous is None or (previous.get("error") is not None and row.get("error") is None):
            chosen[row["id"]] = row
    return [chosen[key] for key in sorted(chosen)]


def decimal_value(value):
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def classify(row):
    parsed = decimal_value(row.get("prediction_value")) if row.get("parse_status") == "ok" else None
    gold = decimal_value(row.get("answer_value"))
    numeric_equal = parsed is not None and gold is not None and parsed == gold
    unit_equal = bool(row.get("prediction_is_percent")) == bool(row.get("answer_is_percent"))
    rescored = numeric_equal and unit_equal
    if row.get("parse_status") != "ok":
        category = "parse_failure"
    elif numeric_equal and not unit_equal:
        category = "unit_mismatch"
    elif rescored and not bool(row.get("correct")):
        category = "format_equivalent_rescued"
    elif rescored:
        category = "exact_match"
    else:
        category = "numeric_mismatch"
    return rescored, category


def compact(row, category):
    return {
        "id": row["id"],
        "model": row.get("model"),
        "gold": row.get("answer_value"),
        "gold_percent": bool(row.get("answer_is_percent")),
        "prediction": row.get("prediction_value"),
        "prediction_percent": bool(row.get("prediction_is_percent")),
        "parse_status": row.get("parse_status"),
        "category": category,
        "raw_response": row.get("raw_response", "")[:240],
    }


def main():
    report = {
        "scoring_rule": {
            "extraction": "last numeric token after removing commas and currency symbols",
            "numeric_comparison": "exact Decimal equality; textual precision is ignored",
            "units": "the percent marker must match the reference",
            "rounding_tolerance": "none",
            "scientific_notation": "accepted and compared by exact decimal value",
        },
        "files": {},
        "spot_audit_candidates": {},
    }
    candidates = {name: [] for name in (
        "format_equivalent_rescued",
        "unit_mismatch",
        "scientific_response",
        "currency_or_comma_response",
        "close_but_not_equal",
        "parse_failure",
    )}
    for name, path in FILES.items():
        rows = final_rows(path)
        counts = Counter()
        for row in rows:
            rescored, category = classify(row)
            counts[category] += 1
            counts["stored_correct"] += int(bool(row.get("correct")))
            counts["rescored_correct"] += int(rescored)
            record = compact(row, category)
            if category in candidates:
                candidates[category].append(record)
            raw = row.get("raw_response", "")
            if SCIENTIFIC.search(raw):
                candidates["scientific_response"].append(record)
            if "$" in raw or "," in raw:
                candidates["currency_or_comma_response"].append(record)
            predicted = decimal_value(row.get("prediction_value"))
            gold = decimal_value(row.get("answer_value"))
            if predicted is not None and gold is not None and predicted != gold and abs(predicted - gold) <= Decimal("0.01"):
                candidates["close_but_not_equal"].append(record)
        report["files"][name] = {
            "path": str(path.relative_to(ROOT)),
            "rows": len(rows),
            **dict(sorted(counts.items())),
        }
    for category, rows in candidates.items():
        unique = {row["id"] + "|" + row["model"]: row for row in rows}
        report["spot_audit_candidates"][category] = [unique[key] for key in sorted(unique)[:6]]
    output = ROOT / "results/downstream_scoring_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
