#!/usr/bin/env python3
"""Enforce the frozen exact-response correctness rule on saved paired rows."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for path in sorted((ROOT / "results").glob("paired_order_generalization_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        changed = 0
        for row in rows:
            expected = row.get("parse_status") == "exact" and row.get("prediction") == row.get("answer")
            if row.get("correct") != expected:
                row["correct"] = expected
                changed += 1
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        print(path, len(rows), changed)


if __name__ == "__main__":
    main()
