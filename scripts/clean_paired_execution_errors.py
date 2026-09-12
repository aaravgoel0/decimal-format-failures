#!/usr/bin/env python3
"""Preserve and remove execution failures before an exact resume run."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "paired_order_generalization_google-gemma-2-9b-it.jsonl"
ERROR_LOG = ROOT / "results" / "execution_errors_paired_order_generalization_gemma.jsonl"


def main():
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    failed = [row for row in rows if row.get("parse_status") == "error"]
    valid = [row for row in rows if row.get("parse_status") != "error"]
    if len(rows) != 600 or len(failed) != 4 or len(valid) != 596:
        raise RuntimeError(f"unexpected counts: total={len(rows)} failed={len(failed)}")
    ERROR_LOG.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in failed))
    SOURCE.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in valid))
    print(f"preserved {len(failed)} failures; retained {len(valid)} valid rows")


if __name__ == "__main__":
    main()
