#!/usr/bin/env python3
"""Generate deterministic, balanced decimal-comparison datasets."""

import argparse
import json
import random
from decimal import Decimal
from pathlib import Path


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def ordered_pair(a, b, index):
    if index % 2:
        a, b = b, a
    da, db = Decimal(a), Decimal(b)
    answer = 1 if da > db else 2 if db > da else 3
    return a, b, answer


def integers(n, rng):
    rows = []
    for i in range(n):
        a = rng.randint(0, 999)
        b = rng.randint(0, 998)
        if b >= a:
            b += 1
        x, y, answer = ordered_pair(str(a), str(b), i)
        rows.append({"id": f"integer-{i:04d}", "task": "integer", "a": x,
                     "b": y, "answer": answer})
    return rows


def misleading(n, rng):
    """Longer fractional digit string always denotes the smaller value."""
    rows = []
    for i in range(n):
        whole = rng.randint(0, 20)
        high_digit = rng.randint(6, 9)
        low_suffix = rng.randint(11, 13)
        short = f"{whole}.{high_digit}"
        long = f"{whole}.{low_suffix}"
        x, y, answer = ordered_pair(short, long, i)
        rows.append({"id": f"misleading-{i:04d}", "task": "misleading",
                     "a": x, "b": y, "answer": answer, "whole": whole,
                     "high_digit": high_digit, "low_suffix": low_suffix})
    return rows


def zero_padding(n, rng):
    rows = []
    for i in range(n):
        whole = rng.randint(0, 20)
        digit = rng.randint(0, 9)
        zeros = 1 + (i % 3)
        canonical = f"{whole}.{digit}"
        padded = canonical + ("0" * zeros)
        x, y, answer = ordered_pair(canonical, padded, i)
        rows.append({"id": f"zero-padding-{i:04d}", "task": "zero_padding",
                     "a": x, "b": y, "answer": answer, "whole": whole,
                     "digit": digit, "zeros": zeros,
                     "padded_position": 1 if x == padded else 2})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7302026)
    p.add_argument("--output-dir", type=Path, default=Path("data"))
    args = p.parse_args()
    rng = random.Random(args.seed)
    write_jsonl(args.output_dir / "integers.jsonl", integers(args.n, rng))
    write_jsonl(args.output_dir / "misleading.jsonl", misleading(args.n, rng))
    write_jsonl(args.output_dir / "zero_padding.jsonl", zero_padding(args.n, rng))


if __name__ == "__main__":
    main()
