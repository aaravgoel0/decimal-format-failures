#!/usr/bin/env python3
"""Create the fixed, balanced held-out factorial dataset."""

import json
import random
from pathlib import Path


def main():
    rng = random.Random(8192026)
    rows = []
    row_id = 0
    for digit in range(10):
        for zeros in range(1, 6):
            for padded_position in (1, 2):
                wholes = list(range(21, 100))
                rng.shuffle(wholes)
            for cell_index, whole in enumerate(wholes[:20]):
                    canonical = f"{whole}.{digit}"
                    padded = canonical + "0" * zeros
                    a, b = ((padded, canonical) if padded_position == 1
                            else (canonical, padded))
                    rows.append({
                        "id": f"confirmatory-zero-padding-{row_id:04d}",
                        "task": "confirmatory_zero_padding", "a": a, "b": b,
                        "answer": 3, "whole": whole, "digit": digit,
                        "zeros": zeros, "padded_position": padded_position,
                        "cell_replicate": cell_index,
                    })
                    row_id += 1
    rng.shuffle(rows)
    output = Path("data/confirmatory_zero_padding.jsonl")
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(output, len(rows))


if __name__ == "__main__":
    main()
