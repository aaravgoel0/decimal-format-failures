#!/usr/bin/env python3
"""Build fresh cases for fixed-layer many-random-site causal confirmation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATES = (
    "Compare these decimal values, listed first and second.\n{a}\n{b}\n1. The first is larger\n2. The second is larger\n3. They are equal\nAnswer with only the option number.",
    "Treat the following two strings as real numbers.\n{a}\n{b}\nWhich relation is true?\n1. The first is greater\n2. The second is greater\n3. The values are equal\nReturn only 1, 2, or 3.",
    "Select the correct numerical comparison for the two decimals below.\n{a}\n{b}\n1. The first is greater\n2. The second is greater\n3. Both have the same value\nReply with one option number.",
)


def main():
    rows = []
    for case, whole in enumerate(range(701, 851)):
        digit = case % 10
        zeros = case % 5 + 1
        template_index = case % len(TEMPLATES)
        short = f"{whole}.{digit}"
        padded = short + "0" * zeros
        template = TEMPLATES[template_index]
        rows.append({
            "id": f"random-site-{case:03d}",
            "case": case,
            "template_index": template_index,
            "whole": whole,
            "digit": digit,
            "zeros": zeros,
            "short": short,
            "padded": padded,
            "easy_prompt": template.format(a=short, b=padded),
            "hard_prompt": template.format(a=padded, b=short),
            "answer": 3,
        })
    assert len(rows) == 150
    path = ROOT / "data/causal_random_site_confirmation.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(path, len(rows))


if __name__ == "__main__":
    main()
