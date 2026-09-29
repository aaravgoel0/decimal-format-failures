#!/usr/bin/env python3
"""Build fresh Qwen cases for attention, MLP, and head-level interventions."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = (
    "Decide which statement is mathematically correct for these values:\n{a}\n{b}\n1. The first value is greater\n2. The second value is greater\n3. The values are equal\nOutput only 1, 2, or 3.",
    "Compare the numerical quantities written below.\nFirst: {a}\nSecond: {b}\n1. First is larger\n2. Second is larger\n3. They are equal\nGive only the option number.",
    "Which relation holds between the following decimal numbers?\n{a}\n{b}\n1. The upper number is larger\n2. The lower number is larger\n3. Both numbers have equal value\nAnswer with one digit.",
)


def main():
    rows = []
    for case, whole in enumerate(range(1001, 1121)):
        digit = case % 10
        zeros = case % 5 + 1
        template_index = case % 3
        short = f"{whole}.{digit}"
        padded = short + "0" * zeros
        template = TEMPLATES[template_index]
        split = "discovery" if case < 30 else "heldout"
        rows.append({
            "id": f"qwen-component-{case:03d}",
            "case": case,
            "split": split,
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
    assert len(rows) == 120
    assert sum(row["split"] == "discovery" for row in rows) == 30
    assert all(sum(row["template_index"] == template and row["split"] == split for row in rows)
               == (10 if split == "discovery" else 30)
               for template in range(3) for split in ("discovery", "heldout"))
    path = ROOT / "data/qwen_component_causal.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(path, len(rows))


if __name__ == "__main__":
    main()
