#!/usr/bin/env python3
"""Build the frozen 10-template paired prompt-robustness dataset."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATES = (
    "Decimal A: {a}\nDecimal B: {b}\nWhich relation is correct?\n{options}\nReply with only the option number.",
    "Compare {a} with {b} as numerical values.\n{options}\nReturn exactly one option number.",
    "Let x = {a} and y = {b}. Select the true statement.\n{options}\nAnswer using only its number.",
    "Choose the mathematically correct comparison between {a} and {b}.\n{options}\nOutput one number only.",
    "Treat both strings as decimal numbers: {a}, {b}.\n{options}\nGive only the numbered choice.",
    "For the real-number values {a} and {b}, which statement holds?\n{options}\nRespond with just the choice number.",
    "Evaluate the ordering of A={a} and B={b}.\n{options}\nWrite only the option number.",
    "Pick the valid relation for these decimals. First: {a}. Second: {b}.\n{options}\nYour response must be one option number.",
    "Numerically compare the first decimal ({a}) and second decimal ({b}).\n{options}\nAnswer with the number of the correct relation only.",
    "Which comparison is true for {a} and {b}?\n{options}\nSelect exactly one numbered answer.",
)


def options(equal_label, rng):
    other = [label for label in (1, 2, 3) if label != equal_label]
    rng.shuffle(other)
    relations = {equal_label: "The two values are equal.",
                 other[0]: "The first value is greater.",
                 other[1]: "The second value is greater."}
    return "\n".join(f"{label}. {relations[label]}" for label in (1, 2, 3))


def main():
    rng = random.Random(20260913)
    rows = []
    whole = 1001
    for template_index, template in enumerate(TEMPLATES):
        for equal_label in (1, 2, 3):
            for digit in range(10):
                for zeros in range(1, 6):
                    canonical = f"{whole}.{digit}"
                    padded = canonical + "0" * zeros
                    base_id = f"prompt-confirm-{template_index}-{equal_label}-{digit}-{zeros}"
                    rendered_options = options(equal_label, random.Random(20260913 + whole))
                    for padded_position in (1, 2):
                        a, b = ((padded, canonical) if padded_position == 1 else
                                (canonical, padded))
                        rows.append({
                            "id": f"{base_id}-p{padded_position}",
                            "base_id": base_id,
                            "template_index": template_index,
                            "equal_label": equal_label,
                            "answer": equal_label,
                            "whole": whole,
                            "digit": digit,
                            "zeros": zeros,
                            "canonical": canonical,
                            "padded": padded,
                            "padded_position": padded_position,
                            "user_prompt": template.format(a=a, b=b, options=rendered_options),
                        })
                    whole += 1
    assert len(rows) == 3000 and whole == 2501
    rng.shuffle(rows)
    path = ROOT / "data/prompt_robustness_confirmation.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(path, len(rows))


if __name__ == "__main__":
    main()

