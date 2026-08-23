#!/usr/bin/env python3
"""Generate model-blind, disjoint causal-generalization cases."""
import json
from pathlib import Path


TEMPLATES = {
    "relation_statements": (
        "Compare these numerical values:\nValue A: {a}\nValue B: {b}\n"
        "Which statement is mathematically true?\n1. A is larger\n"
        "2. B is larger\n3. They are equal\nReply with only 1, 2, or 3."
    ),
    "direct_choice": (
        "Numerical value A is {a}. Numerical value B is {b}.\n"
        "Choose the correct relation:\n1. The first value is larger\n"
        "2. The second value is larger\n3. The values are equal\n"
        "Return only the choice number."
    ),
}


def main():
    rows = []
    case = 0
    for template_index, (template_name, template) in enumerate(TEMPLATES.items()):
        for local in range(50):
            whole = 101 + template_index * 50 + local
            digit = local % 10
            zeros = 1 + (local // 10)
            short = f"{whole}.{digit}"
            padded = short + "0" * zeros
            # Same written-length pattern, but mathematically unequal. The donor
            # second digit is deterministically different and never wraps to equal.
            donor_digit = (digit + 3) % 10
            donor_short = f"{whole}.{donor_digit}"
            donor_padded = short + "0" * zeros
            rows.append({
                "id": f"causal-gen-{case:03d}", "case": case,
                "template": template_name, "whole": whole, "digit": digit,
                "zeros": zeros, "short": short, "padded": padded,
                "easy_prompt": template.format(a=short, b=padded),
                "hard_prompt": template.format(a=padded, b=short),
                "donor_prompt": template.format(a=donor_padded, b=donor_short),
                "donor_digit": donor_digit, "answer": 3,
            })
            case += 1
    assert len(rows) == 100
    assert len({row["whole"] for row in rows}) == 100
    out = Path("data/causal_generalization.jsonl")
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(out, len(rows))


if __name__ == "__main__":
    main()
