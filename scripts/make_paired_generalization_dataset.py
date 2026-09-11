#!/usr/bin/env python3
"""Build the frozen paired order and label-position dataset."""
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELATIONS = ("first_larger", "second_larger", "equal")
STATEMENTS = {
    "first_larger": "The first value is larger",
    "second_larger": "The second value is larger",
    "equal": "The two values are equal",
}
TEMPLATES = (
    "Compare {a} with {b}.\nWhich statement is true?\n{options}\nReturn only 1, 2, or 3.",
    "Decide the mathematical relation between the first number, {a}, and the second number, {b}.\n{options}\nAnswer with exactly one option number.",
)


def option_order(equal_position):
    others = [relation for relation in RELATIONS if relation != "equal"]
    order = others[:]
    order.insert(equal_position - 1, "equal")
    return order


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows = []
    base = 0
    for template_index in range(2):
        for equal_position in (1, 2, 3):
            labels = option_order(equal_position)
            options = "\n".join(f"{i}. {STATEMENTS[relation]}" for i, relation in enumerate(labels, 1))
            for digit in range(10):
                for zeros in range(1, 6):
                    whole = 201 + base
                    short = f"{whole}.{digit}"
                    padded = short + "0" * zeros
                    base_id = f"paired-{base:03d}"
                    for padded_position in (1, 2):
                        a, b = (padded, short) if padded_position == 1 else (short, padded)
                        rows.append({
                            "id": f"{base_id}-o{padded_position}",
                            "base_id": base_id,
                            "task": "paired_order_generalization",
                            "a": a,
                            "b": b,
                            "answer": equal_position,
                            "whole": whole,
                            "digit": digit,
                            "zeros": zeros,
                            "padded_position": padded_position,
                            "template_index": template_index,
                            "option_order": labels,
                            "user_prompt": TEMPLATES[template_index].format(a=a, b=b, options=options),
                        })
                    base += 1
    assert base == 300 and len(rows) == 600
    rng = random.Random(91_120_26)
    rng.shuffle(rows)
    out = ROOT / "data" / "paired_order_generalization.jsonl"
    out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    protocol = ROOT / "PAIRED_GENERALIZATION_PROTOCOL.md"
    manifest = {
        "frozen_at": "2026-09-11 America/Los_Angeles",
        "dataset_rows": len(rows),
        "base_item_pairs": base,
        "dataset_sha256": sha256(out),
        "protocol_sha256": sha256(protocol),
        "generator_sha256": sha256(Path(__file__)),
        "evaluator_sha256": sha256(ROOT / "scripts/evaluate_paired_generalization.py"),
        "analysis_sha256": sha256(ROOT / "scripts/analyze_paired_generalization.py"),
    }
    (ROOT / "PAIRED_GENERALIZATION_FREEZE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out, len(rows), manifest["dataset_sha256"])


if __name__ == "__main__":
    main()
