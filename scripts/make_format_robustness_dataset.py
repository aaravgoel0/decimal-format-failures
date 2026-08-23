#!/usr/bin/env python3
"""Generate balanced broad-format comparisons and paired canonicalized prompts."""
import json
from decimal import Decimal
from pathlib import Path

RELATIONS = {"gt": "The first value is larger", "lt": "The second value is larger", "eq": "The values are equal"}
TEMPLATES = [
    "Compare A = {a} with B = {b}.\n{options}\nReply with only the correct option number.",
    "Which relation between these numerical values is true?\nFirst: {a}\nSecond: {b}\n{options}\nReturn only 1, 2, or 3.",
]


def canonical(text):
    value = Decimal(text)
    if value == 0:
        return "0"
    result = format(value, "f")
    sign = "-" if result.startswith("-") else ""
    unsigned = result.lstrip("-")
    whole, dot, fraction = unsigned.partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return sign + whole + (("." + fraction) if dot and fraction else "")


def relation(a, b):
    x, y = Decimal(a), Decimal(b)
    return "gt" if x > y else "lt" if x < y else "eq"


def render(template, a, b, order):
    options = "\n".join(f"{index + 1}. {RELATIONS[key]}" for index, key in enumerate(order))
    return template.format(a=a, b=b, options=options)


def main():
    rows = []
    families = ("negative", "leading_zero", "long_fraction", "scientific", "signed_zero")
    for family_index, family in enumerate(families):
        for local in range(100):
            equal = local % 2 == 0
            direction = 1 if (local // 2) % 2 == 0 else -1
            whole = 301 + family_index * 100 + local
            digit = (local * 3 + 1) % 10
            if family == "negative":
                a = f"-{whole}.{digit}"
                b = a + "0" * (1 + local % 4) if equal else str(Decimal(a) + Decimal(direction) / 10)
            elif family == "leading_zero":
                a = "00" + f"{whole}.{digit}"
                b = f"{whole}.{digit}" if equal else str(Decimal(f"{whole}.{digit}") + Decimal(direction) / 10)
            elif family == "long_fraction":
                frac = f"{digit}{(digit + 3) % 10}{(digit + 6) % 10}"
                a = f"{whole}.{frac}"
                b = a + "0" * (2 + local % 4) if equal else str(Decimal(a) + Decimal(direction) / Decimal(10_000))
            elif family == "scientific":
                exponent = 1 + local % 2
                coefficient = Decimal(whole) / (Decimal(10) ** exponent) + Decimal(digit) / (Decimal(10) ** (exponent + 1))
                a = f"{coefficient}e{exponent}"
                exact = Decimal(a)
                b = format(exact, "f") if equal else format(exact + Decimal(direction) / 10, "f")
            else:
                zeros = 1 + local % 5
                a = "-0." + "0" * zeros
                b = "0.0" if equal else ("0." + "0" * zeros + "1" if direction > 0 else "-0." + "0" * zeros + "1")
            truth = relation(a, b)
            desired_answer = 1 + local % 3
            remaining = [key for key in ("gt", "lt", "eq") if key != truth]
            order_list = [None, None, None]
            order_list[desired_answer - 1] = truth
            for position, key in zip([i for i, value in enumerate(order_list) if value is None], remaining):
                order_list[position] = key
            order = tuple(order_list)
            answer = order.index(truth) + 1
            template_index = (local // 2) % 2
            ca, cb = canonical(a), canonical(b)
            row = {"id": f"format-{family}-{local:03d}", "family": family, "local": local,
                   "a": a, "b": b, "canonical_a": ca, "canonical_b": cb,
                   "equal": equal, "relation": truth, "option_order": list(order),
                   "answer": answer, "template_index": template_index,
                   "prompt": render(TEMPLATES[template_index], a, b, order),
                   "canonical_prompt": render(TEMPLATES[template_index], ca, cb, order)}
            assert relation(ca, cb) == truth
            rows.append(row)
    assert len(rows) == 500 and len({row["id"] for row in rows}) == 500
    Path("data/format_robustness.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print("data/format_robustness.jsonl", len(rows))


if __name__ == "__main__":
    main()
