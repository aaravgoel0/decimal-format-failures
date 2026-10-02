#!/usr/bin/env python3
"""Generate the fresh selective-normalization confirmation dataset."""

import json
from decimal import Decimal
from pathlib import Path


RELATIONS = {
    "gt": "The first value is larger",
    "lt": "The second value is larger",
    "eq": "The values are equal",
}
TEMPLATES = [
    "Decide which statement matches X = {a} and Y = {b}.\n{options}\nOutput only the statement number.",
    "Read these two numerical values.\nX: {a}\nY: {b}\n{options}\nAnswer using only 1, 2, or 3.",
    "Choose the true comparison for {a} and {b}.\n{options}\nReturn the correct option number only.",
]
FAMILIES = ("negative", "leading_zero", "long_fraction", "scientific", "signed_zero")
SELECTIVE_FAMILIES = {"leading_zero", "scientific"}


def canonical(text):
    value = Decimal(text)
    if value == 0:
        return "0"
    result = format(value, "f")
    sign = "-" if result.startswith("-") else ""
    whole, dot, fraction = result.lstrip("-").partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return sign + whole + (("." + fraction) if dot and fraction else "")


def relation(a, b):
    x, y = Decimal(a), Decimal(b)
    return "gt" if x > y else "lt" if x < y else "eq"


def render(template, a, b, order):
    options = "\n".join(f"{index + 1}. {RELATIONS[key]}" for index, key in enumerate(order))
    return template.format(a=a, b=b, options=options)


def values(family, local, whole, equal, direction):
    digit = (local * 7 + 3) % 10
    if family == "negative":
        a = f"-{whole}.{digit}"
        b = a + "0" * (2 + local % 3) if equal else str(Decimal(a) + Decimal(direction) / 10)
    elif family == "leading_zero":
        a = "000" + f"{whole}.{digit}"
        b = f"{whole}.{digit}" if equal else str(Decimal(f"{whole}.{digit}") + Decimal(direction) / 10)
    elif family == "long_fraction":
        frac = f"{digit}{(digit + 2) % 10}{(digit + 5) % 10}{(digit + 8) % 10}"
        a = f"{whole}.{frac}"
        b = a + "0" * (3 + local % 3) if equal else str(Decimal(a) + Decimal(direction) / Decimal(100_000))
    elif family == "scientific":
        exponent = 1 + local % 3
        coefficient = Decimal(whole) / (Decimal(10) ** exponent)
        coefficient += Decimal(digit) / (Decimal(10) ** (exponent + 1))
        a = f"{coefficient}e{exponent}"
        exact = Decimal(a)
        b = format(exact, "f") if equal else format(exact + Decimal(direction) / 10, "f")
    elif family == "signed_zero":
        zeros = 2 + local % 4
        a = "+0." + "0" * zeros
        if equal:
            b = "-0.0"
        elif direction > 0:
            b = "0." + "0" * zeros + "1"
        else:
            b = "-0." + "0" * zeros + "1"
    else:
        raise ValueError(family)
    return a, b


def main():
    rows = []
    for family_index, family in enumerate(FAMILIES):
        for local in range(120):
            equal = local % 2 == 0
            direction = 1 if (local // 2) % 2 == 0 else -1
            whole = 1301 + family_index * 120 + local
            a, b = values(family, local, whole, equal, direction)
            truth = relation(a, b)
            desired_answer = 1 + local % 3
            remaining = [key for key in ("gt", "lt", "eq") if key != truth]
            order = [None, None, None]
            order[desired_answer - 1] = truth
            for position, key in zip([i for i, value in enumerate(order) if value is None], remaining):
                order[position] = key
            answer = order.index(truth) + 1
            template_index = (local // 2) % 3
            ca, cb = canonical(a), canonical(b)
            original_prompt = render(TEMPLATES[template_index], a, b, order)
            canonical_prompt = render(TEMPLATES[template_index], ca, cb, order)
            selective_prompt = canonical_prompt if family in SELECTIVE_FAMILIES else original_prompt
            row = {
                "id": f"selective-{family}-{local:03d}",
                "family": family,
                "local": local,
                "a": a,
                "b": b,
                "canonical_a": ca,
                "canonical_b": cb,
                "equal": equal,
                "relation": truth,
                "option_order": order,
                "answer": answer,
                "template_index": template_index,
                "original_prompt": original_prompt,
                "full_canonical_prompt": canonical_prompt,
                "selective_prompt": selective_prompt,
                "selective_changed": family in SELECTIVE_FAMILIES,
            }
            assert relation(ca, cb) == truth
            assert (selective_prompt == original_prompt) is (family not in SELECTIVE_FAMILIES)
            rows.append(row)
    assert len(rows) == 600 and len({row["id"] for row in rows}) == 600
    for family in FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        assert len(selected) == 120
        assert sum(row["equal"] for row in selected) == 60
        assert {row["template_index"] for row in selected} == {0, 1, 2}
        assert all(sum(row["answer"] == answer for row in selected) == 40 for answer in (1, 2, 3))
    output = Path("data/selective_normalization_confirmation.jsonl")
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(output, len(rows))


if __name__ == "__main__":
    main()
