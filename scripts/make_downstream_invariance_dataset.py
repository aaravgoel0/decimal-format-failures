#!/usr/bin/env python3
"""Build paired trailing-zero variants from official GSM8K and FinQA tests."""

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


DECIMAL = re.compile(r"(?<![\w.])(?P<sign>[-+]?)(?P<int>\d[\d,]*)\.(?P<frac>\d+)(?![\w.])")
GSM_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
FINQA_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"


def render_forms(text, extra_zeros):
    changes = []

    def replace(match, condition):
        sign, integer, fraction = match.group("sign", "int", "frac")
        canonical_fraction = fraction.rstrip("0") or "0"
        canonical = f"{sign}{integer}.{canonical_fraction}"
        padded = canonical + "0" * extra_zeros
        changes.append({"source": match.group(0), "canonical": canonical, "padded": padded})
        return canonical if condition == "canonical" else padded

    canonical_changes = []

    def canonical_repl(match):
        sign, integer, fraction = match.group("sign", "int", "frac")
        canonical_fraction = fraction.rstrip("0") or "0"
        canonical = f"{sign}{integer}.{canonical_fraction}"
        padded = canonical + "0" * extra_zeros
        canonical_changes.append({"source": match.group(0), "canonical": canonical, "padded": padded})
        return canonical

    canonical_text = DECIMAL.sub(canonical_repl, text)
    by_source = iter(canonical_changes)

    def padded_repl(match):
        change = next(by_source)
        return change["padded"]

    padded_text = DECIMAL.sub(padded_repl, text)
    if next(by_source, None) is not None:
        raise RuntimeError("literal accounting failed")
    return canonical_text, padded_text, canonical_changes


def numeric_answer(value):
    text = str(value).strip().replace(",", "").replace("$", "")
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    text = text.strip("() ")
    try:
        return format(Decimal(text), "f"), is_percent
    except InvalidOperation:
        return None


def paired_rows(base_id, domain, source_index, source_id, source_commit,
                problem, answer_value, answer_is_percent):
    extra_zeros = 2 + (source_index % 3)
    canonical, padded, changes = render_forms(problem, extra_zeros)
    if not changes or canonical == padded:
        return []
    rows = []
    for condition, rendered in (("canonical", canonical), ("padded", padded)):
        rows.append({
            "id": f"{base_id}-{condition}",
            "base_id": base_id,
            "domain": domain,
            "source_split": "test",
            "source_index": source_index,
            "source_id": source_id,
            "source_commit": source_commit,
            "condition": condition,
            "extra_zeros": extra_zeros,
            "problem": rendered,
            "answer_value": answer_value,
            "answer_is_percent": answer_is_percent,
            "transformed_literals": changes,
        })
    return rows


def build_gsm8k(path, limit):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    output = []
    selected = 0
    for index, row in enumerate(rows):
        if not DECIMAL.search(row["question"]):
            continue
        parsed = numeric_answer(row["answer"].rsplit("####", 1)[-1])
        if parsed is None:
            continue
        output.extend(paired_rows(
            f"gsm8k-{selected:03d}", "gsm8k", index, f"test-{index}", GSM_COMMIT,
            row["question"], parsed[0], parsed[1]))
        selected += 1
        if selected == limit:
            break
    if selected != limit:
        raise RuntimeError(f"only {selected} eligible GSM8K rows")
    return output


def gold_evidence(row):
    items = row["qa"]["gold_inds"].items()

    def key(item):
        match = re.search(r"(\d+)$", item[0])
        return (item[0].split("_")[0], int(match.group(1)) if match else 0)

    return [text for _, text in sorted(items, key=key)]


def build_finqa(path, limit):
    rows = json.loads(path.read_text())
    output = []
    selected = 0
    for index, row in enumerate(rows):
        evidence = gold_evidence(row)
        problem = "Evidence:\n" + "\n".join(evidence) + "\nQuestion: " + row["qa"]["question"]
        if not DECIMAL.search(problem):
            continue
        parsed = numeric_answer(row["qa"]["answer"])
        if parsed is None:
            continue
        output.extend(paired_rows(
            f"finqa-{selected:03d}", "finqa", index, row.get("id", f"test-{index}"),
            FINQA_COMMIT, problem, parsed[0], parsed[1]))
        selected += 1
        if selected == limit:
            break
    if selected != limit:
        raise RuntimeError(f"only {selected} eligible FinQA rows")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--finqa", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("data/downstream_format_invariance.jsonl"))
    args = parser.parse_args()
    rows = build_gsm8k(args.gsm8k, args.per_domain)
    rows += build_finqa(args.finqa, args.per_domain)
    if len(rows) != 4 * args.per_domain:
        raise RuntimeError("unexpected output size")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate row IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(args.output, len(rows))


if __name__ == "__main__":
    main()
