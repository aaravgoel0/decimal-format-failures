#!/usr/bin/env python3
"""Build the amended untouched four-format downstream evaluation set."""

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


DECIMAL = re.compile(r"(?<![\w.])(?P<sign>[-+]?)(?P<int>\d[\d,]*)\.(?P<frac>\d+)(?![\w.])")
GSM_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
FINQA_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
TATQA_COMMIT = "870accc41953dcde885aabeb963d94aabdc0fbc3"
FORMS = ("canonical", "padded", "leading_zero", "scientific")


def canonical_parts(match):
    sign, integer, fraction = match.group("sign", "int", "frac")
    fraction = fraction.rstrip("0") or "0"
    return sign, integer, fraction


def scientific(value):
    number = Decimal(value.replace(",", ""))
    if number == 0:
        return "0e0"
    exponent = number.copy_abs().adjusted()
    coefficient = number.scaleb(-exponent)
    coefficient_text = format(coefficient, "f").rstrip("0").rstrip(".")
    return f"{coefficient_text}e{exponent}"


def rewrite_text(text, extra_zeros):
    changes = []

    def replace(match, form):
        sign, integer, fraction = canonical_parts(match)
        canonical = f"{sign}{integer}.{fraction}"
        if form == "canonical":
            rendered = canonical
        elif form == "padded":
            rendered = canonical + "0" * extra_zeros
        elif form == "leading_zero":
            rendered = f"{sign}00{integer}.{fraction}"
        elif form == "scientific":
            rendered = scientific(canonical)
        else:
            raise ValueError(form)
        return rendered

    rendered = {}
    matches = list(DECIMAL.finditer(text))
    if not matches:
        return None
    for form in FORMS:
        rendered[form] = DECIMAL.sub(lambda match: replace(match, form), text)
    for match in matches:
        sign, integer, fraction = canonical_parts(match)
        canonical = f"{sign}{integer}.{fraction}"
        forms = {
            "canonical": canonical,
            "padded": canonical + "0" * extra_zeros,
            "leading_zero": f"{sign}00{integer}.{fraction}",
            "scientific": scientific(canonical),
        }
        values = {Decimal(value.replace(",", "")) for value in forms.values()}
        if len(values) != 1:
            raise AssertionError((match.group(0), forms))
        changes.append({"source": match.group(0), **forms})
    return rendered, changes


def numeric_answer(value):
    if isinstance(value, list):
        if len(value) != 1:
            return None
        value = value[0]
    text = str(value).strip().replace(",", "").replace("$", "")
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1].strip()
    text = text.strip("() ")
    try:
        return format(Decimal(text), "f"), is_percent
    except InvalidOperation:
        return None


def make_rows(base_id, domain, source_index, source_id, source_commit, problem,
              answer_value, answer_is_percent, source_split="test"):
    extra_zeros = 2 + (source_index % 3)
    rewritten = rewrite_text(problem, extra_zeros)
    if rewritten is None:
        return []
    forms, changes = rewritten
    return [{
        "id": f"{base_id}-{form}",
        "base_id": base_id,
        "domain": domain,
        "source_split": source_split,
        "source_index": source_index,
        "source_id": source_id,
        "source_commit": source_commit,
        "form": form,
        "problem": forms[form],
        "answer_value": answer_value,
        "answer_is_percent": answer_is_percent,
        "transformed_literals": changes,
    } for form in FORMS]


def build_gsm8k(path, skip, limit):
    source = [json.loads(line) for line in path.read_text().splitlines()]
    output, eligible = [], 0
    for index, row in enumerate(source):
        if not DECIMAL.search(row["question"]):
            continue
        parsed = numeric_answer(row["answer"].rsplit("####", 1)[-1])
        if parsed is None:
            continue
        if eligible >= skip and len(output) < limit * len(FORMS):
            selected = eligible - skip
            output.extend(make_rows(
                f"gsm8k-new-{selected:03d}", "gsm8k", index, f"test-{index}",
                GSM_COMMIT, row["question"], parsed[0], parsed[1]))
        eligible += 1
        if len(output) == limit * len(FORMS):
            break
    if len(output) != limit * len(FORMS):
        raise RuntimeError(f"only {len(output) // len(FORMS)} eligible GSM8K rows")
    return output


def gold_evidence(row):
    def key(item):
        match = re.search(r"(\d+)$", item[0])
        return item[0].split("_")[0], int(match.group(1)) if match else 0
    return [text for _, text in sorted(row["qa"]["gold_inds"].items(), key=key)]


def build_finqa(path, skip, limit):
    source = json.loads(path.read_text())
    output, eligible = [], 0
    for index, row in enumerate(source):
        evidence = gold_evidence(row)
        problem = "Evidence:\n" + "\n".join(evidence) + "\nQuestion: " + row["qa"]["question"]
        if not DECIMAL.search(problem):
            continue
        parsed = numeric_answer(row["qa"]["answer"])
        if parsed is None:
            continue
        if eligible >= skip and len(output) < limit * len(FORMS):
            selected = eligible - skip
            output.extend(make_rows(
                f"finqa-new-{selected:03d}", "finqa", index,
                row.get("id", f"test-{index}"), FINQA_COMMIT, problem,
                parsed[0], parsed[1]))
        eligible += 1
        if len(output) == limit * len(FORMS):
            break
    if len(output) != limit * len(FORMS):
        raise RuntimeError(f"only {len(output) // len(FORMS)} eligible FinQA rows")
    return output


def render_tatqa_context(item, question):
    answer_from = str(question.get("answer_from", ""))
    sections = []
    if "table" in answer_from:
        table = item["table"]["table"]
        sections.append("Table:\n" + "\n".join(
            " | ".join(str(cell) for cell in row) for row in table))
    if "text" in answer_from:
        relevant = {str(order) for order in question.get("rel_paragraphs", [])}
        paragraphs = [
            paragraph["text"] for paragraph in item.get("paragraphs", [])
            if str(paragraph.get("order")) in relevant
        ]
        if paragraphs:
            sections.append("Relevant text:\n" + "\n".join(paragraphs))
    sections.append("Question: " + question["question"])
    scale = str(question.get("scale", "")).strip()
    if scale:
        sections.append("Answer scale: " + scale)
    return "\n".join(sections)


def build_tatqa(path, limit):
    source = json.loads(path.read_text())
    output, question_index = [], 0
    for item in source:
        for question in item.get("questions", []):
            parsed = numeric_answer(question.get("answer"))
            problem = render_tatqa_context(item, question)
            if parsed is not None and DECIMAL.search(problem):
                selected = len(output) // len(FORMS)
                output.extend(make_rows(
                    f"tatqa-new-{selected:03d}", "tatqa", question_index,
                    question["uid"], TATQA_COMMIT, problem, parsed[0],
                    str(question.get("scale", "")) == "percent",
                    source_split="dev"))
                if len(output) == limit * len(FORMS):
                    return output
            question_index += 1
    raise RuntimeError(f"only {len(output) // len(FORMS)} eligible TAT-QA rows")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--finqa", type=Path, required=True)
    parser.add_argument("--tatqa", type=Path, required=True)
    parser.add_argument("--skip", type=int, default=100)
    parser.add_argument("--gsm-count", type=int, default=13)
    parser.add_argument("--finqa-count", type=int, default=100)
    parser.add_argument("--tatqa-count", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("data/numeric_invariance_downstream.jsonl"))
    args = parser.parse_args()
    rows = build_gsm8k(args.gsm8k, args.skip, args.gsm_count)
    rows += build_finqa(args.finqa, args.skip, args.finqa_count)
    rows += build_tatqa(args.tatqa, args.tatqa_count)
    base_count = args.gsm_count + args.finqa_count + args.tatqa_count
    expected = base_count * len(FORMS)
    assert len(rows) == expected
    assert len({row["id"] for row in rows}) == expected
    assert len({row["base_id"] for row in rows}) == base_count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(args.output, len(rows))


if __name__ == "__main__":
    main()
