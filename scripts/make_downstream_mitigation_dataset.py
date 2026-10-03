#!/usr/bin/env python3
"""Build the frozen fresh downstream mitigation confirmation set."""

import argparse
import json
from pathlib import Path

from make_numeric_invariance_dataset_amended import (
    DECIMAL,
    FORMS,
    TATQA_COMMIT,
    build_finqa,
    make_rows,
    numeric_answer,
    render_tatqa_context,
)


ROOT = Path(__file__).resolve().parents[1]


def build_tatqa(path, skip, limit):
    source = json.loads(path.read_text())
    output = []
    eligible = 0
    question_index = 0
    for item in source:
        for question in item.get("questions", []):
            parsed = numeric_answer(question.get("answer"))
            problem = render_tatqa_context(item, question)
            if parsed is not None and DECIMAL.search(problem):
                if eligible >= skip and len(output) < limit * len(FORMS):
                    selected = eligible - skip
                    output.extend(make_rows(
                        f"tatqa-mitigation-{selected:03d}",
                        "tatqa",
                        question_index,
                        question["uid"],
                        TATQA_COMMIT,
                        problem,
                        parsed[0],
                        str(question.get("scale", "")) == "percent",
                        source_split="dev",
                    ))
                eligible += 1
                if len(output) == limit * len(FORMS):
                    return output
            question_index += 1
    raise RuntimeError(f"only {len(output) // len(FORMS)} eligible TAT-QA rows after skip")


def prior_source_ids():
    paths = [
        ROOT / "data/downstream_format_invariance.jsonl",
        ROOT / "data/numeric_invariance_downstream.jsonl",
    ]
    return {
        (row["domain"], row["source_id"])
        for path in paths
        for row in (json.loads(line) for line in path.read_text().splitlines())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--finqa", type=Path, required=True)
    parser.add_argument("--tatqa", type=Path, required=True)
    parser.add_argument("--finqa-skip", type=int, default=200)
    parser.add_argument("--tatqa-skip", type=int, default=100)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/downstream_mitigation_confirmation.jsonl",
    )
    args = parser.parse_args()

    rows = build_finqa(args.finqa, args.finqa_skip, args.count)
    for row in rows:
        selected = int(row["base_id"].rsplit("-", 1)[-1])
        row["base_id"] = f"finqa-mitigation-{selected:03d}"
        row["id"] = f"{row['base_id']}-{row['form']}"
    rows += build_tatqa(args.tatqa, args.tatqa_skip, args.count)

    expected_bases = 2 * args.count
    expected_rows = expected_bases * len(FORMS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, got {len(rows)}")
    if len({row["id"] for row in rows}) != expected_rows:
        raise RuntimeError("duplicate row ids")
    if len({row["base_id"] for row in rows}) != expected_bases:
        raise RuntimeError("duplicate base ids")
    overlap = {(row["domain"], row["source_id"]) for row in rows} & prior_source_ids()
    if overlap:
        raise RuntimeError(f"source overlap with earlier experiments: {sorted(overlap)[:5]}")

    groups = {}
    for row in rows:
        groups.setdefault(row["base_id"], {})[row["form"]] = row
    for base_id, forms in groups.items():
        if set(forms) != set(FORMS):
            raise RuntimeError(f"incomplete forms: {base_id}")
        canonical = forms["canonical"]
        for form, row in forms.items():
            if row["answer_value"] != canonical["answer_value"]:
                raise RuntimeError(f"answer changed: {base_id} {form}")
        for change in canonical["transformed_literals"]:
            values = {change[form] for form in FORMS}
            if len(values) < 2:
                raise RuntimeError(f"degenerate rewrite: {base_id}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    print(args.output, expected_rows, expected_bases)


if __name__ == "__main__":
    main()
