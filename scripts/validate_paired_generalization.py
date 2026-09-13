#!/usr/bin/env python3
"""Integrity checks for the frozen paired order experiment."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "paired_order_generalization.jsonl"
EXPECTED = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
    "Qwen/Qwen3-4B-Instruct-2507": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "google/gemma-2-9b-it": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
}


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    manifest = json.loads((ROOT / "PAIRED_GENERALIZATION_FREEZE_MANIFEST.json").read_text())
    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    assert digest == manifest["dataset_sha256"]
    data = load_jsonl(DATA_PATH)
    assert len(data) == 600 == len({row["id"] for row in data})
    assert len({row["base_id"] for row in data}) == 300
    assert all(count == 1 for count in Counter(
        (row["base_id"], row["padded_position"]) for row in data
    ).values())
    assert Counter(row["template_index"] for row in data) == {0: 300, 1: 300}
    assert Counter(row["answer"] for row in data) == {1: 200, 2: 200, 3: 200}
    assert Counter(row["padded_position"] for row in data) == {1: 300, 2: 300}
    assert Counter(row["digit"] for row in data) == {digit: 60 for digit in range(10)}
    assert Counter(row["zeros"] for row in data) == {zeros: 120 for zeros in range(1, 6)}
    source = {row["id"]: row for row in data}

    files = sorted((ROOT / "results").glob("paired_order_generalization_*.jsonl"))
    if len(files) != 3:
        raise RuntimeError(f"expected 3 model result files, found {len(files)}")
    report = []
    for path in files:
        rows = load_jsonl(path)
        assert len(rows) == 600 == len({row["id"] for row in rows})
        assert {row["model"] for row in rows} == {rows[0]["model"]}
        assert rows[0]["model"] in EXPECTED
        assert {row["model_revision"] for row in rows} == {EXPECTED[rows[0]["model"]]}
        assert not any(row["parse_status"] == "error" or row["error"] for row in rows)
        for row in rows:
            original = source[row["id"]]
            assert all(row[key] == value for key, value in original.items())
            assert row["checkpoint_precision"] == "native-unquantized"
            assert row["decoding"] == "greedy-temperature-0" and row["max_tokens"] == 4
            assert row["prompt_token_ids"] and row["sequence_length"] == len(row["prompt_token_ids"])
            assert {int(key) for key in row["answer_label_logits"]} == {1, 2, 3}
            assert row["correct"] == (row["parse_status"] == "exact" and row["prediction"] == row["answer"])
            assert row["constrained_correct"] == (row["constrained_prediction"] == row["answer"])
        report.append({
            "model": rows[0]["model"],
            "rows": len(rows),
            "strict_correct": sum(row["correct"] for row in rows),
            "constrained_correct": sum(row["constrained_correct"] for row in rows),
            "nonexact": sum(row["parse_status"] != "exact" for row in rows),
        })
    print(json.dumps({"dataset_sha256": digest, "models": report}, indent=2))


if __name__ == "__main__":
    main()
