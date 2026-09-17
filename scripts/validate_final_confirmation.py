#!/usr/bin/env python3
"""Validate frozen prompt and causal confirmation data before inference."""
import json
import math
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "You are a helpful assistant that compares numbers."
MODELS = (
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
    ("Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"),
    ("google/gemma-2-9b-it", "11c9b309abf73637e4b6f9a3fa1e92e615547819"),
)


def hits(sequence, subsequence):
    return [index for index in range(len(sequence) - len(subsequence) + 1)
            if sequence[index:index + len(subsequence)] == subsequence]


def formatted(tokenizer, user):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    mode = "system-role"
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=True, return_tensors=None)
    except Exception:
        mode = "system-prepended-to-user"
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": SYSTEM + "\n\n" + user}], tokenize=True,
            add_generation_prompt=True, return_tensors=None)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return ids, mode


def main():
    prompt_rows = [json.loads(line) for line in
                   (ROOT / "data/prompt_robustness_confirmation.jsonl").read_text().splitlines()]
    causal_rows = [json.loads(line) for line in
                   (ROOT / "data/causal_random_site_confirmation.jsonl").read_text().splitlines()]
    assert len(prompt_rows) == 3000 == len({row["id"] for row in prompt_rows})
    assert len(causal_rows) == 150 == len({row["id"] for row in causal_rows})
    prompt_pairs = {(row["base_id"], row["padded_position"]): row for row in prompt_rows}
    assert len({row["base_id"] for row in prompt_rows}) == 1500
    assert all((base, 1) in prompt_pairs and (base, 2) in prompt_pairs
               for base in {row["base_id"] for row in prompt_rows})
    compliance = [row for row in prompt_rows
                  if (row["digit"] + row["template_index"] + row["equal_label"] +
                      row["zeros"]) % 5 == 0]
    assert len(compliance) == 600
    for field, values in (("template_index", range(10)), ("equal_label", (1, 2, 3)),
                          ("zeros", range(1, 6)), ("digit", range(10))):
        counts = [sum(row[field] == value for row in compliance) for value in values]
        assert len(set(counts)) == 1, (field, counts)
    summaries = []
    for model, revision in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision,
                                                  local_files_only=True)
        for label in (1, 2, 3):
            assert len(tokenizer.encode(str(label), add_special_tokens=False)) == 1
        modes = set()
        prompt_lengths = []
        for row in prompt_rows:
            ids, mode = formatted(tokenizer, row["user_prompt"])
            modes.add(mode)
            prompt_lengths.append(len(ids))
            assert tokenizer.encode(row["canonical"], add_special_tokens=False)
            assert tokenizer.encode(row["padded"], add_special_tokens=False)
        minimum_combinations = math.inf
        causal_lengths = []
        for row in causal_rows:
            easy, mode = formatted(tokenizer, row["easy_prompt"])
            hard, hard_mode = formatted(tokenizer, row["hard_prompt"])
            assert mode == hard_mode and len(easy) == len(hard)
            modes.add(mode)
            short = tokenizer.encode(row["short"], add_special_tokens=False)
            padded = tokenizer.encode(row["padded"], add_special_tokens=False)
            easy_short, easy_padded = min(hits(easy, short)), max(hits(easy, padded))
            hard_padded, hard_short = min(hits(hard, padded)), max(hits(hard, short))
            excluded = {len(hard) - 1}
            for start, count in ((easy_short, len(short)), (easy_padded, len(padded)),
                                 (hard_short, len(short)), (hard_padded, len(padded))):
                excluded.update(range(start, start + count))
            eligible = [position for position in range(1, len(hard) - 1)
                        if position not in excluded]
            for size in (len(short), len(padded), len(short) + len(padded)):
                minimum_combinations = min(minimum_combinations,
                                           math.comb(len(eligible), size))
                assert math.comb(len(eligible), size) >= 50
            causal_lengths.append(len(hard))
        summaries.append({
            "model": model,
            "revision": revision,
            "template_modes": sorted(modes),
            "prompt_length_range": [min(prompt_lengths), max(prompt_lengths)],
            "causal_length_range": [min(causal_lengths), max(causal_lengths)],
            "minimum_random_subsets": minimum_combinations,
        })
    print(json.dumps(summaries, indent=2))
    print("PASS: frozen confirmation datasets and token locations validated")


if __name__ == "__main__":
    main()
