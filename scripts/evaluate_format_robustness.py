#!/usr/bin/env python3
"""Evaluate original and paired-canonicalized broad-format prompts."""
import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load

from evaluate import parse_answer

SYSTEM = "You are a helpful assistant that compares numbers."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in open("data/format_robustness.jsonl")]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model)
    output_path = Path(f"results/format_robustness_{safe}.jsonl")
    completed = set()
    if output_path.exists():
        for line in output_path.read_text().splitlines():
            old = json.loads(line)
            if old.get("parse_status") != "error":
                completed.add((old["id"], old["condition"]))
    model, tokenizer = load(args.model, revision=args.revision)
    template_mode = "system-role"
    try:
        tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": "test"}],
                                      tokenize=False, add_generation_prompt=True)
    except Exception:
        template_mode = "system-prepended-to-user"
    total = len(rows) * 2
    finished = 0
    for row in rows:
        for condition, prompt_key in (("original", "prompt"), ("canonicalized", "canonical_prompt")):
            finished += 1
            if (row["id"], condition) in completed:
                continue
            messages = ([{"role": "system", "content": SYSTEM}, {"role": "user", "content": row[prompt_key]}]
                        if template_mode == "system-role" else
                        [{"role": "user", "content": SYSTEM + "\n\n" + row[prompt_key]}])
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            started = time.time()
            error = None
            try:
                raw = generate(model, tokenizer, prompt=prompt, max_tokens=4, verbose=False)
                prediction, parse_status = parse_answer(raw)
            except Exception as exception:
                raw, prediction, parse_status, error = "", None, "error", repr(exception)
            result = dict(row)
            result.update({"condition": condition, "model": args.model, "revision": args.revision,
                           "backend": "mlx-lm", "template_mode": template_mode,
                           "raw_response": raw, "prediction": prediction,
                           "parse_status": parse_status, "correct": prediction == row["answer"],
                           "error": error, "elapsed_seconds": round(time.time() - started, 4)})
            with output_path.open("a") as file:
                file.write(json.dumps(result, sort_keys=True) + "\n")
            if finished % 10 == 0:
                mx.clear_cache()
            if finished % 25 == 0:
                print(f"{finished}/{total}", flush=True)
    print(output_path)


if __name__ == "__main__":
    main()
