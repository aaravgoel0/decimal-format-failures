#!/usr/bin/env python3
"""Evaluate original, full, and selective normalization prompts."""

import argparse
import json
import re
import time
from pathlib import Path

import mlx
import mlx.core as mx
import mlx_lm
from mlx_lm import generate, load

from evaluate import parse_answer


ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "You are a helpful assistant that compares numbers."
CONDITIONS = (
    ("original", "original_prompt"),
    ("full_canonical", "full_canonical_prompt"),
    ("selective", "selective_prompt"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--memory-limit-gb", type=float)
    parser.add_argument("--clear-cache-every", type=int, default=20)
    args = parser.parse_args()
    if args.memory_limit_gb:
        mx.set_memory_limit(int(args.memory_limit_gb * (1024 ** 3)))
    dataset = ROOT / "data/selective_normalization_confirmation.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model)
    output = ROOT / "results" / f"selective_normalization_{safe}.jsonl"
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if existing and ({row["model"] for row in existing} != {args.model} or
                     {row["model_revision"] for row in existing} != {args.revision}):
        raise RuntimeError("existing output has different model provenance")
    complete = {(row["id"], row["condition"]) for row in existing if row.get("error") is None}
    successful = {(row["id"], row["condition"]): row
                  for row in existing if row.get("error") is None}

    model, tokenizer = load(args.model, revision=args.revision)
    if getattr(model.args, "quantization", None) is not None:
        raise RuntimeError("native unquantized weights required")
    template_mode = "system-role"
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "test"}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        template_mode = "system-prepended-to-user"

    finished, total = 0, len(rows) * len(CONDITIONS)
    for row in rows:
        for condition, prompt_key in CONDITIONS:
            finished += 1
            if (row["id"], condition) in complete:
                continue
            if condition == "selective" and row["selective_prompt"] == row["original_prompt"]:
                source = successful.get((row["id"], "original"))
                if source is not None:
                    result = dict(source)
                    result["condition"] = "selective"
                    result["reused_from_condition"] = "original"
                    with output.open("a", encoding="utf-8") as file:
                        file.write(json.dumps(result, sort_keys=True) + "\n")
                    successful[(row["id"], "selective")] = result
                    continue
            user = row[prompt_key]
            messages = ([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
                        if template_mode == "system-role" else
                        [{"role": "user", "content": SYSTEM + "\n\n" + user}])
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            started = time.time()
            error = None
            try:
                raw = generate(model, tokenizer, prompt=prompt, max_tokens=4, verbose=False)
                prediction, parse_status = parse_answer(raw)
            except Exception as exc:
                raw, prediction, parse_status, error = "", None, "error", repr(exc)
            result = dict(row)
            result.update({
                "condition": condition,
                "model": args.model,
                "model_revision": args.revision,
                "checkpoint_precision": "native-unquantized",
                "backend": "mlx-lm",
                "mlx_version": getattr(mlx, "__version__", "unknown"),
                "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
                "chat_template_mode": template_mode,
                "decoding": "greedy",
                "raw_response": raw,
                "prediction": prediction,
                "parse_status": parse_status,
                "correct": prediction == row["answer"],
                "error": error,
                "elapsed_seconds": round(time.time() - started, 4),
            })
            with output.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result, sort_keys=True) + "\n")
            if error is None:
                successful[(row["id"], condition)] = result
            if finished % args.clear_cache_every == 0:
                mx.clear_cache()
                print(f"{finished}/{total}", flush=True)
    attempts = [json.loads(line) for line in output.read_text().splitlines()]
    failures = [row for row in attempts if row.get("error") is not None]
    if failures:
        error_output = ROOT / "results" / f"selective_normalization_execution_errors_{safe}.jsonl"
        error_output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in failures))
    final_by_key = {}
    for row in attempts:
        key = (row["id"], row["condition"])
        previous = final_by_key.get(key)
        if previous is None or (previous.get("error") is not None and row.get("error") is None):
            final_by_key[key] = row
    final = list(final_by_key.values())
    if len(final) == total and not any(row.get("error") is not None for row in final):
        order = {(row["id"], condition): i * len(CONDITIONS) + j
                 for i, row in enumerate(rows) for j, (condition, _) in enumerate(CONDITIONS)}
        final.sort(key=lambda row: order[row["id"], row["condition"]])
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in final))
    print(output)


if __name__ == "__main__":
    main()
