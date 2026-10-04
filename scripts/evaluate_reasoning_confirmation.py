#!/usr/bin/env python3
"""Run the frozen reasoning-enabled downstream confirmation on Qwen3 4B."""

import argparse
import json
import re
import time
from pathlib import Path

import mlx
import mlx.core as mx
import mlx_lm
from mlx_lm import batch_generate, load

from evaluate_numeric_invariance import numerically_equal


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
SYSTEM = (
    "You solve numerical reasoning problems accurately. Use at most three short "
    "calculation lines, then end with exactly one line in the form "
    "Final answer: <number>."
)
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?"
FINAL = re.compile(rf"final\s+answer\s*:\s*({NUMBER})", re.IGNORECASE)


def parse_final(raw):
    matches = FINAL.findall(raw.replace(",", "").replace("$", ""))
    if not matches:
        return None, None, "missing-final-marker"
    token = matches[-1]
    percent = token.endswith("%")
    if percent:
        token = token[:-1]
    try:
        from decimal import Decimal
        value = format(Decimal(token), "f")
    except Exception:
        return None, None, "invalid-final-number"
    return value, percent, "ok"


def build_user(row):
    unit = (
        "The expected answer is a percentage, so include the percent sign."
        if row["answer_is_percent"]
        else "The expected answer is a number without a percent sign."
    )
    return f"Solve the following problem. {unit}\n\n{row['problem']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--memory-limit-gb", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--clear-cache-every", type=int, default=20)
    args = parser.parse_args()

    mx.set_memory_limit(int(args.memory_limit_gb * (1024 ** 3)))
    dataset = ROOT / "data/downstream_mitigation_confirmation.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    output = ROOT / "results/reasoning_confirmation_Qwen-Qwen3-4B-Instruct-2507.jsonl"
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    complete = {row["id"] for row in existing if row.get("error") is None}

    model, tokenizer = load(MODEL, revision=REVISION)
    if getattr(model.args, "quantization", None) is not None:
        raise RuntimeError("reasoning confirmation requires native unquantized weights")

    pending = [row for row in rows if row["id"] not in complete]
    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start:batch_start + args.batch_size]
        prompts = []
        prompt_tokens = []
        for row in batch:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user(row)},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            prompt_tokens.append(tokenizer.encode(prompt))
        started = time.time()
        try:
            response = batch_generate(
                model, tokenizer, prompt_tokens, max_tokens=args.max_tokens, verbose=False
            )
            raw_responses = response.texts
        except Exception as exc:
            raise RuntimeError(f"batch failed at row {batch_start}: {exc!r}") from exc
        elapsed = round(time.time() - started, 4)
        for row, raw, prompt_ids in zip(batch, raw_responses, prompt_tokens):
            value, percent, parse_status = parse_final(raw)
            correct = (
                parse_status == "ok"
                and numerically_equal(value, row["answer_value"])
                and bool(percent) == bool(row["answer_is_percent"])
            )
            result = dict(row)
            result.update({
                "model": MODEL,
                "model_revision": REVISION,
                "checkpoint_precision": "native-unquantized",
                "backend": "mlx-lm-batch",
                "mlx_version": getattr(mlx, "__version__", "unknown"),
                "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
                "decoding": "greedy",
                "reasoning_enabled": True,
                "max_tokens": args.max_tokens,
                "batch_size": len(batch),
                "prompt_token_count": len(prompt_ids),
                "raw_response": raw,
                "prediction_value": value,
                "prediction_is_percent": percent,
                "parse_status": parse_status,
                "correct": correct,
                "error": None,
                "batch_elapsed_seconds": elapsed,
            })
            with output.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result, sort_keys=True) + "\n")
        completed = batch_start + len(batch)
        if completed % args.clear_cache_every == 0:
            mx.clear_cache()
            print(f"{len(complete) + completed}/{len(rows)}", flush=True)

    attempts = [json.loads(line) for line in output.read_text().splitlines()]
    final = {}
    for row in attempts:
        previous = final.get(row["id"])
        if previous is None or (previous.get("error") is not None and row.get("error") is None):
            final[row["id"]] = row
    if len(final) != len(rows) or any(row.get("error") is not None for row in final.values()):
        raise RuntimeError(f"incomplete reasoning confirmation: {len(final)}/{len(rows)}")
    order = {row["id"]: index for index, row in enumerate(rows)}
    ordered = sorted(final.values(), key=lambda row: order[row["id"]])
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered))
    print(output)


if __name__ == "__main__":
    main()
