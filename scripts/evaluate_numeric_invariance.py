#!/usr/bin/env python3
"""Evaluate the frozen four-format downstream invariance benchmark."""

import argparse
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import mlx
import mlx.core as mx
import mlx_lm
from mlx_lm import generate, load


ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "You solve numerical reasoning problems accurately."
NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?")


def parse_prediction(raw):
    cleaned = raw.replace(",", "").replace("$", "")
    matches = NUMBER.findall(cleaned)
    if not matches:
        return None, None, "no-number"
    token = matches[-1]
    is_percent = token.endswith("%")
    if is_percent:
        token = token[:-1]
    try:
        value = format(Decimal(token), "f")
    except InvalidOperation:
        return None, None, "invalid-number"
    return value, is_percent, "ok"


def build_user(row):
    suffix = ("The expected answer is a percentage. Include the percent sign."
              if row["answer_is_percent"] else
              "The expected answer is a number, not a sentence.")
    return (f"Solve this problem. {suffix} Return only the final numeric answer and no reasoning.\n\n"
            f"{row['problem']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--allow-quantized", action="store_true")
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data/numeric_invariance_downstream.jsonl")
    parser.add_argument("--only-base-id")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--memory-limit-gb", type=float)
    parser.add_argument("--clear-cache-every", type=int, default=20)
    args = parser.parse_args()
    if args.memory_limit_gb:
        mx.set_memory_limit(int(args.memory_limit_gb * (1024 ** 3)))
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    if args.only_base_id:
        rows = [row for row in rows if row["base_id"] == args.only_base_id]
        if not rows:
            raise RuntimeError(f"unknown base id: {args.only_base_id}")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model)
    output = ROOT / "results" / f"numeric_invariance_{safe}.jsonl"
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if existing and ({row["model"] for row in existing} != {args.model} or
                     {row["model_revision"] for row in existing} != {args.revision}):
        raise RuntimeError("existing output has different model provenance")
    complete = {row["id"] for row in existing if row.get("error") is None}

    model, tokenizer = load(args.model, revision=args.revision)
    quantization = getattr(model.args, "quantization", None)
    if quantization is not None and not args.allow_quantized:
        raise RuntimeError("quantized checkpoint requires --allow-quantized")
    precision = "quantized" if quantization is not None else "native-unquantized"
    template_mode = "system-role"
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "test"}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        template_mode = "system-prepended-to-user"

    for index, row in enumerate(rows):
        if row["id"] in complete:
            continue
        user = build_user(row)
        messages = ([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
                    if template_mode == "system-role" else
                    [{"role": "user", "content": SYSTEM + "\n\n" + user}])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        started = time.time()
        error = None
        try:
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
            value, is_percent, parse_status = parse_prediction(raw)
        except Exception as exc:
            raw, value, is_percent, parse_status, error = "", None, None, "error", repr(exc)
        correct = (parse_status == "ok" and value == row["answer_value"] and
                   (not row["answer_is_percent"] or is_percent))
        result = dict(row)
        result.update({
            "model": args.model,
            "model_revision": args.revision,
            "checkpoint_precision": precision,
            "quantization": quantization,
            "backend": "mlx-lm",
            "mlx_version": getattr(mlx, "__version__", "unknown"),
            "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
            "chat_template_mode": template_mode,
            "decoding": "greedy",
            "max_tokens": args.max_tokens,
            "prompt_token_count": len(tokenizer.encode(prompt)),
            "raw_response": raw,
            "prediction_value": value,
            "prediction_is_percent": is_percent,
            "parse_status": parse_status,
            "correct": correct,
            "error": error,
            "elapsed_seconds": round(time.time() - started, 4),
        })
        with output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, sort_keys=True) + "\n")
        if (index + 1) % args.clear_cache_every == 0:
            mx.clear_cache()
            print(f"{index + 1}/{len(rows)}", flush=True)
    attempts = [json.loads(line) for line in output.read_text().splitlines()]
    failures = [row for row in attempts if row.get("error") is not None]
    if failures:
        error_output = ROOT / "results" / f"numeric_invariance_execution_errors_{safe}.jsonl"
        error_output.write_text("".join(
            json.dumps(row, sort_keys=True) + "\n" for row in failures))
    final_by_id = {}
    for row in attempts:
        previous = final_by_id.get(row["id"])
        if previous is None or (previous.get("error") is not None and row.get("error") is None):
            final_by_id[row["id"]] = row
    final = list(final_by_id.values())
    if len(final) == len(rows) and not any(row.get("error") is not None for row in final):
        order = {row["id"]: i for i, row in enumerate(rows)}
        final.sort(key=lambda row: order[row["id"]])
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in final))
    print(output)


if __name__ == "__main__":
    main()
