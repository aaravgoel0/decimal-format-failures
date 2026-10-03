#!/usr/bin/env python3
"""Evaluate paired downstream numerical-format prompts with an exact checkpoint."""

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


def numerically_equal(left, right):
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, TypeError, ValueError):
        return False


def build_user(row):
    suffix = ("The expected answer is a percentage. Include the percent sign."
              if row["answer_is_percent"] else
              "The expected answer is a number, not a sentence.")
    return (
        f"Solve this problem. {suffix} Return only the final numeric answer and no reasoning.\n\n"
        f"{row['problem']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data/downstream_format_invariance.jsonl")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--memory-limit-gb", type=float)
    args = parser.parse_args()
    if args.memory_limit_gb:
        limit = int(args.memory_limit_gb * (1024 ** 3))
        mx.set_memory_limit(limit)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model)
    output = ROOT / "results" / f"downstream_format_invariance_{safe}.jsonl"
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if existing and ({row["model"] for row in existing} != {args.model} or
                     {row["model_revision"] for row in existing} != {args.revision}):
        raise RuntimeError("existing output has different model provenance")
    successful = [row for row in existing if row.get("error") is None]
    if len(successful) != len(existing):
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in successful))
        existing = successful
    complete = {row["id"] for row in existing}

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

    for index, row in enumerate(rows):
        if row["id"] in complete:
            continue
        user = build_user(row)
        messages = ([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
                    if template_mode == "system-role" else
                    [{"role": "user", "content": SYSTEM + "\n\n" + user}])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt)
        started = time.time()
        error = None
        try:
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens,
                           verbose=False)
            value, is_percent, parse_status = parse_prediction(raw)
        except Exception as exc:
            raw, value, is_percent, parse_status, error = "", None, None, "error", repr(exc)
        correct = (parse_status == "ok" and numerically_equal(value, row["answer_value"]) and
                   bool(is_percent) == bool(row["answer_is_percent"]))
        result = dict(row)
        result.update({
            "model": args.model,
            "model_revision": args.revision,
            "checkpoint_precision": "native-unquantized",
            "backend": "mlx-lm",
            "mlx_version": getattr(mlx, "__version__", "unknown"),
            "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
            "chat_template_mode": template_mode,
            "decoding": "greedy",
            "max_tokens": args.max_tokens,
            "prompt_token_count": len(prompt_ids),
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
        if (index + 1) % 20 == 0:
            print(f"{index + 1}/{len(rows)}", flush=True)
    final = [json.loads(line) for line in output.read_text().splitlines()]
    if len(final) == len(rows) and len({row["id"] for row in final}) == len(rows):
        order = {row["id"]: i for i, row in enumerate(rows)}
        final.sort(key=lambda row: order[row["id"]])
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in final))
    print(output)


if __name__ == "__main__":
    main()
