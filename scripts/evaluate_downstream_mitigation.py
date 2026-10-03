#!/usr/bin/env python3
"""Evaluate the frozen downstream mitigation set on a local MLX checkpoint."""

import argparse
import json
import re
import time
from pathlib import Path

import mlx
import mlx.core as mx
import mlx_lm
from mlx_lm import generate, load

from evaluate_numeric_invariance import SYSTEM, build_user, parse_prediction


ROOT = Path(__file__).resolve().parents[1]


def safe_name(model):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/downstream_mitigation_confirmation.jsonl",
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--memory-limit-gb", type=float, default=18.0)
    parser.add_argument("--clear-cache-every", type=int, default=20)
    args = parser.parse_args()

    mx.set_memory_limit(int(args.memory_limit_gb * (1024 ** 3)))
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    output = ROOT / "results" / f"downstream_mitigation_{safe_name(args.model)}.jsonl"
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if existing and (
        {row["model"] for row in existing} != {args.model}
        or {row["model_revision"] for row in existing} != {args.revision}
    ):
        raise RuntimeError("existing output has different model provenance")
    complete = {row["id"] for row in existing if row.get("error") is None}

    model, tokenizer = load(args.model, revision=args.revision)
    quantization = getattr(model.args, "quantization", None)
    if quantization is not None:
        raise RuntimeError("primary checkpoint must be native precision")
    template_mode = "system-role"
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        template_mode = "system-prepended-to-user"

    for index, row in enumerate(rows):
        if row["id"] in complete:
            continue
        user = build_user(row)
        messages = (
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            if template_mode == "system-role"
            else [{"role": "user", "content": SYSTEM + "\n\n" + user}]
        )
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        started = time.time()
        error = None
        try:
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
            value, is_percent, parse_status = parse_prediction(raw)
        except Exception as exc:
            raw, value, is_percent, parse_status, error = "", None, None, "error", repr(exc)
        correct = (
            parse_status == "ok"
            and value == row["answer_value"]
            and (not row["answer_is_percent"] or is_percent)
        )
        result = dict(row)
        result.update({
            "model": args.model,
            "model_revision": args.revision,
            "checkpoint_precision": "native-unquantized",
            "quantization": None,
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
    final = {}
    for row in attempts:
        previous = final.get(row["id"])
        if previous is None or (previous.get("error") is not None and row.get("error") is None):
            final[row["id"]] = row
    if len(final) != len(rows) or any(row.get("error") is not None for row in final.values()):
        failures = [row for row in final.values() if row.get("error") is not None]
        error_output = ROOT / "results" / f"downstream_mitigation_execution_errors_{safe_name(args.model)}.jsonl"
        error_output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in failures))
        raise RuntimeError(f"incomplete evaluation: {len(final)}/{len(rows)}, {len(failures)} errors")
    order = {row["id"]: index for index, row in enumerate(rows)}
    ordered = sorted(final.values(), key=lambda row: order[row["id"]])
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered))
    print(output)


if __name__ == "__main__":
    main()
