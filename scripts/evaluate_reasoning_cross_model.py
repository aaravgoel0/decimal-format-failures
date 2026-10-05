#!/usr/bin/env python3
"""Run frozen reasoning-enabled confirmations for Llama or Gemma."""

import argparse
import gc
import json
import re
import time
from pathlib import Path

import mlx
import mlx.core as mx
import mlx_lm
from mlx_lm import batch_generate, generate, load

from evaluate_numeric_invariance import numerically_equal


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "llama": {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
    },
    "gemma": {
        "model": "google/gemma-2-9b-it",
        "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    },
}
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
    parser.add_argument("--model-key", choices=sorted(MODELS), required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--memory-limit-gb", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--clear-cache-every", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    allowed_batch_sizes = {1, 2} if args.model_key == "llama" else {1}
    if args.batch_size not in allowed_batch_sizes:
        raise RuntimeError(
            f"allowed batch sizes for {args.model_key}: {sorted(allowed_batch_sizes)}"
        )

    config = MODELS[args.model_key]
    mx.set_memory_limit(int(args.memory_limit_gb * (1024 ** 3)))
    if args.model_key == "gemma":
        mx.set_cache_limit(0)
    dataset = ROOT / "data/downstream_mitigation_confirmation.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", config["model"])
    output = ROOT / "results" / f"reasoning_confirmation_{safe}.jsonl"
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if existing and (
        {row["model"] for row in existing} != {config["model"]}
        or {row["model_revision"] for row in existing} != {config["revision"]}
    ):
        raise RuntimeError("existing output has different model provenance")
    complete = {row["id"] for row in existing if row.get("error") is None}

    model, tokenizer = load(config["model"], revision=config["revision"])
    quantization = getattr(model.args, "quantization", None)
    if quantization is not None:
        raise RuntimeError("reasoning confirmation requires native unquantized weights")
    template_mode = "system-role"
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        template_mode = "system-prepended-to-user"

    pending = [row for row in rows if row["id"] not in complete]
    print(
        f"model={config['model']} revision={config['revision']} "
        f"template={template_mode} complete={len(complete)} pending={len(pending)}",
        flush=True,
    )
    for batch_start in range(0, len(pending), args.batch_size):
        batch = pending[batch_start:batch_start + args.batch_size]
        prompts = []
        prompt_tokens = []
        for row in batch:
            user = build_user(row)
            messages = (
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
                if template_mode == "system-role"
                else [{"role": "user", "content": SYSTEM + "\n\n" + user}]
            )
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            prompt_tokens.append(tokenizer.encode(prompt))
        started = time.time()
        try:
            if args.model_key == "gemma":
                raw_responses = [
                    generate(
                        model,
                        tokenizer,
                        prompt=prompts[0],
                        max_tokens=args.max_tokens,
                        verbose=False,
                    )
                ]
            else:
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
                "model": config["model"],
                "model_revision": config["revision"],
                "checkpoint_precision": "native-unquantized",
                "quantization": quantization,
                "backend": "mlx-lm-serial" if args.model_key == "gemma" else "mlx-lm-batch",
                "mlx_version": getattr(mlx, "__version__", "unknown"),
                "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
                "chat_template_mode": template_mode,
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
            gc.collect()
        if completed % args.progress_every == 0 or completed == len(pending):
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
