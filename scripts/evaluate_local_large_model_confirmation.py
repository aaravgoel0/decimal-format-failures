#!/usr/bin/env python3
"""Run the frozen local Gemma 3 27B 4-bit reasoning confirmation."""

import argparse
import gc
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import mlx
import mlx.core as mx
import mlx_lm
from huggingface_hub import hf_hub_download
from mlx_lm import generate, load


ROOT = Path(__file__).resolve().parents[1]
MODEL = "mlx-community/gemma-3-text-27b-it-4bit"
MODEL_REVISION = "feccbf793f8404211939458acfa9b857f22a9fe4"
SOURCE_MODEL = "google/gemma-3-27b-it"
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
        value = format(Decimal(token), "f")
    except InvalidOperation:
        return None, None, "invalid-final-number"
    return value, percent, "ok"


def numerically_equal(left, right):
    try:
        return Decimal(str(left)) == Decimal(str(right))
    except (InvalidOperation, TypeError, ValueError):
        return False


def build_user(row):
    unit = (
        "The expected answer is a percentage, so include the percent sign."
        if row["answer_is_percent"]
        else "The expected answer is a number without a percent sign."
    )
    return f"Solve the following problem. {unit}\n\n{row['problem']}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--memory-limit-gb", type=float, default=18.0)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if not 1.0 <= args.memory_limit_gb <= 18.0:
        raise RuntimeError("memory limit must remain between 1 and 18 GiB")
    if args.max_tokens != 256:
        raise RuntimeError("frozen maximum generation is 256 tokens")

    mx.set_memory_limit(int(args.memory_limit_gb * (1024 ** 3)))
    mx.set_cache_limit(0)
    dataset = ROOT / "data/downstream_mitigation_confirmation.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    if args.limit is not None:
        rows = rows[:args.limit]
    output = ROOT / "results/reasoning_confirmation_mlx-community-gemma-3-text-27b-it-4bit.jsonl"
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if existing and (
        {row["model"] for row in existing} != {MODEL}
        or {row["model_revision"] for row in existing} != {MODEL_REVISION}
    ):
        raise RuntimeError("existing output has different model provenance")
    complete = {row["id"] for row in existing if row.get("error") is None}

    config_path = hf_hub_download(MODEL, "config.json", revision=MODEL_REVISION)
    model_config = json.loads(Path(config_path).read_text())
    quantization = model_config.get("quantization_config") or model_config.get("quantization")
    if not isinstance(quantization, dict) or quantization.get("bits") != 4:
        raise RuntimeError("frozen scaling control requires the pinned quantized checkpoint")
    model, tokenizer = load(MODEL, revision=MODEL_REVISION)
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
        f"model={MODEL} revision={MODEL_REVISION} template={template_mode} "
        f"complete={len(complete)} pending={len(pending)}",
        flush=True,
    )
    for index, row in enumerate(pending, start=1):
        user = build_user(row)
        messages = (
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            if template_mode == "system-role"
            else [{"role": "user", "content": SYSTEM + "\n\n" + user}]
        )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(prompt)
        started = time.time()
        try:
            raw = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"generation failed at pending row {index - 1}: {exc!r}") from exc
        value, percent, parse_status = parse_final(raw)
        correct = (
            parse_status == "ok"
            and numerically_equal(value, row["answer_value"])
            and bool(percent) == bool(row["answer_is_percent"])
        )
        result = dict(row)
        result.update({
            "model": MODEL,
            "source_model": SOURCE_MODEL,
            "model_revision": MODEL_REVISION,
            "checkpoint_precision": "4-bit-quantized",
            "quantization": quantization,
            "backend": "mlx-lm-serial",
            "mlx_version": getattr(mlx, "__version__", "unknown"),
            "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
            "chat_template_mode": template_mode,
            "decoding": "greedy",
            "reasoning_enabled": True,
            "max_tokens": args.max_tokens,
            "batch_size": 1,
            "memory_limit_gb": args.memory_limit_gb,
            "prompt_token_count": len(prompt_ids),
            "raw_response": raw,
            "prediction_value": value,
            "prediction_is_percent": percent,
            "parse_status": parse_status,
            "correct": correct,
            "error": None,
            "elapsed_seconds": round(time.time() - started, 4),
        })
        with output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, sort_keys=True) + "\n")
        mx.clear_cache()
        gc.collect()
        if index % args.progress_every == 0 or index == len(pending):
            print(f"{len(complete) + index}/{len(rows)}", flush=True)

    attempts = [json.loads(line) for line in output.read_text().splitlines()]
    final = {}
    for row in attempts:
        if row["id"] in {target["id"] for target in rows}:
            final[row["id"]] = row
    if len(final) == len(rows) and not any(row.get("error") is not None for row in final.values()):
        order = {row["id"]: index for index, row in enumerate(rows)}
        ordered = sorted(final.values(), key=lambda row: order[row["id"]])
        if args.limit is None:
            output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered))
            print(output)
    elif args.limit is None:
        raise RuntimeError(f"incomplete local confirmation: {len(final)}/{len(rows)}")


if __name__ == "__main__":
    main()
