#!/usr/bin/env python3
"""Evaluate the frozen paired dataset on one exact MLX checkpoint."""
import argparse
import json
import re
import time
from pathlib import Path

import mlx
import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from evaluate import parse_answer

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "You are a helpful assistant that compares numbers."


def formatted(tokenizer, user, mode):
    messages = ([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
                if mode == "system-role" else [{"role": "user", "content": SYSTEM + "\n\n" + user}])
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors=None)
    return text, ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / "data/paired_order_generalization.jsonl").read_text().splitlines()]
    model, tokenizer = load(args.model, revision=args.revision)
    if getattr(model.args, "quantization", None) is not None:
        raise RuntimeError("The paired protocol forbids quantized checkpoint weights")
    template_mode = "system-role"
    try:
        tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "test"}],
                                      tokenize=False, add_generation_prompt=True)
    except Exception:
        template_mode = "system-prepended-to-user"
    answer_token_ids = {}
    for label in (1, 2, 3):
        encoded = tokenizer.encode(str(label), add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"answer label {label} is not one token: {encoded}")
        answer_token_ids[label] = encoded[0]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model)
    out = ROOT / "results" / f"paired_order_generalization_{safe}.jsonl"
    existing = [json.loads(line) for line in out.read_text().splitlines()] if out.exists() else []
    if existing and ({row["model"] for row in existing} != {args.model} or
                     {row["model_revision"] for row in existing} != {args.revision}):
        raise RuntimeError("existing output has different provenance")
    complete = {row["id"] for row in existing if row.get("parse_status") != "error"}
    sampler = make_sampler(temp=0.0)
    for index, row in enumerate(rows):
        if row["id"] in complete:
            continue
        prompt, token_ids = formatted(tokenizer, row["user_prompt"], template_mode)
        started = time.time()
        error = None
        try:
            logits = model(mx.array([token_ids]))
            label_logits = {label: float(np.asarray(logits[0, -1, token].astype(mx.float32)))
                            for label, token in answer_token_ids.items()}
            constrained_prediction = max(label_logits, key=label_logits.get)
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=4,
                           sampler=sampler, verbose=False)
            prediction, parse_status = parse_answer(raw)
        except Exception as exc:
            raw, prediction, parse_status = "", None, "error"
            constrained_prediction, label_logits, error = None, {}, repr(exc)
        result = dict(row)
        result.update({
            "model": args.model,
            "model_revision": args.revision,
            "backend": "mlx-lm",
            "mlx_version": getattr(mlx, "__version__", "unknown"),
            "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
            "chat_template_mode": template_mode,
            "checkpoint_precision": "native-unquantized",
            "decoding": "greedy-temperature-0",
            "max_tokens": 4,
            "sequence_length": len(token_ids),
            "prompt_token_ids": token_ids,
            "answer_token_ids": answer_token_ids,
            "answer_label_logits": label_logits,
            "constrained_prediction": constrained_prediction,
            "raw_response": raw,
            "prediction": prediction,
            "parse_status": parse_status,
            "correct": prediction == row["answer"],
            "constrained_correct": constrained_prediction == row["answer"],
            "error": error,
            "elapsed_seconds": round(time.time() - started, 4),
        })
        with out.open("a") as file:
            file.write(json.dumps(result, sort_keys=True) + "\n")
        if (index + 1) % 25 == 0:
            print(f"{index + 1}/{len(rows)}", flush=True)
    print(out)


if __name__ == "__main__":
    main()
