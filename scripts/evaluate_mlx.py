#!/usr/bin/env python3
"""Evaluate the exact Hugging Face checkpoint with MLX on Apple silicon."""

import argparse
import json
import re
import time
from pathlib import Path

from mlx_lm import generate, load

from evaluate import PROMPTS, parse_answer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Exact Hugging Face repository, including release tag")
    p.add_argument("--revision", help="Immutable Hugging Face commit SHA")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--prompt-variant", type=int, choices=range(3), default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model)
    out = args.results_dir / (
        f"{safe_model}__{args.dataset.stem}__p{args.prompt_variant}__mlx.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        for line in out.read_text().splitlines():
            previous = json.loads(line)
            if previous.get("parse_status") != "error":
                completed.add(previous["id"])

    model, tokenizer = load(args.model, revision=args.revision)
    system = "You are a helpful assistant that compares numbers."
    template_mode = "system-role"
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": "test"}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        template_mode = "system-prepended-to-user"
    for index, row in enumerate(rows):
        if row["id"] in completed:
            continue
        user = PROMPTS[args.prompt_variant].format(**row)
        if template_mode == "system-role":
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
        else:
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        started = time.time()
        error = None
        try:
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=16,
                           verbose=False)
            prediction, parse_status = parse_answer(raw)
        except Exception as exc:
            raw, prediction, parse_status, error = "", None, "error", repr(exc)
        result = dict(row)
        result.update({
            "model": args.model, "model_revision": args.revision,
            "backend": "mlx-lm", "chat_template_mode": template_mode,
            "prompt_variant": args.prompt_variant, "raw_response": raw,
            "prediction": prediction, "parse_status": parse_status,
            "correct": prediction == row["answer"], "error": error,
            "elapsed_seconds": round(time.time() - started, 4),
        })
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, sort_keys=True) + "\n")
        if (index + 1) % 25 == 0:
            print(f"{index + 1}/{len(rows)}", flush=True)
    print(out)


if __name__ == "__main__":
    main()
