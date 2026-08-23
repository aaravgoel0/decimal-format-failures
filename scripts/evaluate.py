#!/usr/bin/env python3
"""Evaluate a JSONL comparison dataset against a local Ollama model."""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

PROMPTS = (
    "Which is bigger?\nOptions:\n1. {a}\n2. {b}\n3. Both are equal\n"
    "Answer with only the option number (1, 2, or 3). Do not write anything else.",
    "Compare these two numerical values.\n1. {a}\n2. {b}\n3. They are equal\n"
    "Return exactly one character: 1, 2, or 3.",
    "Select the mathematically correct statement about {a} and {b}:\n"
    "1. The first is larger\n2. The second is larger\n3. The values are equal\n"
    "Reply only with 1, 2, or 3.",
)


def ollama_chat(model, system, user, seed, timeout):
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"temperature": 0, "seed": seed, "num_predict": 16},
        "keep_alive": "30m",
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def parse_answer(text):
    stripped = text.strip()
    if stripped in {"1", "2", "3"}:
        return int(stripped), "exact"
    match = re.search(r"(?<!\d)([123])(?!\d)", stripped)
    return (int(match.group(1)), "recovered") if match else (None, "invalid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--prompt-variant", type=int, choices=range(3), default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--seed", type=int, default=7302026)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    args = p.parse_args()

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model)
    out = args.results_dir / f"{safe_model}__{args.dataset.stem}__p{args.prompt_variant}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if out.exists():
        for line in out.read_text().splitlines():
            completed.add(json.loads(line)["id"])

    # Variant 0 reproduces the system prompt printed in the paper. Keep this
    # neutral for all variants so prompt-robustness runs change only user text.
    system = "You are a helpful assistant that compares numbers."
    for index, row in enumerate(rows):
        if row["id"] in completed:
            continue
        user = PROMPTS[args.prompt_variant].format(**row)
        started = time.time()
        try:
            response = ollama_chat(args.model, system, user, args.seed + index,
                                   args.timeout)
            raw = response["message"]["content"]
            prediction, parse_status = parse_answer(raw)
            error = None
        except Exception as exc:
            raw, prediction, parse_status, error = "", None, "error", repr(exc)
        result = dict(row)
        result.update({
            "model": args.model, "prompt_variant": args.prompt_variant,
            "seed": args.seed + index, "raw_response": raw,
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
