#!/usr/bin/env python3
"""Run the frozen hosted Gemma 3 27B reasoning confirmation."""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal, InvalidOperation
from pathlib import Path

from huggingface_hub import InferenceClient


ROOT = Path(__file__).resolve().parents[1]
MODEL = "google/gemma-3-27b-it"
PROVIDER = "deepinfra"
SEED = 20261004
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
    return SYSTEM + f"\n\nSolve the following problem. {unit}\n\n{row['problem']}"


def one_request(row, max_tokens):
    started = time.time()
    error = None
    response = None
    for attempt in range(1, 4):
        try:
            client = InferenceClient(provider=PROVIDER)
            response = client.chat_completion(
                model=MODEL,
                messages=[{"role": "user", "content": build_user(row)}],
                max_tokens=max_tokens,
                temperature=0.0,
                seed=SEED,
            )
            error = None
            break
        except Exception as exc:
            error = repr(exc)
            if attempt < 3:
                time.sleep(5 * attempt)
    raw = "" if response is None else (response.choices[0].message.content or "")
    value, percent, parse_status = (
        (None, None, "error") if response is None else parse_final(raw)
    )
    correct = (
        parse_status == "ok"
        and numerically_equal(value, row["answer_value"])
        and bool(percent) == bool(row["answer_is_percent"])
    )
    usage = None
    finish_reason = None
    if response is not None:
        finish_reason = response.choices[0].finish_reason
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
    result = dict(row)
    result.update({
        "model": MODEL,
        "provider": PROVIDER,
        "provider_revision_pin_available": False,
        "backend": "huggingface-inference-providers",
        "chat_template_mode": "instruction-prepended-to-user",
        "decoding": "temperature-0",
        "seed_requested": SEED,
        "max_tokens": max_tokens,
        "raw_response": raw,
        "prediction_value": value,
        "prediction_is_percent": percent,
        "parse_status": parse_status,
        "correct": correct,
        "finish_reason": finish_reason,
        "usage": usage,
        "error": error,
        "elapsed_seconds": round(time.time() - started, 4),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise RuntimeError("frozen hosted confirmation permits one to four workers")

    dataset = ROOT / "data/downstream_mitigation_confirmation.jsonl"
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    if args.limit is not None:
        rows = rows[:args.limit]
    output = ROOT / "results/reasoning_confirmation_google-gemma-3-27b-it_deepinfra.jsonl"
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if existing and (
        {row["model"] for row in existing} != {MODEL}
        or {row["provider"] for row in existing} != {PROVIDER}
    ):
        raise RuntimeError("existing output has different hosted provenance")
    complete = {row["id"] for row in existing if row.get("error") is None}
    pending = [row for row in rows if row["id"] not in complete]
    print(f"model={MODEL} provider={PROVIDER} complete={len(complete)} pending={len(pending)}", flush=True)

    finished = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(one_request, row, args.max_tokens): row for row in pending}
        for future in as_completed(futures):
            result = future.result()
            with output.open("a", encoding="utf-8") as file:
                file.write(json.dumps(result, sort_keys=True) + "\n")
            finished += 1
            if finished % args.progress_every == 0 or finished == len(pending):
                print(f"{len(complete) + finished}/{len(rows)}", flush=True)

    attempts = [json.loads(line) for line in output.read_text().splitlines()]
    final = {}
    for row in attempts:
        previous = final.get(row["id"])
        if previous is None or (previous.get("error") is not None and row.get("error") is None):
            final[row["id"]] = row
    target_ids = {row["id"] for row in rows}
    selected = [row for key, row in final.items() if key in target_ids]
    if len(selected) == len(rows) and not any(row.get("error") is not None for row in selected):
        order = {row["id"]: index for index, row in enumerate(rows)}
        selected.sort(key=lambda row: order[row["id"]])
        if args.limit is None:
            output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
            print(output)
    elif args.limit is None:
        failures = sum(row.get("error") is not None for row in selected)
        raise RuntimeError(
            f"incomplete hosted confirmation: {len(selected)}/{len(rows)}, failures={failures}"
        )


if __name__ == "__main__":
    main()
