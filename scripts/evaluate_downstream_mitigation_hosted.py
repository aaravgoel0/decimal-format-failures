#!/usr/bin/env python3
"""Run the frozen hosted 72B scale control through Hugging Face routing."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import InferenceClient, __version__ as hf_hub_version

from evaluate_numeric_invariance import SYSTEM, build_user, numerically_equal, parse_prediction


ROOT = Path(__file__).resolve().parents[1]
PILOTS = (
    "Reply with only the number: 2 + 3.",
    "Reply with only the number: 7 - 4.",
    "Reply with only the number: 6 times 5.",
    "Reply with only the number: half of 18.",
    "Reply with only the number: 12 divided by 3.",
    "Reply with only the number: 9 + 8.",
    "Reply with only the number: 15 - 6.",
    "Reply with only the number: 4 times 7.",
    "Reply with only the number: one quarter of 20.",
    "Reply with only the number: 21 divided by 7.",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def completion(client, model, messages, max_tokens):
    last_error = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            usage = getattr(response, "usage", None)
            return {
                "raw_response": response.choices[0].message.content or "",
                "response_model": getattr(response, "model", None),
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                    "estimated_cost": getattr(usage, "estimated_cost", None),
                },
                "error": None,
                "attempts": attempt + 1,
            }
        except Exception as exc:
            last_error = repr(exc)
            if attempt < 2:
                time.sleep(2 ** attempt)
    return {
        "raw_response": "",
        "response_model": None,
        "usage": {},
        "error": last_error,
        "attempts": 3,
    }


def run_gate(client, args):
    records = []
    for index, prompt in enumerate(PILOTS):
        started = time.time()
        result = completion(
            client,
            args.model,
            [{"role": "user", "content": prompt}],
            max_tokens=16,
        )
        records.append({
            "pilot_index": index,
            "prompt": prompt,
            "requested_at": utc_now(),
            "elapsed_seconds": round(time.time() - started, 4),
            **result,
        })
    repeated = []
    for repeat_index in range(3):
        result = completion(
            client,
            args.model,
            [{"role": "user", "content": PILOTS[0]}],
            max_tokens=16,
        )
        repeated.append({"repeat_index": repeat_index, **result})
    passed = (
        all(record["error"] is None for record in records)
        and all(record["error"] is None for record in repeated)
        and len({record["raw_response"] for record in repeated}) == 1
    )
    gate = {
        "model": args.model,
        "public_repository_revision": args.revision,
        "provider": args.provider,
        "backend": "huggingface-inference-providers",
        "huggingface_hub_version": hf_hub_version,
        "served_revision_verifiable": False,
        "serving_note": "The provider did not attest the immutable weight revision behind the endpoint.",
        "decoding": "temperature-zero provider decoding",
        "completed_at": utc_now(),
        "passed": passed,
        "pilots": records,
        "determinism_repeats": repeated,
    }
    args.pilot_output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("hosted-model pilot gate failed")
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--provider", default="novita")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/downstream_mitigation_confirmation.jsonl",
    )
    parser.add_argument(
        "--pilot-output",
        type=Path,
        default=ROOT / "results/downstream_mitigation_hosted_pilot.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/downstream_mitigation_Qwen-Qwen2.5-72B-Instruct.jsonl",
    )
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()

    client = InferenceClient(provider=args.provider, token=True)
    if args.pilot_output.exists():
        gate = json.loads(args.pilot_output.read_text())
        identity = (gate.get("model"), gate.get("public_repository_revision"), gate.get("provider"))
        if identity != (args.model, args.revision, args.provider) or not gate.get("passed"):
            raise RuntimeError("existing pilot gate does not match or did not pass")
    else:
        gate = run_gate(client, args)
    if args.pilot_only:
        print(args.pilot_output)
        return

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    existing = [json.loads(line) for line in args.output.read_text().splitlines()] if args.output.exists() else []
    if existing and (
        {row["model"] for row in existing} != {args.model}
        or {row["public_repository_revision"] for row in existing} != {args.revision}
        or {row["provider"] for row in existing} != {args.provider}
    ):
        raise RuntimeError("existing hosted output has different provenance")
    complete = {row["id"] for row in existing if row.get("error") is None}

    for index, row in enumerate(rows):
        if row["id"] in complete:
            continue
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user(row)},
        ]
        started = time.time()
        result = completion(client, args.model, messages, args.max_tokens)
        raw = result["raw_response"]
        if result["error"] is None:
            value, is_percent, parse_status = parse_prediction(raw)
        else:
            value, is_percent, parse_status = None, None, "error"
        correct = (
            parse_status == "ok"
            and numerically_equal(value, row["answer_value"])
            and bool(is_percent) == bool(row["answer_is_percent"])
        )
        record = dict(row)
        record.update({
            "model": args.model,
            "public_repository_revision": args.revision,
            "provider": args.provider,
            "backend": "huggingface-inference-providers",
            "served_revision_verifiable": False,
            "decoding": "temperature-zero provider decoding",
            "max_tokens": args.max_tokens,
            "requested_at": utc_now(),
            "elapsed_seconds": round(time.time() - started, 4),
            "raw_response": raw,
            "prediction_value": value,
            "prediction_is_percent": is_percent,
            "parse_status": parse_status,
            "correct": correct,
            **{key: result[key] for key in ("response_model", "usage", "error", "attempts")},
        })
        with args.output.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
        if (index + 1) % 25 == 0:
            print(f"{index + 1}/{len(rows)}", flush=True)

    attempts = [json.loads(line) for line in args.output.read_text().splitlines()]
    final = {}
    for row in attempts:
        previous = final.get(row["id"])
        if previous is None or (previous.get("error") is not None and row.get("error") is None):
            final[row["id"]] = row
    failures = [row for row in final.values() if row.get("error") is not None]
    if len(final) != len(rows) or failures:
        raise RuntimeError(f"incomplete hosted evaluation: {len(final)}/{len(rows)}, {len(failures)} errors")
    order = {row["id"]: index for index, row in enumerate(rows)}
    ordered = sorted(final.values(), key=lambda row: order[row["id"]])
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered))
    print(args.output)


if __name__ == "__main__":
    main()
