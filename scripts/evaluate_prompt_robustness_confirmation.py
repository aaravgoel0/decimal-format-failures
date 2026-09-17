#!/usr/bin/env python3
"""Evaluate the frozen 10-template prompt confirmation on one exact checkpoint."""
import argparse
import json
import re
import time
from pathlib import Path

import mlx
import mlx.core as mx
import mlx_lm
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "You are a helpful assistant that compares numbers."


def formatted(tokenizer, user, mode):
    messages = ([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
                if mode == "system-role" else
                [{"role": "user", "content": SYSTEM + "\n\n" + user}])
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        return_tensors=None)
    return text, ids


def lcs_length(left, right):
    previous = [0] * (len(right) + 1)
    for a in left:
        current = [0]
        for index, b in enumerate(right, 1):
            current.append(previous[index - 1] + 1 if a == b else
                           max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def three_label_logits(model, token_batch, ordered_ids):
    """Return exact final logits for the three registered answer-token IDs only."""
    h = model.model.embed_tokens(mx.array(token_batch))
    if getattr(model, "model_type", "") == "gemma2":
        h = h * (model.args.hidden_size ** .5)
        mask = create_attention_mask(h, None, return_array=True)
    else:
        mask = create_attention_mask(h, None)
    for layer in model.model.layers:
        h = layer(h, mask, None)
    h = model.model.norm(h)[:, -1, :]
    index = mx.array(ordered_ids)
    if getattr(model.args, "tie_word_embeddings", True):
        weights = model.model.embed_tokens.weight[index]
        logits = h @ weights.T
    else:
        weights = model.lm_head.weight[index]
        logits = h @ weights.T
        bias = getattr(model.lm_head, "bias", None)
        if bias is not None:
            logits = logits + bias[index]
    if getattr(model, "model_type", "") == "gemma2":
        logits = mx.tanh(logits / model.final_logit_softcapping) * model.final_logit_softcapping
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    return np.array(logits, copy=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in
            (ROOT / "data/prompt_robustness_confirmation.jsonl").read_text().splitlines()]
    model, tokenizer = load(args.model, revision=args.revision)
    if getattr(model.args, "quantization", None) is not None:
        raise RuntimeError("the confirmation protocol forbids quantized weights")
    template_mode = "system-role"
    try:
        tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": "test"}],
                                      tokenize=False, add_generation_prompt=True)
    except Exception:
        template_mode = "system-prepended-to-user"
    answer_ids = {}
    for label in (1, 2, 3):
        encoded = tokenizer.encode(str(label), add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"answer label {label} is not one token: {encoded}")
        answer_ids[label] = encoded[0]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model)
    output_path = ROOT / "results" / f"prompt_robustness_confirmation_{safe}.jsonl"
    existing = ([json.loads(line) for line in output_path.read_text().splitlines()]
                if output_path.exists() else [])
    if existing and ({row["model"] for row in existing} != {args.model} or
                     {row["model_revision"] for row in existing} != {args.revision}):
        raise RuntimeError("existing output has different provenance")
    complete = {row["id"] for row in existing
                if row.get("error") is None and row.get("constrained_prediction") is not None}
    pending = []
    for row in rows:
        if row["id"] in complete:
            continue
        prompt, token_ids = formatted(tokenizer, row["user_prompt"], template_mode)
        canonical_ids = tokenizer.encode(row["canonical"], add_special_tokens=False)
        padded_ids = tokenizer.encode(row["padded"], add_special_tokens=False)
        overlap = lcs_length(canonical_ids, padded_ids)
        pending.append({
            "row": row, "prompt": prompt, "token_ids": token_ids,
            "canonical_ids": canonical_ids, "padded_ids": padded_ids,
            "overlap": overlap,
        })
    pending.sort(key=lambda item: (len(item["token_ids"]), item["row"]["id"]))
    completed_count = len(complete)
    cursor = 0
    while cursor < len(pending):
        first_length = len(pending[cursor]["token_ids"])
        end = cursor
        while (end < len(pending) and end - cursor < args.batch_size and
               len(pending[end]["token_ids"]) == first_length):
            end += 1
        batch = pending[cursor:end]
        cursor = end
        started = time.time()
        error = None
        try:
            ordered_ids = [answer_ids[label] for label in (1, 2, 3)]
            label_values = three_label_logits(
                model, [item["token_ids"] for item in batch], ordered_ids)
            mx.clear_cache()
        except Exception as exc:
            label_values, error = None, repr(exc)
        batch_elapsed = time.time() - started
        for batch_index, item in enumerate(batch):
            row = item["row"]
            if label_values is not None:
                label_logits = {label: float(label_values[batch_index, label - 1])
                                for label in (1, 2, 3)}
                constrained_prediction = max(label_logits, key=label_logits.get)
            else:
                constrained_prediction, label_logits = None, {}
            result = dict(row)
            result.update({
                "model": args.model,
                "model_revision": args.revision,
                "backend": "mlx-lm",
                "mlx_version": getattr(mlx, "__version__", "unknown"),
                "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
                "chat_template_mode": template_mode,
                "checkpoint_precision": "native-unquantized",
                "outcome_definition": "argmax-over-single-token-labels-1-2-3",
                "label_projection": "exact-selected-output-weights",
                "inference_batch_size": len(batch),
                "sequence_length": len(item["token_ids"]),
                "prompt_token_ids": item["token_ids"],
                "canonical_token_ids": item["canonical_ids"],
                "padded_token_ids": item["padded_ids"],
                "canonical_token_count": len(item["canonical_ids"]),
                "padded_token_count": len(item["padded_ids"]),
                "token_count_difference": len(item["padded_ids"]) - len(item["canonical_ids"]),
                "token_lcs_length": item["overlap"],
                "token_lcs_ratio": item["overlap"] /
                    max(len(item["canonical_ids"]), len(item["padded_ids"])),
                "answer_token_ids": answer_ids,
                "answer_label_logits": label_logits,
                "constrained_prediction": constrained_prediction,
                "constrained_correct": constrained_prediction == row["answer"],
                "free_text_generation": "not_run_by_operational_amendment",
                "error": error,
                "elapsed_seconds": round(batch_elapsed / len(batch), 4),
            })
            with output_path.open("a") as file:
                file.write(json.dumps(result, sort_keys=True) + "\n")
            completed_count += 1
            if completed_count % 25 == 0:
                print(f"{completed_count}/{len(rows)}", flush=True)
        mx.clear_cache()
    print(output_path)


if __name__ == "__main__":
    main()
