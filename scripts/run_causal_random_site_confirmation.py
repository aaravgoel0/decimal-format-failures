#!/usr/bin/env python3
"""Fixed-layer causal confirmation against 50 random-site sets per component."""
import argparse
import json
import math
import random
import re
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from run_causal_generalization import (SYSTEM, formatted_ids, hits,
                                       mapping, replace, state_at)

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("short", "padded", "joint")
N_RANDOM = 50


def final_three_label_logits(model, h, mask, selected_layer, ordered_ids):
    """Continue from a patched state and project exactly onto labels 1, 2, and 3."""
    for layer in model.model.layers[selected_layer + 1:]:
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
    return logits


def margin_values(logits):
    correct = logits[:, 2]
    alternative = mx.maximum(logits[:, 0], logits[:, 1])
    mx.eval(correct, alternative)
    return np.asarray((correct - alternative).astype(mx.float32), dtype=float)


def unique_random_sets(eligible, size, seed, count=N_RANDOM):
    rng = random.Random(seed)
    maximum = math.comb(len(eligible), size)
    if maximum < count:
        raise RuntimeError(f"only {maximum} random subsets available")
    seen = set()
    while len(seen) < count:
        seen.add(tuple(sorted(rng.sample(eligible, size))))
    return sorted(seen)


def batched_margins(model, hard_state, source_state, hard_mask, layer, mappings, ordered_ids,
                    batch_size):
    values = []
    for start in range(0, len(mappings), batch_size):
        batch_maps = mappings[start:start + batch_size]
        patched = mx.concatenate([replace(hard_state, source_state, item)
                                  for item in batch_maps], axis=0)
        logits = final_three_label_logits(model, patched, hard_mask, layer, ordered_ids)
        values.extend(margin_values(logits))
        del patched
        mx.clear_cache()
    return [float(value) for value in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    cases = [json.loads(line) for line in
             (ROOT / "data/causal_random_site_confirmation.jsonl").read_text().splitlines()]
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
        ids = tokenizer.encode(str(label), add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"answer label {label} is not one token: {ids}")
        answer_ids[label] = ids[0]
    ordered_ids = [answer_ids[label] for label in (1, 2, 3)]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.model)
    output_path = ROOT / "results" / f"causal_random_site_confirmation_{safe}.jsonl"
    existing = ([json.loads(line) for line in output_path.read_text().splitlines()]
                if output_path.exists() else [])
    if existing and ({row["model"] for row in existing} != {args.model} or
                     {row["revision"] for row in existing} != {args.revision} or
                     {row["layer"] for row in existing} != {args.layer}):
        raise RuntimeError("existing output has different provenance")
    complete = {row["id"] for row in existing if row.get("status") == "complete"}
    cases.sort(key=lambda row: (
        len(formatted_ids(tokenizer, row["hard_prompt"], template_mode)), row["id"]))
    completed_count = len(complete)
    for row in cases:
        if row["id"] in complete:
            continue
        easy_ids = formatted_ids(tokenizer, row["easy_prompt"], template_mode)
        hard_ids = formatted_ids(tokenizer, row["hard_prompt"], template_mode)
        if len(easy_ids) != len(hard_ids):
            raise RuntimeError(f"unaligned sequence length in {row['id']}")
        short_ids = tokenizer.encode(row["short"], add_special_tokens=False)
        padded_ids = tokenizer.encode(row["padded"], add_special_tokens=False)
        easy_short = min(hits(easy_ids, short_ids))
        easy_padded = max(hits(easy_ids, padded_ids))
        hard_padded = min(hits(hard_ids, padded_ids))
        hard_short = max(hits(hard_ids, short_ids))
        maps = {
            "short": mapping(hard_short, easy_short, len(short_ids)),
            "padded": mapping(hard_padded, easy_padded, len(padded_ids)),
        }
        maps["joint"] = maps["short"] + maps["padded"]
        excluded = {len(hard_ids) - 1}
        for start, count in ((easy_short, len(short_ids)), (easy_padded, len(padded_ids)),
                             (hard_short, len(short_ids)), (hard_padded, len(padded_ids))):
            excluded.update(range(start, start + count))
        eligible = [position for position in range(1, len(hard_ids) - 1)
                    if position not in excluded]
        easy_state, _ = state_at(model, easy_ids, args.layer)
        hard_state, hard_mask = state_at(model, hard_ids, args.layer)
        baseline_logits = final_three_label_logits(
            model, mx.concatenate([hard_state, easy_state], axis=0), hard_mask,
            args.layer, ordered_ids)
        baseline, easy_margin = [float(value) for value in margin_values(baseline_logits)]
        del baseline_logits
        interventions = {}
        component_inputs = {}
        all_maps = []
        for component_index, component in enumerate(COMPONENTS):
            random_sets = unique_random_sets(eligible, len(maps[component]),
                                             2026091300 + row["case"] * 10 + component_index)
            random_maps = [[(position, position) for position in positions]
                           for positions in random_sets]
            component_inputs[component] = (random_sets, random_maps)
            all_maps.extend([maps[component], *random_maps])
        all_margins = batched_margins(model, hard_state, easy_state, hard_mask,
                                      args.layer, all_maps, ordered_ids, args.batch_size)
        offset = 0
        for component in COMPONENTS:
            random_sets, _ = component_inputs[component]
            aligned_margin = all_margins[offset]
            random_margins = all_margins[offset + 1:offset + 1 + N_RANDOM]
            offset += 1 + N_RANDOM
            interventions[component] = {
                "aligned_target_positions": sorted(target for target, _ in maps[component]),
                "aligned_source_positions": [source for _, source in maps[component]],
                "aligned_margin": aligned_margin,
                "aligned_effect": aligned_margin - baseline,
                "aligned_flip": baseline <= 0 < aligned_margin,
                "random_position_sets": random_sets,
                "random_margins": random_margins,
                "random_effects": [value - baseline for value in random_margins],
                "random_flips": [baseline <= 0 < value for value in random_margins],
            }
        if offset != len(all_margins):
            raise RuntimeError("intervention result count mismatch")
        result = dict(row)
        result.update({
            "status": "complete",
            "model": args.model,
            "revision": args.revision,
            "layer": args.layer,
            "template_mode": template_mode,
            "checkpoint_precision": "native-unquantized",
            "label_projection": "exact-selected-output-weights",
            "sequence_length": len(hard_ids),
            "hard_prompt_token_ids": hard_ids,
            "easy_prompt_token_ids": easy_ids,
            "short_token_ids": short_ids,
            "padded_token_ids": padded_ids,
            "eligible_random_positions": eligible,
            "n_random_sets_per_component": N_RANDOM,
            "hard_margin": baseline,
            "easy_margin": easy_margin,
            "hard_correct": baseline > 0,
            "easy_correct": easy_margin > 0,
            "interventions": interventions,
        })
        with output_path.open("a") as file:
            file.write(json.dumps(result, sort_keys=True) + "\n")
        completed_count += 1
        del easy_state, hard_state
        mx.clear_cache()
        print(f"{completed_count}/{len(cases)}", flush=True)
    final_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    if len(final_rows) == len(cases) and len({row["id"] for row in final_rows}) == len(cases):
        final_rows.sort(key=lambda row: row["case"])
        output_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                       for row in final_rows))
    print(output_path)


if __name__ == "__main__":
    main()
