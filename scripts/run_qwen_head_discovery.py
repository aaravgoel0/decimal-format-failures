#!/usr/bin/env python3
"""Discovery-only sweep of all Qwen layer-2 attention heads."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from qwen_component_utils import (LAYER, MODEL, REVISION, intervened_output,
                                  prepare_case, selected_logits,
                                  unique_random_sets)


ROOT = Path(__file__).resolve().parents[1]
RANDOM_SETS = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / "data/qwen_component_causal.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["split"] == "discovery"]
    model, tokenizer = load(MODEL, revision=REVISION)
    if getattr(model.args, "quantization", None) is not None:
        raise RuntimeError("native unquantized weights required")
    template_mode = "system-role"
    try:
        tokenizer.apply_chat_template([{"role": "system", "content": "test"},
                                       {"role": "user", "content": "test"}],
                                      tokenize=False, add_generation_prompt=True)
    except Exception:
        template_mode = "system-prepended-to-user"
    ordered_ids = []
    for label in (1, 2, 3):
        ids = tokenizer.encode(str(label), add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"label {label} is not one token")
        ordered_ids.append(ids[0])
    output = ROOT / "results/qwen_component_head_discovery.jsonl"
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    complete = {row["id"] for row in existing}
    for row in rows:
        if row["id"] in complete:
            continue
        case = prepare_case(model, tokenizer, row, template_mode)
        random_sets = unique_random_sets(case["eligible"], len(case["mappings"]),
                                         2026091710 + row["case"], RANDOM_SETS)
        random_mappings = [[(position, position) for position in positions]
                           for positions in random_sets]
        baseline, easy_margin = selected_logits(
            model, [case["hard"]["block_output"], case["easy"]["block_output"]],
            case["mask"], ordered_ids, args.batch_size)
        outputs = []
        for head in range(model.args.num_attention_heads):
            outputs.append(intervened_output(case["layer"], case["hard"], case["easy"],
                                              case["mappings"], "heads", [head]))
            for mappings in random_mappings:
                outputs.append(intervened_output(case["layer"], case["hard"], case["easy"],
                                                  mappings, "heads", [head]))
        margins = selected_logits(model, outputs, case["mask"], ordered_ids, args.batch_size)
        heads = []
        offset = 0
        for head in range(model.args.num_attention_heads):
            aligned = margins[offset]
            random_values = margins[offset + 1:offset + 1 + RANDOM_SETS]
            offset += 1 + RANDOM_SETS
            heads.append({
                "head": head,
                "aligned_margin": aligned,
                "aligned_effect": aligned - baseline,
                "random_margins": random_values,
                "random_effects": [value - baseline for value in random_values],
            })
        result = dict(row)
        result.update({
            "model": MODEL,
            "revision": REVISION,
            "layer": LAYER,
            "checkpoint_precision": "native-unquantized",
            "template_mode": template_mode,
            "hard_margin": baseline,
            "easy_margin": easy_margin,
            "sequence_length": len(case["hard_ids"]),
            "aligned_target_positions": sorted(target for target, _ in case["mappings"]),
            "aligned_source_positions": [source for _, source in case["mappings"]],
            "random_position_sets": random_sets,
            "heads": heads,
        })
        with output.open("a") as file:
            file.write(json.dumps(result, sort_keys=True) + "\n")
        mx.clear_cache()
        print(f"{len(complete) + 1}/{len(rows)}", flush=True)
        complete.add(row["id"])
    print(output)


if __name__ == "__main__":
    main()
