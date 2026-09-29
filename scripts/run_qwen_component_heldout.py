#!/usr/bin/env python3
"""Held-out attention, MLP, block, and selected-head Qwen interventions."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from qwen_component_utils import (LAYER, MODEL, REVISION, intervened_output,
                                  prepare_case, selected_logits,
                                  unique_random_sets)


ROOT = Path(__file__).resolve().parents[1]
RANDOM_SETS = 20
INTERVENTIONS = ("selected_heads", "attention", "mlp", "whole_block")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / "data/qwen_component_causal.jsonl").read_text().splitlines()]
    rows = [row for row in rows if row["split"] == "heldout"]
    selection = json.loads((ROOT / "results/qwen_component_head_selection.json").read_text())
    selected_heads = selection["selected_heads"]
    if len(selected_heads) != 4 or len(set(selected_heads)) != 4:
        raise RuntimeError("selection must contain four unique heads")
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
    output = ROOT / "results/qwen_component_heldout.jsonl"
    existing = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if existing and {tuple(row["selected_heads"]) for row in existing} != {tuple(selected_heads)}:
        raise RuntimeError("held-out file uses different selected heads")
    complete = {row["id"] for row in existing}
    for row in rows:
        if row["id"] in complete:
            continue
        case = prepare_case(model, tokenizer, row, template_mode)
        random_sets = unique_random_sets(case["eligible"], len(case["mappings"]),
                                         2026091750 + row["case"], RANDOM_SETS)
        random_mappings = [[(position, position) for position in positions]
                           for positions in random_sets]
        baseline, easy_margin = selected_logits(
            model, [case["hard"]["block_output"], case["easy"]["block_output"]],
            case["mask"], ordered_ids, args.batch_size)
        outputs = []
        for intervention in INTERVENTIONS:
            kind = "heads" if intervention == "selected_heads" else intervention
            heads = selected_heads if intervention == "selected_heads" else None
            outputs.append(intervened_output(case["layer"], case["hard"], case["easy"],
                                              case["mappings"], kind, heads))
            for mappings in random_mappings:
                outputs.append(intervened_output(case["layer"], case["hard"], case["easy"],
                                                  mappings, kind, heads))
        margins = selected_logits(model, outputs, case["mask"], ordered_ids, args.batch_size)
        interventions = {}
        offset = 0
        for intervention in INTERVENTIONS:
            aligned = margins[offset]
            random_values = margins[offset + 1:offset + 1 + RANDOM_SETS]
            offset += 1 + RANDOM_SETS
            interventions[intervention] = {
                "aligned_margin": aligned,
                "aligned_effect": aligned - baseline,
                "aligned_flip": baseline <= 0 < aligned,
                "random_margins": random_values,
                "random_effects": [value - baseline for value in random_values],
                "random_flips": [baseline <= 0 < value for value in random_values],
            }
        result = dict(row)
        result.update({
            "model": MODEL,
            "revision": REVISION,
            "layer": LAYER,
            "checkpoint_precision": "native-unquantized",
            "template_mode": template_mode,
            "selected_heads": selected_heads,
            "hard_margin": baseline,
            "easy_margin": easy_margin,
            "hard_correct": baseline > 0,
            "easy_correct": easy_margin > 0,
            "sequence_length": len(case["hard_ids"]),
            "aligned_target_positions": sorted(target for target, _ in case["mappings"]),
            "aligned_source_positions": [source for _, source in case["mappings"]],
            "random_position_sets": random_sets,
            "interventions": interventions,
        })
        with output.open("a") as file:
            file.write(json.dumps(result, sort_keys=True) + "\n")
        mx.clear_cache()
        complete.add(row["id"])
        print(f"{len(complete)}/{len(rows)}", flush=True)
    print(output)


if __name__ == "__main__":
    main()
