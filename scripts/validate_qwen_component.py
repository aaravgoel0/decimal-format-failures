#!/usr/bin/env python3
"""Validate token alignment and manual Qwen layer reconstruction before outcomes."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

from qwen_component_utils import (MODEL, REVISION, prepare_case,
                                  unique_random_sets)


ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = [json.loads(line) for line in
            (ROOT / "data/qwen_component_causal.jsonl").read_text().splitlines()]
    model, tokenizer = load(MODEL, revision=REVISION)
    if getattr(model.args, "quantization", None) is not None:
        raise RuntimeError("native unquantized weights required")
    template_mode = "system-role"
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"},
             {"role": "user", "content": "test"}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        template_mode = "system-prepended-to-user"
    for label in (1, 2, 3):
        if len(tokenizer.encode(str(label), add_special_tokens=False)) != 1:
            raise RuntimeError(f"label {label} is not one token")

    maximum_difference = 0.0
    for row in rows:
        case = prepare_case(model, tokenizer, row, template_mode)
        for side in ("easy", "hard"):
            native = case["layer"](case[side]["input"], case["mask"], None)
            difference = float(np.max(np.abs(np.asarray(
                (native - case[side]["block_output"]).astype(mx.float32)))))
            maximum_difference = max(maximum_difference, difference)
        if len({target for target, _ in case["mappings"]}) != len(case["mappings"]):
            raise RuntimeError(f"duplicate target mapping: {row['id']}")
        count = 5 if row["split"] == "discovery" else 20
        unique_random_sets(case["eligible"], len(case["mappings"]),
                           2026091710 + row["case"] if row["split"] == "discovery"
                           else 2026091750 + row["case"], count)
        mx.clear_cache()
    if maximum_difference > 1e-5:
        raise RuntimeError(f"manual layer reconstruction differs by {maximum_difference}")
    print(json.dumps({
        "rows_validated": len(rows),
        "template_mode": template_mode,
        "maximum_block_output_absolute_difference": maximum_difference,
        "status": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
