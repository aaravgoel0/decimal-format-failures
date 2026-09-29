#!/usr/bin/env python3
"""Freeze top four Qwen heads from discovery data only."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "results/qwen_component_head_discovery.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 30 or len({row["id"] for row in rows}) != 30:
        raise RuntimeError("discovery must contain 30 unique cases")
    scores = []
    for head in range(32):
        centered = []
        for row in rows:
            result = row["heads"][head]
            centered.append(result["aligned_effect"] -
                            sum(result["random_effects"]) / len(result["random_effects"]))
        scores.append({"head": head, "discovery_centered_mean": sum(centered) / len(centered)})
    scores.sort(key=lambda row: (-row["discovery_centered_mean"], row["head"]))
    result = {
        "selection_rule": "top four heads by discovery mean aligned effect minus case random mean",
        "selected_heads": [row["head"] for row in scores[:4]],
        "head_ranking": scores,
        "n_discovery_cases": 30,
        "heldout_outcomes_inspected": False,
    }
    output = ROOT / "results/qwen_component_head_selection.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output, result["selected_heads"])


if __name__ == "__main__":
    main()
