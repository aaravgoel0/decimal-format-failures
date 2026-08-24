#!/usr/bin/env python3
"""Summarize official full-precision Llama prompt variants."""
import json, math
from pathlib import Path

def wilson(k, n, z=1.959963984540054):
    p = k / n; den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return center-radius, center+radius

def main():
    rows = []
    for variant in range(3):
        path = Path(f"results/meta-llama-Meta-Llama-3.1-8B-Instruct__confirmatory_zero_padding__p{variant}__mlx.jsonl")
        data = [json.loads(line) for line in path.read_text().splitlines()]
        if len(data) != 2000 or any(row["parse_status"] == "error" for row in data):
            raise RuntimeError(f"{path} is incomplete or contains execution errors")
        correct = sum(row["correct"] for row in data); low, high = wilson(correct, len(data))
        by_position = {}
        for position in (1, 2):
            subset = [row for row in data if row["padded_position"] == position]
            by_position[position] = sum(row["correct"] for row in subset) / len(subset)
        rows.append({"model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                     "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
                     "prompt_variant": variant, "correct": correct, "n": len(data),
                     "accuracy": correct/len(data), "ci_low": low, "ci_high": high,
                     "padded_first_accuracy": by_position[1],
                     "padded_second_accuracy": by_position[2]})
    Path("results/confirmatory_prompt_robustness.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows: print(row)

if __name__ == "__main__": main()
