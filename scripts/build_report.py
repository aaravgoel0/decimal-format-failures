#!/usr/bin/env python3
"""Build checked tables and figures from the raw experimental outputs."""

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

RUNS = {
    ("Llama 3.1 8B", "Integer", 0): RESULTS / "llama3.1-8b__integers__p0.jsonl",
    ("Llama 3.1 8B", "Misleading", 0): RESULTS / "llama3.1-8b__misleading__p0.jsonl",
    ("Llama 3.1 8B", "Zero-padding", 0): RESULTS / "llama3.1-8b__zero_padding__p0.jsonl",
    ("Llama 3.1 8B", "Zero-padding", 1): RESULTS / "llama3.1-8b__zero_padding__p1.jsonl",
    ("Llama 3.1 8B", "Zero-padding", 2): RESULTS / "llama3.1-8b__zero_padding__p2.jsonl",
    ("Qwen3 4B Instruct 2507", "Integer", 0): RESULTS / "Qwen-Qwen3-4B-Instruct-2507__integers__p0__mlx.jsonl",
    ("Qwen3 4B Instruct 2507", "Misleading", 0): RESULTS / "Qwen-Qwen3-4B-Instruct-2507__misleading__p0__mlx.jsonl",
    ("Qwen3 4B Instruct 2507", "Zero-padding", 0): RESULTS / "Qwen-Qwen3-4B-Instruct-2507__zero_padding__p0__mlx.jsonl",
    ("Gemma 2 9B", "Integer", 0): RESULTS / "gemma2-9b__integers__p0.jsonl",
    ("Gemma 2 9B", "Misleading", 0): RESULTS / "gemma2-9b__misleading__p0.jsonl",
    ("Gemma 2 9B", "Zero-padding", 0): RESULTS / "gemma2-9b__zero_padding__p0.jsonl",
    ("Llama 3.1 8B full precision", "Integer", 0): RESULTS / "meta-llama-Meta-Llama-3.1-8B-Instruct__integers__p0__mlx.jsonl",
    ("Llama 3.1 8B full precision", "Misleading", 0): RESULTS / "meta-llama-Meta-Llama-3.1-8B-Instruct__misleading__p0__mlx.jsonl",
    ("Llama 3.1 8B full precision", "Zero-padding held-out", 0): RESULTS / "meta-llama-Meta-Llama-3.1-8B-Instruct__confirmatory_zero_padding__p0__mlx.jsonl",
    ("Llama 3.1 8B full precision", "Zero-padding held-out", 1): RESULTS / "meta-llama-Meta-Llama-3.1-8B-Instruct__confirmatory_zero_padding__p1__mlx.jsonl",
    ("Llama 3.1 8B full precision", "Zero-padding held-out", 2): RESULTS / "meta-llama-Meta-Llama-3.1-8B-Instruct__confirmatory_zero_padding__p2__mlx.jsonl",
    ("Qwen3 4B Instruct 2507", "Zero-padding held-out", 0): RESULTS / "Qwen-Qwen3-4B-Instruct-2507__confirmatory_zero_padding__p0__mlx.jsonl",
    ("Gemma 2 9B full precision", "Integer", 0): RESULTS / "google-gemma-2-9b-it__integers__p0__mlx.jsonl",
    ("Gemma 2 9B full precision", "Misleading", 0): RESULTS / "google-gemma-2-9b-it__misleading__p0__mlx.jsonl",
    ("Gemma 2 9B full precision", "Zero-padding held-out", 0): RESULTS / "google-gemma-2-9b-it__confirmatory_zero_padding__p0__mlx.jsonl",
}


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - radius, center + radius


def load_run(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = 2000 if "confirmatory_zero_padding" in path.name else 1000
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} rows in {path}, found {len(rows)}")
    if any(r.get("parse_status") == "error" for r in rows):
        raise ValueError(f"Execution error present in {path}")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids in {path}")
    return rows


def summarize(model, task, prompt, path):
    rows = load_run(path)
    k = sum(r["correct"] for r in rows)
    lo, hi = wilson(k, len(rows))
    first = [r for r in rows if r["answer"] == 1]
    second = [r for r in rows if r["answer"] == 2]
    padded_first = [r for r in rows if r.get("padded_position") == 1]
    padded_second = [r for r in rows if r.get("padded_position") == 2]
    return {
        "model": model, "task": task, "prompt_variant": prompt,
        "n": len(rows), "correct": k, "accuracy": k / len(rows),
        "ci_low": lo, "ci_high": hi,
        "option_1_rate": sum(r["prediction"] == 1 for r in rows) / len(rows),
        "option_2_rate": sum(r["prediction"] == 2 for r in rows) / len(rows),
        "option_3_rate": sum(r["prediction"] == 3 for r in rows) / len(rows),
        "correct_when_answer_1": (sum(r["correct"] for r in first) / len(first)) if first else None,
        "correct_when_answer_2": (sum(r["correct"] for r in second) / len(second)) if second else None,
        "correct_padded_first": (sum(r["correct"] for r in padded_first) / len(padded_first)) if padded_first else None,
        "correct_padded_second": (sum(r["correct"] for r in padded_second) / len(padded_second)) if padded_second else None,
        "parse_counts": dict(Counter(r["parse_status"] for r in rows)),
        "source": str(path.relative_to(ROOT)),
    }


def make_cross_model_plot(rows):
    models = ["Llama 3.1 8B full precision", "Qwen3 4B Instruct 2507", "Gemma 2 9B full precision"]
    model_labels = ["Llama 3.1 8B", "Qwen3 4B", "Gemma 2 9B"]
    tasks = ["Integer", "Misleading", "Zero-padding held-out"]
    task_labels = ["Integer (n=1,000)", "Misleading decimal (n=1,000)",
                   "Equal zero-padding (n=2,000)"]
    colors = ["#4062BB", "#59A14F", "#E15759"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.24
    x = list(range(len(models)))
    for j, (task, task_label) in enumerate(zip(tasks, task_labels)):
        vals = [next(r for r in rows if r["model"] == m and r["task"] == task and r["prompt_variant"] == 0) for m in models]
        xs = [v + (j - 1) * width for v in x]
        ys = [100 * r["accuracy"] for r in vals]
        errors = [[100 * (r["accuracy"] - r["ci_low"]) for r in vals],
                  [100 * (r["ci_high"] - r["accuracy"]) for r in vals]]
        bars = ax.bar(xs, ys, width, label=task_label, color=colors[j], yerr=errors,
                      capsize=3)
        ax.bar_label(bars, labels=[f"{y:.1f}" for y in ys], padding=3, fontsize=9)
    ax.set_xticks(x, model_labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Official checkpoints show a large three-model gap", pad=46)
    ax.legend(frameon=False, ncols=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.01))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "cross_model_accuracy.png", dpi=200)
    plt.close(fig)


def make_prompt_plot(rows):
    vals = [next(r for r in rows if r["model"] == "Llama 3.1 8B full precision" and
                 r["task"] == "Zero-padding held-out" and r["prompt_variant"] == p)
            for p in range(3)]
    labels = ["Primary prompt", "Compare values", "Select statement"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ys = [100 * r["accuracy"] for r in vals]
    bars = ax.bar(labels, ys, color=["#4062BB", "#F28E2B", "#B07AA1"])
    ax.bar_label(bars, labels=[f"{y:.1f}%" for y in ys], padding=4, fontsize=11)
    ax.set_ylim(0, 66)
    ax.set_ylabel("Accuracy on equal zero-padded pairs")
    ax.set_title("Full-precision Llama changes sharply with prompt wording")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "llama_prompt_sensitivity.png", dpi=200)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    summaries = [summarize(*key, path) for key, path in RUNS.items()]
    (RESULTS / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    make_cross_model_plot(summaries)
    make_prompt_plot(summaries)
    print(RESULTS / "summary.json")
    print(FIGURES / "cross_model_accuracy.png")
    print(FIGURES / "llama_prompt_sensitivity.png")


if __name__ == "__main__":
    main()
