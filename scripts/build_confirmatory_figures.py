#!/usr/bin/env python3
"""Build held-out figures from the three official checkpoint runs."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
payload = json.loads((ROOT / "results" / "summary.json").read_text())
models = ["Llama 3.1 8B full precision", "Qwen3 4B Instruct 2507",
          "Gemma 2 9B full precision"]
data = [next(row for row in payload if row["model"] == model and
             row["task"] == "Zero-padding held-out" and
             row["prompt_variant"] == 0) for model in models]
figures = ROOT / "figures"
figures.mkdir(exist_ok=True)

short = {"Llama 3.1 8B full precision": "Llama 3.1 8B",
         "Qwen3 4B Instruct 2507": "Qwen3 4B Instruct 2507",
         "Gemma 2 9B full precision": "Gemma 2 9B"}
labels = [short[x["model"]] for x in data]
colors = ["#4062BB", "#59A14F", "#E15759"]

fig, ax = plt.subplots(figsize=(9, 5.2))
y = [row["accuracy"] * 100 for row in data]
err = [[(row["accuracy"] - row["ci_low"]) * 100 for row in data],
       [(row["ci_high"] - row["accuracy"]) * 100 for row in data]]
bars = ax.bar(labels, y, color=colors, yerr=err, capsize=4)
ax.bar_label(bars, labels=[f"{x:.1f}%" for x in y], padding=4, fontsize=11)
ax.set_ylim(0, 106)
ax.set_ylabel("Accuracy on held-out zero-padding pairs")
ax.set_title("Official checkpoints on held-out zero-padding pairs")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(figures / "confirmatory_accuracy.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.2))
x = range(len(data))
width = 0.34
first = [row["correct_padded_first"] * 100 for row in data]
second = [row["correct_padded_second"] * 100 for row in data]
b1 = ax.bar([i - width / 2 for i in x], first, width,
            color="#F28E2B", label="Padded form first")
b2 = ax.bar([i + width / 2 for i in x], second, width,
            color="#4E79A7", label="Padded form second")
ax.bar_label(b1, labels=[f"{v:.1f}" for v in first], padding=3)
ax.bar_label(b2, labels=[f"{v:.1f}" for v in second], padding=3)
ax.set_xticks(list(x), labels)
ax.set_ylim(0, 108)
ax.set_ylabel("Accuracy (%)")
fig.suptitle("Numeral presentation order changes equality judgments", y=0.98)
ax.legend(frameon=False, loc="lower center", ncols=2,
          bbox_to_anchor=(0.5, 1.01))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(figures / "confirmatory_numeral_order.png", dpi=200)
plt.close(fig)

print(figures / "confirmatory_accuracy.png")
print(figures / "confirmatory_numeral_order.png")
