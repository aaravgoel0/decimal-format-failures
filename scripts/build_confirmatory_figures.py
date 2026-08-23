#!/usr/bin/env python3
"""Build publication figures from the confirmatory analysis."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
payload = json.loads((ROOT / "results" / "confirmatory_analysis.json").read_text())
data = pd.DataFrame(payload["summaries"])
figures = ROOT / "figures"
figures.mkdir(exist_ok=True)

short = {"Llama 3.1 8B": "Llama 3.1 8B",
         "Qwen3 4B Instruct 2507": "Qwen3 4B Instruct 2507",
         "Gemma 2 9B": "Gemma 2 9B"}
labels = [short[x] for x in data.model]
colors = ["#4062BB", "#59A14F", "#E15759"]

fig, ax = plt.subplots(figsize=(9, 5.2))
y = data.accuracy * 100
err = [(data.accuracy - data.ci_low) * 100,
       (data.ci_high - data.accuracy) * 100]
bars = ax.bar(labels, y, color=colors, yerr=err, capsize=4)
ax.bar_label(bars, labels=[f"{x:.1f}%" for x in y], padding=4, fontsize=11)
ax.set_ylim(0, 106)
ax.set_ylabel("Accuracy on held-out zero-padding pairs")
ax.set_title("Held-out confirmation (2,000 examples per model)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(figures / "confirmatory_accuracy.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5.2))
x = range(len(data))
width = 0.34
first = data.padded_first_accuracy * 100
second = data.padded_second_accuracy * 100
b1 = ax.bar([i - width / 2 for i in x], first, width,
            color="#F28E2B", label="Padded form first")
b2 = ax.bar([i + width / 2 for i in x], second, width,
            color="#4E79A7", label="Padded form second")
ax.bar_label(b1, labels=[f"{v:.1f}" for v in first], padding=3)
ax.bar_label(b2, labels=[f"{v:.1f}" for v in second], padding=3)
ax.set_xticks(list(x), labels)
ax.set_ylim(0, 108)
ax.set_ylabel("Accuracy (%)")
fig.suptitle("Answer order dominates Llama's equality judgments", y=0.98)
ax.legend(frameon=False, loc="lower center", ncols=2,
          bbox_to_anchor=(0.5, 1.01))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(rect=(0, 0, 1, 0.88))
fig.savefig(figures / "confirmatory_order_effect.png", dpi=200)
plt.close(fig)

print(figures / "confirmatory_accuracy.png")
print(figures / "confirmatory_order_effect.png")
