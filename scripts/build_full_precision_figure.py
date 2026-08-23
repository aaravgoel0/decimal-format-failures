#!/usr/bin/env python3
import json
from pathlib import Path
import matplotlib.pyplot as plt

rows=json.loads(Path("results/full_precision_analysis.json").read_text())
labels=[r["model"] for r in rows]; quant=[100*r["quantized_accuracy"] for r in rows]
full=[100*r["full_accuracy"] for r in rows]; x=range(len(rows)); width=.34
fig,ax=plt.subplots(figsize=(7.2,4.4))
ax.bar([i-width/2 for i in x],quant,width,label="Quantized Ollama",color="#8da0cb")
ax.bar([i+width/2 for i in x],full,width,label="Official full precision",color="#66c2a5")
for offset,values in ((-width/2,quant),(width/2,full)):
    for j,v in enumerate(values): ax.text(j+offset,v+1,f"{v:.2f}%",ha="center",fontsize=9)
ax.set_xticks(list(x),labels); ax.set_ylabel("Held-out accuracy (%)"); ax.set_ylim(0,105)
ax.set_title("Quantization sensitivity on identical held-out rows")
ax.legend(frameon=False,loc="upper left"); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/full_precision_comparison.png",dpi=180); plt.close(fig)
