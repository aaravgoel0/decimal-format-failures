#!/usr/bin/env python3
"""Build publication figures for the additional analyses."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
COLORS = {"Qwen": "#2878B5", "Gemma": "#D65F5F", "Llama": "#6A51A3"}


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="#777777", linewidth=.8, zorder=0)
    ax.grid(axis="y", color="#dddddd", linewidth=.6, alpha=.7, zorder=0)


def model_name(model):
    if model.startswith("Qwen/"):
        return "Qwen"
    if model.startswith("google/"):
        return "Gemma"
    return "Llama"


def causal_generalization():
    data = json.loads((ROOT / "results/causal_generalization_analysis.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.8), sharex="col")
    labels = ["Pooled", "Relations", "Direct choice"]
    templates = ["pooled", "relation_statements", "direct_choice"]
    for col, item in enumerate(data):
        name = model_name(item["model"])
        for row, source in enumerate(("easy", "donor")):
            values=[]; lows=[]; highs=[]
            for template in templates:
                x=next(z for z in item["analyses"] if z["template"] == template)
                value=x[f"{source}_aligned_minus_random"]
                interval=x[f"{source}_contrast_bootstrap_95_ci"]
                values.append(value)
                lows.append(value-interval[0])
                highs.append(interval[1]-value)
            ax=axes[row,col]; xpos=np.arange(3)
            ax.errorbar(xpos,values,yerr=[lows,highs],fmt="o",markersize=7,capsize=4,
                        color=COLORS[name],elinewidth=1.6,zorder=3)
            clean(ax); ax.set_xticks(xpos,labels)
            ax.set_title(f"{name}: {'easy-source rescue' if source=='easy' else 'incompatible-donor corruption'}")
            ax.set_ylabel("Aligned − random margin effect")
    fig.suptitle("Fixed-site causal effects on 100 new prompts and numbers", fontsize=15, y=.99)
    fig.tight_layout(rect=[0,0,1,.965])
    fig.savefig(FIG / "causal_generalization.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def token_decomposition():
    data=json.loads((ROOT / "results/token_decomposition_analysis.json").read_text())
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.4))
    for ax,item in zip(axes,data):
        name=model_name(item["model"])
        rows=[x for x in item["analyses"] if x["template"]=="pooled"]
        order=[("easy","padded"),("easy","short"),("donor","padded"),("donor","short")]
        rows=[next(x for x in rows if (x["source"],x["component"])==key) for key in order]
        values=[x["aligned_minus_random"] for x in rows]
        lows=[x["aligned_minus_random"]-x["bootstrap_95_ci"][0] for x in rows]
        highs=[x["bootstrap_95_ci"][1]-x["aligned_minus_random"] for x in rows]
        x=np.arange(4)
        colors=["#2878B5","#4FA3D1","#D65F5F","#E58A8A"]
        ax.bar(x,values,color=colors,width=.64,zorder=2)
        ax.errorbar(x,values,yerr=[lows,highs],fmt="none",ecolor="#222222",capsize=4,zorder=3)
        ax.set_xticks(x,["Easy\npadded", "Easy\nshort", "Donor\npadded", "Donor\nshort"])
        ax.set_ylabel("Aligned − random margin effect")
        ax.set_title(name); clean(ax)
    fig.suptitle("Which numeral tokens carry the fixed-site causal effect?",fontsize=15)
    fig.tight_layout(rect=[0,0,1,.94])
    fig.savefig(FIG / "token_decomposition.png",dpi=220,bbox_inches="tight")
    plt.close(fig)


def format_robustness():
    data=json.loads((ROOT / "results/format_robustness_analysis.json").read_text())
    family_labels={"negative":"Negative","leading_zero":"Leading zeros","long_fraction":"Long fractions",
                   "scientific":"Scientific","signed_zero":"Signed zero"}
    fig,axes=plt.subplots(1,3,figsize=(12.2,4.6),sharey=True)
    for ax,item in zip(axes,data):
        name=model_name(item["model"]); x=np.arange(5); width=.36
        original=[next(z for z in item["cells"] if z["condition"]=="original" and z["family"]==f)["accuracy"] for f in family_labels]
        canon=[next(z for z in item["cells"] if z["condition"]=="canonicalized" and z["family"]==f)["accuracy"] for f in family_labels]
        ax.bar(x-width/2,original,width,label="Original",color="#777777",zorder=2)
        ax.bar(x+width/2,canon,width,label="Canonicalized",color=COLORS[name],zorder=2)
        ax.set_xticks(x,[family_labels[f] for f in family_labels],rotation=28,ha="right")
        ax.set_ylim(0,1.05); ax.set_ylabel("Accuracy"); ax.set_title(name)
        ax.grid(axis="y",color="#dddddd",linewidth=.6,alpha=.7,zorder=0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,loc="upper center",bbox_to_anchor=(.5,.925),ncol=2)
    fig.suptitle("Canonicalization helps overall, but not in every format family",fontsize=15)
    fig.tight_layout(rect=[0,0,1,.87])
    fig.savefig(FIG / "format_robustness.png",dpi=220,bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    causal_generalization(); token_decomposition(); format_robustness()
    print("built causal_generalization.png, token_decomposition.png, format_robustness.png")


if __name__ == "__main__":
    main()
