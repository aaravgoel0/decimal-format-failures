#!/usr/bin/env python3
"""Compare pinned full-precision MLX runs with matched quantized Ollama runs."""
import json, math
from pathlib import Path

PAIRS = {
    "Llama 3.1 8B": (
        Path("results/llama3.1-8b__confirmatory_zero_padding__p0.jsonl"),
        Path("results/meta-llama-Meta-Llama-3.1-8B-Instruct__confirmatory_zero_padding__p0__mlx.jsonl"),
        "0e9e39f249a16976918f6564b8830bc894c89659",
    ),
    "Gemma 2 9B": (
        Path("results/gemma2-9b__confirmatory_zero_padding__p0.jsonl"),
        Path("results/google-gemma-2-9b-it__confirmatory_zero_padding__p0__mlx.jsonl"),
        "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    ),
}

def wilson(k, n, z=1.959963984540054):
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-radius, center+radius

def exact_mcnemar(b, c):
    n=b+c
    if not n: return 1.0
    tail=sum(math.comb(n,k) for k in range(min(b,c)+1))/(2**n)
    return min(1.0, 2*tail)

def load(path):
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}

def main():
    output=[]
    for model,(quant_path,full_path,revision) in PAIRS.items():
        quant,full=load(quant_path),load(full_path)
        if set(quant)!=set(full) or len(full)!=2000:
            raise RuntimeError(f"{model}: expected 2,000 matched rows")
        if any(r.get("model_revision")!=revision or r["parse_status"]=="error" for r in full.values()):
            raise RuntimeError(f"{model}: revision mismatch or execution error")
        qk=sum(r["correct"] for r in quant.values()); fk=sum(r["correct"] for r in full.values())
        flo,fhi=wilson(fk,len(full)); qlo,qhi=wilson(qk,len(quant))
        q_only=sum(quant[i]["correct"] and not full[i]["correct"] for i in full)
        f_only=sum(full[i]["correct"] and not quant[i]["correct"] for i in full)
        row={"model":model,"n":len(full),"quantized_correct":qk,"quantized_accuracy":qk/len(quant),
             "quantized_ci_low":qlo,"quantized_ci_high":qhi,"full_correct":fk,
             "full_accuracy":fk/len(full),"full_ci_low":flo,"full_ci_high":fhi,
             "full_minus_quantized":(fk-qk)/len(full),"quantized_only_correct":q_only,
             "full_only_correct":f_only,"mcnemar_p_two_sided":exact_mcnemar(q_only,f_only),
             "revision":revision}
        for pos in (1,2):
            subset=[r for r in full.values() if r["padded_position"]==pos]
            row[f"full_padded_position_{pos}_accuracy"]=sum(r["correct"] for r in subset)/len(subset)
        output.append(row)
    Path("results/full_precision_analysis.json").write_text(json.dumps(output,indent=2)+"\n")
    for row in output: print(row)

if __name__=="__main__": main()
