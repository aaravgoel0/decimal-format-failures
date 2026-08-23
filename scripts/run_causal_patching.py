#!/usr/bin/env python3
"""Resume-safe layerwise causal residual interchange on an exact checkpoint."""
import argparse, csv, json, random, re
from pathlib import Path
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from evaluate import PROMPTS

DEFAULT_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_REVISION="0e9e39f249a16976918f6564b8830bc894c89659"
SYSTEM="You are a helpful assistant that compares numbers."

def prompt(tokenizer,a,b,template_mode):
    text=PROMPTS[0].format(a=a,b=b)
    messages=([{"role":"system","content":SYSTEM},{"role":"user","content":text}] if template_mode=="system-role" else [{"role":"user","content":SYSTEM+"\n\n"+text}])
    return tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors=None)

def logits_from_residual(model,h):
    h=model.model.norm(h)
    out=(model.model.embed_tokens.as_linear(h) if getattr(model.args,"tie_word_embeddings",True) else model.lm_head(h))
    if getattr(model,"model_type","")=="gemma2": out=mx.tanh(out/model.final_logit_softcapping)*model.final_logit_softcapping
    return out[:, -1]

def margin(logits,ids):
    values=logits[:,ids[3]]-mx.maximum(logits[:,ids[1]],logits[:,ids[2]])
    mx.eval(values)
    array=np.asarray(values.astype(mx.float32))
    return float(array[0]) if len(array)==1 else array

def states(model,ids):
    h=model.model.embed_tokens(mx.array([ids]))
    if getattr(model,"model_type","")=="gemma2": h=h*(model.args.hidden_size**0.5); mask=create_attention_mask(h,None,return_array=True)
    else: mask=create_attention_mask(h,None)
    out=[]
    for layer in model.model.layers:
        h=layer(h,mask,None); mx.eval(h); out.append(h)
    return out,mask

def continue_from(model,h,mask,start):
    for layer in model.model.layers[start:]: h=layer(h,mask,None)
    logits=logits_from_residual(model,h); mx.eval(logits); return logits

def find_once(sequence,subsequence):
    hits=[i for i in range(len(sequence)-len(subsequence)+1) if sequence[i:i+len(subsequence)]==subsequence]
    if len(hits)!=1: raise RuntimeError(f"expected one aligned numeral span, got {hits}")
    return hits[0]

def find_hits(sequence,subsequence):
    hits=[i for i in range(len(sequence)-len(subsequence)+1) if sequence[i:i+len(subsequence)]==subsequence]
    if not hits: raise RuntimeError("aligned numeral span missing")
    return hits

def replace_positions(target,source,mappings):
    pieces=[]; last=0
    for target_pos,source_pos in sorted(mappings):
        pieces.extend([target[:,last:target_pos,:],source[:,source_pos:source_pos+1,:]])
        last=target_pos+1
    pieces.append(target[:,last:,:])
    return mx.concatenate(pieces,axis=1)

def boot_ci(values,seed=7401,reps=10000):
    rng=random.Random(seed); n=len(values); means=[]
    for _ in range(reps): means.append(sum(values[rng.randrange(n)] for _ in range(n))/n)
    means.sort(); return means[int(.025*reps)],means[int(.975*reps)]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",default=DEFAULT_MODEL); p.add_argument("--revision",default=DEFAULT_REVISION); args=p.parse_args()
    model,tok=load(args.model,revision=args.revision); template_mode="system-role"
    try: tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":"test"}],tokenize=False,add_generation_prompt=True)
    except Exception: template_mode="system-prepended-to-user"
    answer_ids={i:tok.encode(str(i),add_special_tokens=False)[0] for i in (1,2,3)}
    safe=re.sub(r"[^A-Za-z0-9_.-]+","-",args.model); out=Path(f"results/causal_patch_{safe}.csv")
    rows=[]
    if out.exists():
        with out.open() as f: rows=list(csv.DictReader(f))
        if rows and ({r["model"] for r in rows}!={args.model} or {r["revision"] for r in rows}!={args.revision}):
            raise RuntimeError("existing causal output has different model provenance")
        for r in rows:
            for key in ("case","digit","whole","zeros","layer"): r[key]=int(r[key])
            for key in ("source_margin","target_margin","patched_margin","margin_effect"): r[key]=float(r[key])
            for key in ("target_correct","patched_correct"): r[key]=r[key].lower()=="true"
    completed={case for case in range(45) if sum(r["case"]==case for r in rows)==len(model.model.layers)*3}
    cases=[]
    for digit in range(1,10):
        split="discovery" if digit<=4 else "heldout"
        for rep in range(5):
            whole=30+digit*4+rep; zeros=2+(digit+rep)%3
            short=f"{whole}.{digit}"; padded=short+"0"*zeros
            cases.append((split,digit,whole,zeros,short,padded))
    for ci,(split,digit,whole,zeros,short,padded) in enumerate(cases):
        if ci in completed:
            print(f"{ci+1}/{len(cases)} already complete",flush=True); continue
        row_start=len(rows)
        source_ids=prompt(tok,short,padded,template_mode)
        target_ids=prompt(tok,padded,short,template_mode)
        if len(source_ids)!=len(target_ids): raise RuntimeError("unaligned sequence lengths")
        short_ids=tok.encode(short,add_special_tokens=False); padded_ids=tok.encode(padded,add_special_tokens=False)
        ss=min(find_hits(source_ids,short_ids)); sp=max(find_hits(source_ids,padded_ids))
        ts=max(find_hits(target_ids,short_ids)); tp=min(find_hits(target_ids,padded_ids))
        number_mappings=[(ts+i,ss+i) for i in range(len(short_ids))]+[(tp+i,sp+i) for i in range(len(padded_ids))]
        excluded={p for p,_ in number_mappings}|{len(target_ids)-1}
        eligible=[p for p in range(1,len(target_ids)-1) if p not in excluded]
        random_positions=random.Random(91000+ci).sample(eligible,len(number_mappings))
        random_mappings=[(p,p) for p in random_positions]
        source_states,_=states(model,source_ids); target_states,mask=states(model,target_ids)
        source_margin=margin(logits_from_residual(model,source_states[-1]),answer_ids)
        target_margin=margin(logits_from_residual(model,target_states[-1]),answer_ids)
        for layer in range(len(model.model.layers)):
            controls=(("answer_position",[(len(target_ids)-1,len(source_ids)-1)]),
                      ("number_tokens",number_mappings),("random_positions",random_mappings))
            for control,mappings in controls:
                patched=replace_positions(target_states[layer],source_states[layer],mappings)
                patched_margin=margin(continue_from(model,patched,mask,layer+1),answer_ids)
                rows.append({"case":ci,"split":split,"digit":digit,"whole":whole,"zeros":zeros,
                             "layer":layer,"control":control,"source_margin":source_margin,
                             "target_margin":target_margin,"patched_margin":patched_margin,
                             "margin_effect":patched_margin-target_margin,
                             "target_correct":target_margin>0,"patched_correct":patched_margin>0,
                             "model":args.model,"revision":args.revision})
                del patched
        with out.open("a",newline="") as f:
            w=csv.DictWriter(f,fieldnames=rows[0])
            if f.tell()==0: w.writeheader()
            w.writerows(rows[row_start:])
        del source_states,target_states
        mx.clear_cache()
        print(f"{ci+1}/{len(cases)}",flush=True)
    with out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    discovery=[r for r in rows if r["split"]=="discovery"]
    layers=range(len(model.model.layers)); score={}
    for layer in layers:
        actual_discovery=[r["margin_effect"] for r in discovery if r["layer"]==layer and r["control"]=="number_tokens"]
        c=[r["margin_effect"] for r in discovery if r["layer"]==layer and r["control"]=="random_positions"]
        score[layer]=sum(actual_discovery)/len(actual_discovery)-sum(c)/len(c)
    chosen=max(score,key=score.get); held=[r for r in rows if r["split"]=="heldout" and r["layer"]==chosen]
    actual=[r["margin_effect"] for r in held if r["control"]=="number_tokens"]
    control=[r["margin_effect"] for r in held if r["control"]=="random_positions"]
    contrasts=[a-c for a,c in zip(actual,control)]; lo,hi=boot_ci(contrasts)
    summary={"model":args.model,"revision":args.revision,"n_discovery_cases":20,"n_heldout_cases":25,
             "chosen_layer_zero_based":chosen,"discovery_selection_contrast":score[chosen],
             "heldout_number_patch_mean_effect":sum(actual)/len(actual),
             "heldout_random_patch_mean_effect":sum(control)/len(control),
             "heldout_difference_in_effects":sum(contrasts)/len(contrasts),
             "heldout_difference_bootstrap_95_ci":[lo,hi],
             "heldout_number_patch_flip_rate":sum(r["patched_correct"] and not r["target_correct"] for r in held if r["control"]=="number_tokens")/len(actual)}
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
