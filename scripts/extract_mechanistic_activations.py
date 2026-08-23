#!/usr/bin/env python3
"""Resume-safe every-layer residual extraction for the fixed mechanistic set."""
import argparse, json, re
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

SYSTEM="You are a helpful assistant that compares numbers."

def find_target(sequence,subsequence,target_position):
    hits=[i for i in range(len(sequence)-len(subsequence)+1) if sequence[i:i+len(subsequence)]==subsequence]
    if len(hits)==1: return hits[0]
    if len(hits)==2: return hits[target_position-1]
    raise RuntimeError(f"target token span alignment failed: {hits}")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",required=True); p.add_argument("--revision",required=True)
    p.add_argument("--dataset",type=Path,default=Path("data/mechanistic_values.jsonl")); p.add_argument("--out-dir",type=Path,default=Path("activations"))
    a=p.parse_args(); rows=[json.loads(x) for x in a.dataset.read_text().splitlines()]
    model,tok=load(a.model,revision=a.revision); safe=re.sub(r"[^A-Za-z0-9_.-]+","-",a.model)
    a.out_dir.mkdir(exist_ok=True); array_path=a.out_dir/f"{safe}.npy"; meta_path=a.out_dir/f"{safe}.jsonl"
    n_layers=len(model.model.layers); hidden=model.args.hidden_size
    if array_path.exists(): acts=np.lib.format.open_memmap(array_path,mode="r+"); assert acts.shape==(len(rows),n_layers,2,hidden)
    else: acts=np.lib.format.open_memmap(array_path,mode="w+",dtype=np.float16,shape=(len(rows),n_layers,2,hidden))
    completed={}
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            r=json.loads(line); completed[r["id"]]=r
    template_mode="system-role"
    try: tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":"test"}],tokenize=False,add_generation_prompt=True)
    except Exception: template_mode="system-prepended-to-user"
    for index,row in enumerate(rows):
        if row["id"] in completed:
            old=completed[row["id"]]; assert old["row_index"]==index and old["revision"]==a.revision
            continue
        other=row["canonical"]
        if row["target_position"]==1: user=f"Number 1: {row['text']}\nNumber 2: {other}\nAre these numerical values equal? Answer yes or no."
        else: user=f"Number 1: {other}\nNumber 2: {row['text']}\nAre these numerical values equal? Answer yes or no."
        messages=([{"role":"system","content":SYSTEM},{"role":"user","content":user}] if template_mode=="system-role" else [{"role":"user","content":SYSTEM+"\n\n"+user}])
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors=None)
        target_ids=tok.encode(row["text"],add_special_tokens=False); start=find_target(ids,target_ids,row["target_position"]); numeral_pos=start+len(target_ids)-1
        h=model.model.embed_tokens(mx.array([ids]))
        if getattr(model,"model_type","")=="gemma2":
            h=h*(model.args.hidden_size**0.5)
            mask=create_attention_mask(h,None,return_array=True)
        else:
            mask=create_attention_mask(h,None)
        for layer_i,layer in enumerate(model.model.layers):
            h=layer(h,mask,None); mx.eval(h)
            acts[index,layer_i,0]=np.asarray(h[0,numeral_pos].astype(mx.float16)); acts[index,layer_i,1]=np.asarray(h[0,-1].astype(mx.float16))
        acts.flush(); record={"id":row["id"],"row_index":index,"model":a.model,"revision":a.revision,"template_mode":template_mode,
                              "sequence_length":len(ids),"target_token_count":len(target_ids),"numeral_final_position":numeral_pos,"answer_position":len(ids)-1,
                              "layers":n_layers,"hidden_size":hidden}
        with meta_path.open("a") as f: f.write(json.dumps(record,sort_keys=True)+"\n")
        if (index+1)%10==0: print(f"{index+1}/{len(rows)}",flush=True)
    assert len(meta_path.read_text().splitlines())==len(rows); print(array_path,meta_path)

if __name__=="__main__": main()
