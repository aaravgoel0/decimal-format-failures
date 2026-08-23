#!/usr/bin/env python3
import json,math,random
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ALPHAS=[1e-4,1e-3,1e-2,1e-1,1,10,100]
MODELS=['meta-llama-Meta-Llama-3.1-8B-Instruct','Qwen-Qwen3-4B-Instruct-2507','google-gemma-2-9b-it']

def ridge_predict(x,y,z,alpha):
    mu=x.mean(0); scale=x.std(0); scale[scale<1e-6]=1
    x=(x-mu)/scale; z=(z-mu)/scale
    return z@x.T@np.linalg.solve(x@x.T+alpha*np.eye(len(x)),y)

def balanced(y,p):
    return .5*((p[y==1]>=0).mean()+(p[y==-1]<0).mean())

def boot(values,seed=73192,reps=10000):
    rng=np.random.default_rng(seed); n=len(values)
    means=np.array([np.mean(values[rng.integers(0,n,n)]) for _ in range(reps)])
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]

def main():
    rows=[json.loads(x) for x in open('data/mechanistic_values.jsonl')]
    output=[]
    for model in MODELS:
        acts=np.load('activations/'+model+'.npy',mmap_mode='r')
        for position,name in ((0,'numeral_final'),(1,'answer')):
          for direction in ('canonical_to_padded','padded_to_canonical'):
            train_forms={'canonical'} if direction.startswith('canonical') else {'padded_1','padded_2'}
            eval_forms={'padded_1','padded_2'} if direction.startswith('canonical') else {'canonical'}
            for task in ('value','equality'):
              curves=[]
              for layer in range(acts.shape[1]):
                def select(split,forms,include_near=False):
                    idx=[]
                    for i,r in enumerate(rows):
                        ok=r['split']==split and (r['form'] in forms or (include_near and r['form'] in {'near_minus','near_plus'}))
                        if ok: idx.append(i)
                    return np.array(idx)
                near=task=='equality'; tr=select('train',train_forms,near); va=select('validation',eval_forms,near); te=select('test',eval_forms,near)
                x=np.asarray(acts[tr,layer,position],dtype=np.float32); xv=np.asarray(acts[va,layer,position],dtype=np.float32); xt=np.asarray(acts[te,layer,position],dtype=np.float32)
                if task=='value':
                    y=np.array([rows[i]['value'] for i in tr]); yv=np.array([rows[i]['value'] for i in va]); yt=np.array([rows[i]['value'] for i in te]); ym=y.mean(); ys=y.std(); y=(y-ym)/ys
                    scores=[]
                    for a in ALPHAS: scores.append(spearmanr(yv,ridge_predict(x,y,xv,a)).statistic)
                    alpha=ALPHAS[int(np.nanargmax(scores))]; pred=ridge_predict(x,y,xt,alpha)*ys+ym; metric=float(spearmanr(yt,pred).statistic)
                    case=np.abs(pred-yt)
                else:
                    y=np.array([1 if rows[i]['is_equivalent'] else -1 for i in tr]); yv=np.array([1 if rows[i]['is_equivalent'] else -1 for i in va]); yt=np.array([1 if rows[i]['is_equivalent'] else -1 for i in te])
                    scores=[balanced(yv,ridge_predict(x,y,xv,a)) for a in ALPHAS]; alpha=ALPHAS[int(np.argmax(scores))]; pred=ridge_predict(x,y,xt,alpha); metric=float(balanced(yt,pred)); case=(pred>=0)==(yt==1)
                curves.append({'layer':layer,'alpha':alpha,'validation_metric':float(max(scores)),'metric':metric,'case_values':case.tolist()})
              best=max(curves,key=lambda q:q['validation_metric']); vals=np.array(best.pop('case_values'),dtype=float)
              for q in curves: q.pop('case_values',None)
              output.append({'model':model,'position':name,'direction':direction,'task':task,'layers':curves,'selected_layer':best['layer'],'test_metric':best['metric']})
              print(model,name,direction,task,best['layer'],best['metric'],flush=True)
    Path('results/cross_format_probes.json').write_text(json.dumps(output,indent=2)+'\n')

if __name__=='__main__': main()
