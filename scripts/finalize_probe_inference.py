#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

SEED_PERM=73191; SEED_BOOT=73192; NPERM=1000; NBOOT=10000

def corr(a,b):
    a=rankdata(a); b=rankdata(b); a=a-a.mean(); b=b-b.mean()
    den=np.sqrt((a*a).sum()*(b*b).sum()); return float((a*b).sum()/den) if den else 0.0
def bal(y,p): return float(.5*((p[y==1]>=0).mean()+(p[y==-1]<0).mean()))
def design(x,z):
    mu=x.mean(0); sd=x.std(0); sd[sd<1e-6]=1
    x=(x-mu)/sd; z=(z-mu)/sd; return x,z,z@x.T,x@x.T
def indices(rows,split,forms,near):
    return np.array([i for i,r in enumerate(rows) if r['split']==split and (r['form'] in forms or (near and r['form'] in {'near_minus','near_plus'}))])

def main():
    rows=[json.loads(x) for x in open('data/mechanistic_values.jsonl')]
    specs=json.loads(Path('results/cross_format_probes.json').read_text()); rngp=np.random.default_rng(SEED_PERM); rngb=np.random.default_rng(SEED_BOOT)
    out=[]
    for s in specs:
        acts=np.load('activations/'+s['model']+'.npy',mmap_mode='r'); pos=0 if s['position']=='numeral_final' else 1
        train_forms={'canonical'} if s['direction'].startswith('canonical') else {'padded_1','padded_2'}
        test_forms={'padded_1','padded_2'} if s['direction'].startswith('canonical') else {'canonical'}
        near=s['task']=='equality'; tr=indices(rows,'train',train_forms,near); te=indices(rows,'test',test_forms,near)
        layer=s['selected_layer']; alpha=next(q['alpha'] for q in s['layers'] if q['layer']==layer)
        x=np.asarray(acts[tr,layer,pos],np.float32); z=np.asarray(acts[te,layer,pos],np.float32); x,z,cross,gram=design(x,z)
        if s['task']=='value':
            raw=np.array([rows[i]['value'] for i in tr]); ym=raw.mean(); ys=raw.std(); y=(raw-ym)/ys; yt=np.array([rows[i]['value'] for i in te])
            pred=(cross@np.linalg.solve(gram+alpha*np.eye(len(tr)),y))*ys+ym; metric=corr(yt,pred)
        else:
            y=np.array([1 if rows[i]['is_equivalent'] else -1 for i in tr],float); yt=np.array([1 if rows[i]['is_equivalent'] else -1 for i in te])
            pred=cross@np.linalg.solve(gram+alpha*np.eye(len(tr)),y); metric=bal(yt,pred)
        perms=np.column_stack([rngp.permutation(y) for _ in range(NPERM)])
        pp=cross@np.linalg.solve(gram+alpha*np.eye(len(tr)),perms)
        if s['task']=='value': null=np.array([corr(yt,pp[:,j]) for j in range(NPERM)])
        else: null=np.array([bal(yt,pp[:,j]) for j in range(NPERM)])
        keys=np.array([(rows[i]['whole'],rows[i]['digit']) for i in te]); unique=list(dict.fromkeys(map(tuple,keys))); groups=[np.where(np.all(keys==k,axis=1))[0] for k in unique]
        boots=[]
        for _ in range(NBOOT):
            chosen=rngb.integers(0,len(groups),len(groups)); ix=np.concatenate([groups[j] for j in chosen])
            boots.append(corr(yt[ix],pred[ix]) if s['task']=='value' else bal(yt[ix],pred[ix]))
        result={k:v for k,v in s.items() if k!='layers'}; result.update({'alpha':alpha,'observed_metric':metric,'permutation_p':float((1+(null>=metric).sum())/(NPERM+1)),
            'permutation_mean':float(null.mean()),'bootstrap_95_ci':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],
            'n_train_rows':len(tr),'n_test_rows':len(te),'n_test_values':len(groups)})
        out.append(result); print(s['model'],s['position'],s['direction'],s['task'],metric,result['permutation_p'],result['bootstrap_95_ci'],flush=True)
    Path('results/cross_format_probe_inference.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__': main()
