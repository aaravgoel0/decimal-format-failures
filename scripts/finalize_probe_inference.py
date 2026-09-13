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
def indices(rows,split,forms,negative_forms=()):
    return np.array([i for i,r in enumerate(rows) if r['split']==split and (r['form'] in forms or r['form'] in negative_forms)])

def grouped_permutations(rows, selected, y, task, rng, repetitions):
    # Preserve duplicate labels for the same numerical value and prompt-order
    # copies. Equality labels are permuted at the value-form level because the
    # class is a property of form; value targets are permuted by base value.
    keys=[((rows[i]['whole'],rows[i]['digit']) if task=='value'
           else (rows[i]['whole'],rows[i]['digit'],rows[i]['form'])) for i in selected]
    unique=list(dict.fromkeys(keys)); groups=[np.array([j for j,key in enumerate(keys) if key==u]) for u in unique]
    labels=np.array([y[group[0]] for group in groups])
    output=np.empty((len(y),repetitions),float)
    for column in range(repetitions):
        shuffled=rng.permutation(labels)
        for group,label in zip(groups,shuffled): output[group,column]=label
    return output

def main():
    rows=[json.loads(x) for x in open('data/mechanistic_values.jsonl')]
    specs=json.loads(Path('results/cross_format_probes.json').read_text()); rngp=np.random.default_rng(SEED_PERM); rngb=np.random.default_rng(SEED_BOOT)
    out=[]
    for s in specs:
        acts=np.load('activations/'+s['model']+'.npy',mmap_mode='r'); pos=0 if s['position']=='numeral_final' else 1
        train_forms={'canonical'} if s['direction'].startswith('canonical') else {'padded_1','padded_2'}
        test_forms={'padded_1','padded_2'} if s['direction'].startswith('canonical') else {'canonical'}
        if s['task']=='equality':
            train_negative={'near_minus'} if s['direction'].startswith('canonical') else {'near_plus'}
            test_negative={'near_plus'} if s['direction'].startswith('canonical') else {'near_minus'}
        else:
            train_negative=test_negative=set()
        tr=indices(rows,'train',train_forms,train_negative); te=indices(rows,'test',test_forms,test_negative)
        layer=s['selected_layer']; alpha=next(q['alpha'] for q in s['layers'] if q['layer']==layer)
        x=np.asarray(acts[tr,layer,pos],np.float32); z=np.asarray(acts[te,layer,pos],np.float32); x,z,cross,gram=design(x,z)
        if s['task']=='value':
            raw=np.array([rows[i]['value'] for i in tr]); ym=raw.mean(); ys=raw.std(); y=(raw-ym)/ys; yt=np.array([rows[i]['value'] for i in te])
            intercept=y.mean(); centered=y-intercept
            pred=(cross@np.linalg.solve(gram+alpha*np.eye(len(tr)),centered)+intercept)*ys+ym; metric=corr(yt,pred)
        else:
            y=np.array([1 if rows[i]['is_equivalent'] else -1 for i in tr],float); yt=np.array([1 if rows[i]['is_equivalent'] else -1 for i in te])
            intercept=y.mean(); centered=y-intercept
            pred=cross@np.linalg.solve(gram+alpha*np.eye(len(tr)),centered)+intercept; metric=bal(yt,pred)
        perms=grouped_permutations(rows,tr,y,s['task'],rngp,NPERM)
        perm_intercepts=perms.mean(0,keepdims=True)
        pp=cross@np.linalg.solve(gram+alpha*np.eye(len(tr)),perms-perm_intercepts)+perm_intercepts
        if s['task']=='value': null=np.array([corr(yt,pp[:,j]) for j in range(NPERM)])
        else: null=np.array([bal(yt,pp[:,j]) for j in range(NPERM)])
        keys=np.array([(rows[i]['whole'],rows[i]['digit']) for i in te]); unique=list(dict.fromkeys(map(tuple,keys))); groups=[np.where(np.all(keys==k,axis=1))[0] for k in unique]
        boots=[]
        for _ in range(NBOOT):
            chosen=rngb.integers(0,len(groups),len(groups)); ix=np.concatenate([groups[j] for j in chosen])
            boots.append(corr(yt[ix],pred[ix]) if s['task']=='value' else bal(yt[ix],pred[ix]))
        result={k:v for k,v in s.items() if k!='layers'}; result.update({'alpha':alpha,'observed_metric':metric,'permutation_p':float((1+(null>=metric).sum())/(NPERM+1)),
            'permutation_mean':float(null.mean()),'bootstrap_95_ci':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],
            'n_train_rows':len(tr),'n_test_rows':len(te),'n_test_values':len(groups),
            'permutation_scheme':'grouped training-label permutation at fixed validation-selected layer and alpha'})
        out.append(result); print(s['model'],s['position'],s['direction'],s['task'],metric,result['permutation_p'],result['bootstrap_95_ci'],flush=True)
    Path('results/cross_format_probe_inference.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__': main()
