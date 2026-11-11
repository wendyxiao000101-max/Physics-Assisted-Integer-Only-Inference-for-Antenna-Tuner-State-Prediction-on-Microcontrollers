from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from scipy.spatial.distance import cdist
O=Path(__file__).resolve().parent;R=O.parents[1]
sys.path.insert(0,str(O/'source_snapshot'))
from iatopt.core import load_dataset,sweep_fold_indices,metrics
b=load_dataset(R/'esp32p4_iat_demo/input/data/x_train.csv',R/'esp32p4_iat_demo/input/data/y_train.csv')
mag=np.hypot(b.raw[:,0],b.raw[:,1])/32768
lookup=np.full((7560,16),np.nan)
for p,s,g in zip(b.point_groups,b.raw[:,3].astype(int),mag):
    assert np.isnan(lookup[p,s]) or abs(lookup[p,s]-g)<1e-12
    lookup[p,s]=g
label=np.array([b.labels[np.flatnonzero(b.point_groups==p)[0]] for p in range(7560)])
best=np.nanmin(lookup[:,:15],axis=1);lg=lookup[np.arange(7560),label];ok=np.isfinite(lg)
loss=lambda g:-10*np.log10(1-g*g)
delta=loss(lg)-loss(best)
pd.DataFrame(dict(point=np.arange(7560),label=label,label_gamma=lg,best_gamma=best,label_observable=ok,label_is_min=ok&(lg<=best+1e-12),label_penalty_dB=delta)).to_csv(O/'label_audit_per_point.csv',index=False)
audit=dict(points=7560,missing=int((~ok).sum()),observable=int(ok.sum()),label_min_count=int((ok&(lg<=best+1e-12)).sum()),label_min_pct=float((lg[ok]<=best[ok]+1e-12).mean()*100),mean_penalty_dB=float(delta[ok].mean()),max_penalty_dB=float(delta[ok].max()),candidate_states='0..14',tie_tolerance=1e-12)
(O/'label_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8');print(audit,flush=True)
edges=pd.read_csv(O/'selected_edges.csv');point=pd.read_csv(O/'point_group_mapping.csv')
def cut(q):
    parent=np.arange(7560)
    def find(i):
        while parent[i]!=i:i=int(parent[i])
        return i
    t=np.quantile(edges.distance_Q15,q,method='linear')
    for row in edges.itertuples():
        if row.distance_Q15<=t:parent[find(row.to_point)]=find(row.from_point)
    _,g=np.unique([find(i) for i in range(7560)],return_inverse=True)
    return g[b.point_groups]
greedy=np.empty(7560,int);nextid=0
for f in sorted(point.freq.unique()):
    cur=point[point.freq==f].sort_values(['re','im'])
    if f-1 not in set(point.freq):
        greedy[cur.point]=np.arange(nextid,nextid+len(cur));nextid+=len(cur)
    else:
        d=cdist(prev[['re','im']],cur[['re','im']]);usedr=set();usedc=set()
        for ix in np.argsort(d,axis=None,kind='stable'):
            i,j=np.unravel_index(ix,d.shape)
            if i not in usedr and j not in usedc:
                greedy[int(cur.iloc[j].point)]=greedy[int(prev.iloc[i].point)];usedr.add(i);usedc.add(j)
    prev=cur
rng=np.random.default_rng(20260823);rand=np.tile(np.arange(210),36);rng.shuffle(rand)
protocols=dict(original=b.sweep_groups,P99=cut(.99),P95=cut(.95),P90=cut(.90),greedy=greedy[b.point_groups],point=b.point_groups,random210=rand[b.point_groups])
rows=[];x=b.raw[:,[0,1,2,4,5]]
for name,g in protocols.items():
    for f in range(5):
        tr,va,te=sweep_fold_indices(b.labels,g,f)
        m=ExtraTreesClassifier(n_estimators=120,random_state=42+f,n_jobs=4).fit(x[tr],b.labels[tr]);pred=m.predict(x[te]);v=metrics(b.labels[te],pred)
        rows.append(dict(protocol=name,groups=len(np.unique(g)),fold=f,**{k:float(v[k])*100 for k in ['accuracy','balanced_accuracy','minimum_class_recall']}))
    pd.DataFrame(rows).to_csv(O/'ExtraTrees_sensitivity_folds.csv',index=False)
    print(name,pd.DataFrame(rows).query('protocol==@name').accuracy.mean(),flush=True)
(O/'ExtraTrees_config.json').write_text(json.dumps(dict(n_estimators=120,n_jobs=4,seed='42+fold',features='raw5',split='same nested sweep_fold_indices; fit train only',note='New reproducible probe; does not substitute unpublished configuration behind archived T18',random210='balanced random assignment of 7560 points, seed20260823',greedy='global shortest edge first, stable ties'),indent=2),encoding='utf8')
