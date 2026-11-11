from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'iat_fp32_optimized'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from iatopt.core import load_dataset,sweep_fold_indices
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
b=load_dataset(ROOT/'esp32p4_iat_demo/input/data/x_train.csv',ROOT/'esp32p4_iat_demo/input/data/y_train.csv')
p=pd.DataFrame(dict(point=b.point_groups,freq=b.raw[:,2],re=b.raw[:,4],im=b.raw[:,5])).drop_duplicates('point')
freqs=sorted(p.freq.unique()); edges=[]
for f in freqs:
    if f-1 not in freqs: continue
    a=p[p.freq==f-1].sort_values(['re','im']); c=p[p.freq==f].sort_values(['re','im'])
    d=cdist(a[['re','im']],c[['re','im']]); r,s=linear_sum_assignment(d)
    for i,j in zip(r,s): edges.append([int(a.iloc[i].point),int(c.iloc[j].point),f-1,f,float(d[i,j])])
e=pd.DataFrame(edges,columns=['from_point','to_point','from_MHz','to_MHz','distance_Q15'])
threshold=float(np.quantile(e.distance_Q15,.95,method='linear'));e['broken']=e.distance_Q15>threshold
parent=np.arange(len(p))
def find(i):
    while parent[i]!=i: i=int(parent[i])
    return i
for row in e.itertuples():
    if not row.broken: parent[find(row.to_point)]=find(row.from_point)
roots=np.array([find(i) for i in range(len(p))]);_,pg=np.unique(roots,return_inverse=True)
groups=pg[b.point_groups];np.save(OUT/'p95_row_groups.npy',groups)
e.to_csv(OUT/'selected_edges.csv',index=False)
p['original_group']=[b.sweep_groups[np.flatnonzero(b.point_groups==i)[0]] for i in p.point]
p['p95_group']=pg[p.point.to_numpy()];p.to_csv(OUT/'point_group_mapping.csv',index=False)
pd.Series(groups).value_counts().sort_index().rename_axis('group').rename('rows').to_csv(OUT/'group_sizes.csv')
splits=OUT/'splits';splits.mkdir(exist_ok=True);diag=[];cover=np.zeros(len(groups),int)
for fold in range(5):
    tr,va,te=sweep_fold_indices(b.labels,groups,fold);cover[te]+=1
    dest=splits/f'fold{fold}.npz';np.savez_compressed(dest,train=tr,valid=va,test=te)
    for a,c in [(tr,va),(tr,te),(va,te)]:
        assert not set(groups[a])&set(groups[c]);assert not set(b.point_groups[a])&set(b.point_groups[c])
    subset=np.empty(len(groups),int);subset[tr]=0;subset[va]=1;subset[te]=2
    ng=pd.DataFrame(dict(old=b.sweep_groups,subset=subset)).groupby('old').subset.nunique()
    cross=ng[ng>1].index.to_numpy()
    diag.append(dict(fold=fold,train=len(tr),valid=len(va),test=len(te),cross_original_groups=len(cross),cross_original_rows=int(np.isin(b.sweep_groups,cross).sum()),sha256=sha(dest),class_counts={k:np.bincount(b.labels[v],minlength=15).tolist() for k,v in [('train',tr),('valid',va),('test',te)]}))
assert np.all(cover==1)
audit=dict(data=b.audit,p95_groups=len(np.unique(groups)),edges=len(e),broken_edges=int(e.broken.sum()),threshold_Q15=threshold,threshold_gamma=threshold/32768,rule='Pooled selected Hungarian edges; linear 0.95 quantile; strictly greater breaks; new explicit implementation, original T18 generator unavailable',splits=diag)
(OUT/'group_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
for name in ['train.py','qat_fold_pipeline.py','run_int8_generalization.bat']:
    shutil.copy2(SRC/name,snapshot/name)
shutil.copytree(SRC/'iatopt',snapshot/'iatopt',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
(OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(snapshot)):sha(p) for p in snapshot.rglob('*') if p.is_file()},indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in audit.items() if k not in ['data','splits']},indent=2))
