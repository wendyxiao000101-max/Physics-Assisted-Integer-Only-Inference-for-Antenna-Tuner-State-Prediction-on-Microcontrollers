from pathlib import Path
import importlib.util, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
spec=importlib.util.spec_from_file_location('audit',ROOT/'Claude outputs/verify_audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
data=ROOT/'esp32p4_iat_demo/input/data'
x=pd.read_csv(data/'x_train.csv',encoding='gbk').to_numpy()
y=pd.read_csv(data/'y_train.csv').to_numpy().ravel().astype(int)
df=pd.read_csv(ROOT/'output/C2补充验证/C2_all_rows_traceable.csv').sort_values('row_index').reset_index(drop=True)
assert np.array_equal(df.row_index,np.arange(len(x)))
assert np.array_equal(df.true,y)
keys=[tuple(map(int,r)) for r in x[:,[2,4,5]]]
lookup={}
for k,s,g in zip(keys,x[:,3],np.hypot(x[:,0],x[:,1])/32768):
    key=(*k,int(s))
    if key in lookup: assert abs(lookup[key]-g)<1e-12
    lookup[key]=g
rl=lambda g:-20*np.log10(g)
loss=lambda g:-10*np.log10(1-g*g)
prov=[];classification=[]
for arm,file in [('A','model_int8_backbone.bin'),('B','model_int8.bin')]:
    preds=np.full(len(y),-1,dtype=int);seen=np.zeros(len(y),dtype=int)
    for fold in sorted((ROOT/'iat_fp32_optimized/model/foldexp').glob('sweep_group_seed*')):
        split=np.load(fold/'split.npz');idx=split['test'];p=fold/'int8'/file
        assert not set(idx)&set(split['train']) and not set(idx)&set(split['valid'])
        m=audit.IntegerModel(p);pred,_=m.predict(x[idx][:,[0,1,2,4,5]])
        meta=json.loads((fold/'int8/metrics.json').read_text(encoding='utf-8'))
        expected=meta['int8_backbone_holdout_accuracy'] if arm=='A' else meta['int8_holdout']['accuracy']
        assert abs((pred==y[idx]).mean()-expected)<1e-12
        if arm=='B': assert m.sha256==meta['model_sha256']
        preds[idx]=pred;seen[idx]+=1
        recalls=np.array([(pred[y[idx]==c]==c).mean() for c in range(15)])
        classification.append(dict(arm=arm,fold=fold.name,n=len(idx),Acc=100*(pred==y[idx]).mean(),BA=100*recalls.mean(),MR=100*recalls.min()))
        prov.append(dict(arm=arm,fold=fold.name,path=str(p),sha256=m.sha256,split_sha256=hashlib.sha256((fold/'split.npz').read_bytes()).hexdigest()))
    assert np.all(seen==1)
    if arm=='B': assert np.array_equal(preds,df.pred)
    df['pred_'+arm]=preds
    g=np.array([lookup.get((*k,int(p)),np.nan) for k,p in zip(keys,preds)])
    df['RL_'+arm]=rl(g);df['loss_'+arm]=loss(g)
    df['MLP_obs_'+arm]=loss(g)-df.loss_best_observed_dB
    df['MLP_label_'+arm]=loss(g)-df.loss_label_dB
assert np.allclose(df.RL_B,df.rl_pred_dB,equal_nan=True,atol=1e-12,rtol=0)
assert np.allclose(df.MLP_obs_B,df.dL_vs_best_observed_dB,equal_nan=True,atol=1e-12,rtol=0)
df['common_AB']=df.RL_A.notna()&df.RL_B.notna()&df.rl_label_dB.notna()
df['prediction_changed']=df.pred_A!=df.pred_B
df['band']=np.where(df.frequency_MHz<800,'703-748','824-849')
common=df[df.common_AB]
assert len(common)>0
pd.DataFrame(prov).to_csv(OUT/'model_provenance.csv',index=False,encoding='utf-8-sig')
cl=pd.DataFrame(classification);cl.to_csv(OUT/'classification_per_fold.csv',index=False)
cl.groupby('arm')[['Acc','BA','MR']].agg(['mean','std']).to_csv(OUT/'classification_summary.csv')
coverage=[]
for arm in ['A','B']:
    own=df['RL_'+arm].notna()&df.rl_label_dB.notna()
    coverage.append(dict(arm=arm,total=len(df),pred_observed=int(df['RL_'+arm].notna().sum()),own_c2_observed=int(own.sum()),common_AB=len(common),common_pct=100*len(common)/len(df)))
pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
stats=[]
subsets=[('common_AB',common),('common_changed',common[common.prediction_changed])]
subsets += [('band_'+name,part) for name,part in common.groupby('band')]
subsets += [(name,part) for name,part in common.groupby('fold')]
for name,part in subsets:
    for arm in ['A','B']:
        for ref in ['obs','label']:
            z=part['MLP_'+ref+'_'+arm]
            stats.append(dict(subset=name,arm=arm,reference=ref,n=len(z),mean=z.mean(),P95=z.quantile(.95),P99=z.quantile(.99),maximum=z.max(),minimum=z.min()))
pd.DataFrame(stats).to_csv(OUT/'MLP_summary.csv',index=False)
rlstats=[]
for tag,col in [('A','RL_A'),('B','RL_B'),('default','rl_default_dB'),('best_observed','rl_best_observed_dB'),('label','rl_label_dB')]:
    z=common[col]
    rlstats.append(dict(state=tag,n=len(z),mean=z.mean(),median=z.median(),P05=z.quantile(.05),P95=z.quantile(.95),RL_ge6_pct=100*z.ge(6).mean(),RL_ge10_pct=100*z.ge(10).mean()))
pd.DataFrame(rlstats).to_csv(OUT/'RL_thresholds.csv',index=False)
cost=[]
for arm in ['A','B']:
    # Include all misclassification counts and the common-subset observable count separately.
    for (t,p),part in df[df['pred_'+arm]!=df.true].groupby(['true','pred_'+arm]):
        v=part[part.common_AB]
        cost.append(dict(arm=arm,true=t,pred=p,all_errors=len(part),common_observed_errors=len(v),mean_label_dB=v['MLP_label_'+arm].mean(),P95_label_dB=v['MLP_label_'+arm].quantile(.95),max_label_dB=v['MLP_label_'+arm].max(),mean_best_observed_dB=v['MLP_obs_'+arm].mean()))
pd.DataFrame(cost).to_csv(OUT/'confusion_cost_pairs.csv',index=False)
for arm in ['A','B']:
    pairs=pd.DataFrame(cost).query('arm==@arm')
    for col in ['all_errors','common_observed_errors','mean_label_dB','mean_best_observed_dB']:
        matrix=pairs.pivot(index='true',columns='pred',values=col).reindex(index=range(15),columns=range(15))
        if col.endswith('errors'):matrix=matrix.fillna(0).astype(int)
        matrix.to_csv(OUT/f'confusion_matrix_{arm}_{col}.csv')
delta=common.loss_A-common.loss_B
paired=dict(n=len(common),mean_A_minus_B_dB=delta.mean(),P05=delta.quantile(.05),P95=delta.quantile(.95),min=delta.min(),max=delta.max(),A_lower=int((delta<-1e-12).sum()),B_lower=int((delta>1e-12).sum()),equal=int((delta.abs()<=1e-12).sum()),all_prediction_changed=int(df.prediction_changed.sum()),common_prediction_changed=int(common.prediction_changed.sum()))
(OUT/'paired_summary.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
df.to_csv(OUT/'all_rows_AB_traceable.csv',index=False,encoding='utf-8-sig')
common[common.prediction_changed].to_csv(OUT/'changed_rows_common.csv',index=False,encoding='utf-8-sig')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axes=plt.subplots(1,2,figsize=(11,4))
for tag,col in [('A','RL_A'),('B','RL_B'),('Default','rl_default_dB'),('Best observed','rl_best_observed_dB')]:
    z=np.sort(common[col]);axes[0].plot(z,np.arange(1,len(z)+1)/len(z),label=tag)
axes[0].set(xlabel='Return loss (dB; higher is better)',ylabel='Empirical CDF')
for threshold in [6,10]:axes[0].axvline(threshold,color='gray',linestyle=':',linewidth=.8)
for arm in ['A','B']:
    z=np.sort(common['MLP_obs_'+arm]);axes[1].plot(z,np.arange(1,len(z)+1)/len(z),label=arm)
axes[1].set(xlabel='Penalty vs best observed (dB; lower is better)',ylabel='Empirical CDF')
for ax in axes:ax.legend();ax.grid(alpha=.2)
fig.suptitle(f'Matched observable subset: {len(common):,} / {len(df):,} rows');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(OUT/f'AB_C2_distributions.{ext}',dpi=200)
verification=dict(rows=len(df),common_rows=len(common),B_predictions_and_RF_match_previous_C2=True,all_folds_accuracy_match_archived_logs=True,data_hashes={f:hashlib.sha256((data/f).read_bytes()).hexdigest() for f in ['x_train.csv','y_train.csv']})
(OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(cl.groupby('arm')[['Acc','BA','MR']].mean().to_string())
print(pd.DataFrame(coverage).to_string(index=False))
print(pd.DataFrame(stats).query("subset=='common_AB'").to_string(index=False))
print(pd.DataFrame(rlstats).to_string(index=False));print(paired)
