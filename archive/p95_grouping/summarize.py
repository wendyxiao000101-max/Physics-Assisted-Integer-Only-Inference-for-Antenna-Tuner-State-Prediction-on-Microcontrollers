from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np,pandas as pd,torch
O=Path(__file__).resolve().parent;R=O.parents[1]
sys.path.insert(0,str(O/'source_snapshot'))
from iatopt.core import load_dataset,sweep_fold_indices,metrics
from iatopt.quantization import load_fp32_teacher
torch.set_num_threads(4)
b=load_dataset(R/'esp32p4_iat_demo/input/data/x_train.csv',R/'esp32p4_iat_demo/input/data/y_train.csv')
spec=importlib.util.spec_from_file_location('independent_integer',R/'Claude outputs/verify_audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
old=pd.read_csv(R/'output/C2_AB对比/all_rows_AB_traceable.csv').sort_values('row_index').reset_index(drop=True)
assert np.array_equal(old.true,b.labels)
pred=np.full(len(b.labels),-1,int);fp=pred.copy();foldcol=pred.copy();seen=np.zeros(len(pred),int)
rows=[];checks=[];provenance=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
keys=['dropout','head_scale','hybrid_alpha_init','checkpoint_average_k','loss','ldam_max_margin','ala_linear_strength','ala_schedule','ala_warmup_fraction','ala_peak_fraction','ala_final_factor','pairwise_ala_strength','trajectory_consistency_weight','physics_teacher','physics_per_point','physics_sigma','physics_temperature','physics_weight','physics_batch_ratio','physics_mode','physics_min_distance','physics_oversample','physics_weight_schedule','physics_final_weight','physics_decay_end','optimizer','adam_beta2','scheduler']
for f,seed in enumerate(range(42,47)):
    md=O/f'models/sweep_group_seed{seed}';z=np.load(md/'int8_A/predictions.npz');idx=z['test'];te=np.load(O/f'splits/fold{f}.npz')['test'];assert np.array_equal(idx,te)
    binary=md/'int8_A/model_int8.bin';q,_=audit.IntegerModel(binary).predict(b.raw[:,[0,1,2,4,5]])
    assert np.array_equal(q,z['int8']);pred[idx]=q[idx];fp[idx]=z['fp32'][idx];foldcol[idx]=f;seen[idx]+=1
    omd=R/f'iat_fp32_optimized/model/foldexp/sweep_group_seed{seed}';osp=np.load(omd/'split.npz')
    for k,ix in zip(['train','valid','test'],sweep_fold_indices(b.labels,b.sweep_groups,f)):assert np.array_equal(osp[k],ix)
    om=json.loads((omd/'metadata.json').read_text(encoding='utf8'))
    original_model,original_scaler,_=load_fp32_teacher(omd)
    with torch.no_grad(): op=original_model(torch.tensor(original_scaler.transform(b.raw[osp['test']][:,[0,1,2,4,5]]),dtype=torch.float32)).argmax(1).numpy()
    assert abs((op==b.labels[osp['test']]).mean()-om['fp32_holdout']['accuracy'])<1e-12
    olda,_=audit.IntegerModel(omd/'int8/model_int8_backbone.bin').predict(b.raw[osp['test']][:,[0,1,2,4,5]])
    assert np.array_equal(olda,old.pred_A.to_numpy()[osp['test']])
    for arm,truth,pp in [('FP32',b.labels[idx],z['fp32'][idx]),('INT8-A',b.labels[idx],q[idx])]:
        v=metrics(truth,pp);rows.append(dict(grouping='P95',groups=578,fold=f,model=arm,n=len(idx),Acc=v['accuracy']*100,BA=v['balanced_accuracy']*100,MR=v['minimum_class_recall']*100))
    v=om['fp32_holdout'];rows.append(dict(grouping='Original',groups=210,fold=f,model='FP32',n=len(osp['test']),Acc=v['accuracy']*100,BA=v['balanced_accuracy']*100,MR=v['minimum_class_recall']*100))
    ix=osp['test'];v=metrics(b.labels[ix],old.pred_A.to_numpy()[ix]);rows.append(dict(grouping='Original',groups=210,fold=f,model='INT8-A',n=len(ix),Acc=v['accuracy']*100,BA=v['balanced_accuracy']*100,MR=v['minimum_class_recall']*100))
    oldrun=json.loads((R/f'iat_fp32_optimized/results_int8gen_sweep/ablation/runs/raw5__residual32_gelu_relu_hybrid__sweep_group__{seed}.json').read_text(encoding='utf8'))['summary']
    newrun=json.loads((O/f'training/ablation/runs/raw5__residual32_gelu_relu_hybrid__sweep_group__{seed}.json').read_text(encoding='utf8'))['summary']
    differences={k:[oldrun.get(k),newrun.get(k)] for k in keys if oldrun.get(k)!=newrun.get(k)}
    assert not differences,differences
    qa=json.loads((md/'int8_A/metrics.json').read_text(encoding='utf8'))
    assert qa['hybrid_alpha']>=(.994 if seed==43 else .999) and binary.stat().st_size==4656
    checks.append(dict(fold=f,config_matched_keys=keys,alpha=qa['hybrid_alpha'],fold_changes_all=qa['fold_top1_changes_all_rows'],fold_changes_test=int((z['fp32'][idx]!=z['folded'][idx]).sum()),integer_implementations_agree_all=True))
    for p in [binary,md/'model_fp32.pth',md/'scaler.pkl',md/'split.npz',omd/'int8/model_int8_backbone.bin']:
        provenance.append(dict(path=str(p),sha256=sha(p)))
assert np.all(seen==1)
df=pd.DataFrame(rows);df.to_csv(O/'classification_folds.csv',index=False)
summary=[]
for (g,m),part in df.groupby(['grouping','model']):
    summary.append(dict(grouping=g,model=m,**{k+'_'+stat:float(getattr(part[k],stat)()) for k in ['Acc','BA','MR'] for stat in ['mean','std']}))
pd.DataFrame(summary).to_csv(O/'classification_summary.csv',index=False)
wide=df.pivot(index=['grouping','fold'],columns='model',values=['Acc','BA','MR'])
delta=pd.DataFrame({k:wide[k]['INT8-A']-wide[k]['FP32'] for k in ['Acc','BA','MR']}).reset_index();delta.to_csv(O/'quantization_delta_folds.csv',index=False)
delta.groupby('grouping')[['Acc','BA','MR']].agg(['mean','std']).to_csv(O/'quantization_delta_summary.csv')
lookup=np.full((7560,16),np.nan)
for point,state,mag in zip(b.point_groups,b.raw[:,3].astype(int),np.hypot(b.raw[:,0],b.raw[:,1])/32768):lookup[point,state]=mag
g=lookup[b.point_groups,pred];loss=lambda a:-10*np.log10(1-a*a)
np.testing.assert_allclose(loss(np.nanmin(lookup[:,:15],axis=1))[b.point_groups],old.loss_best_observed_dB.to_numpy(),atol=1e-12,rtol=0)
rl=-20*np.log10(g);mlp=loss(g)-old.loss_best_observed_dB.to_numpy()
common=np.isfinite(g)&old.RL_A.notna().to_numpy()&old.rl_label_dB.notna().to_numpy()
trace=pd.DataFrame(dict(row_index=np.arange(len(pred)),p95_fold=foldcol,original_fold=old.fold,true=b.labels,point_id=b.point_groups,original_A=old.pred_A,p95_FP32=fp,p95_A=pred,RL_original_A=old.RL_A,RL_p95_A=rl,MLP_original_A=old.MLP_obs_A,MLP_p95_A=mlp,common=common))
trace.to_csv(O/'oof_and_RF_traceable.csv',index=False)
rf=[]
for arm,r,l in [('Original-A',old.RL_A.to_numpy(),old.MLP_obs_A.to_numpy()),('P95-A',rl,mlp)]:
    rf.append(dict(model=arm,n=int(common.sum()),coverage_pct=float(common.mean()*100),mean_MLP=float(l[common].mean()),P95_MLP=float(np.quantile(l[common],.95)),max_MLP=float(l[common].max()),RL_ge6_pct=float((r[common]>=6).mean()*100),RL_ge10_pct=float((r[common]>=10).mean()*100)))
pd.DataFrame(rf).to_csv(O/'RF_sensitivity_common.csv',index=False)
pd.DataFrame(provenance).to_csv(O/'model_provenance.csv',index=False)
(O/'verification.json').write_text(json.dumps(dict(checks=checks,OOF_exactly_once=True,all_new_groups_disjoint=True,matched_RF_rows=int(common.sum()),data_hashes={p.name:sha(p) for p in [R/'esp32p4_iat_demo/input/data/x_train.csv',R/'esp32p4_iat_demo/input/data/y_train.csv']}),ensure_ascii=False,indent=2),encoding='utf8')
print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(rf).to_string(index=False))
