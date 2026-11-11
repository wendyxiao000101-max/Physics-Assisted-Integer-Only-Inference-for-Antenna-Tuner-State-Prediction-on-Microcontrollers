from pathlib import Path
import sys, os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, sklearn, scipy
import hashlib, json, ast, time, platform
ROOT = Path(os.environ.get('IAT_PROJECT_ROOT', '.'))  # original working tree, see archive/README.md
OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)
scripts = ROOT/'iat_fp32_optimized/C5/scripts'
sys.path.insert(0, str(scripts))
from baseline_pareto_v2 import build, build_raw5, forest_predictions, forest_sizes
import tree_codec as tc
hash_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

data = ROOT/'esp32p4_iat_demo/input/data'
raw = pd.read_csv(data/'x_train.csv', encoding='gbk').to_numpy(dtype=np.float64)
y = pd.read_csv(data/'y_train.csv').to_numpy().ravel().astype(int)
x = build_raw5(raw)
assert x.shape == (96768,5) and len(y)==96768
selected = pd.read_csv(ROOT/'output/C5论文修订_20260927/selected_baselines.csv')
archived = pd.read_csv(ROOT/'iat_fp32_optimized/C5/raw/baseline_pareto_v2_raw.csv')
audit = pd.read_csv(ROOT/'output/C2_AB对比/all_rows_AB_traceable.csv').sort_values('row_index').reset_index(drop=True)
main = pd.read_csv(ROOT/'output/Figure5_Origin_20261011/Figure5_77413_matched_records.csv')
assert np.array_equal(audit.row_index, np.arange(len(y))) and np.array_equal(audit.true,y)
assert set(main.row_index) == set(audit.loc[audit.common_AB, 'row_index'])
assert np.array_equal(main.sort_values('row_index').pred_A, audit.loc[audit.common_AB,'pred_A'])
seen = np.zeros(len(y), dtype=int)
foldrows=[]; provenance=[]
preds={name:np.full(len(y),-1,dtype=int) for name in ['decision_tree','extra_trees']}

for fold in sorted((ROOT/'iat_fp32_optimized/model/foldexp').glob('sweep_group_seed*')):
    seed=int(fold.name.split('seed')[1]); z=np.load(fold/'split.npz')
    tr,va,te=z['train'],z['valid'],z['test']
    assert not(set(tr)&set(te) or set(va)&set(te) or set(tr)&set(va))
    assert np.all(audit.iloc[te].fold == fold.name)
    seen[te]+=1
    metadata=json.loads((fold/'int8/metrics.json').read_text(encoding='utf8'))
    a_acc=(audit.iloc[te].pred_A.to_numpy()==y[te]).mean()
    assert abs(a_acc-metadata['int8_backbone_holdout_accuracy'])<1e-12
    provenance.append({'fold':fold.name,'split_sha256':hash_file(fold/'split.npz'),'INT8_A_model_sha256':hash_file(fold/'int8/model_int8_backbone.bin')})
    for name in preds:
        rec=selected[selected.model==name].iloc[0]
        assert rec.representation=='hard'
        params=ast.literal_eval(rec.params)
        saved=archived[(archived.config==rec.config)&(archived.seed==seed)&(archived.representation=='hard')]
        assert len(saved)==1
        saved=saved.iloc[0]
        started=time.perf_counter()
        model=build(name,params,seed)
        if name=='extra_trees':model.set_params(n_jobs=4)
        model.fit(x[tr],y[tr])
        estimators=getattr(model,'estimators_',[model])
        pv,_=forest_predictions(estimators,x[va]);pt,_=forest_predictions(estimators,x[te])
        sizes=forest_sizes(estimators)
        accv=float((pv==y[va]).mean()*100);acct=float((pt==y[te]).mean()*100)
        checks={'valid_acc_diff_pp':accv-float(saved.valid_acc),'test_acc_diff_pp':acct-float(saved.test_acc),'bytes_diff':sizes['hard']-int(saved['bytes'])}
        row={'model':name,'fold':fold.name,'seed':seed,'n_train':len(tr),'n_valid':len(va),'n_test':len(te),'valid_acc':accv,'test_acc':acct,'bytes':sizes['hard'],'internal_nodes':sizes['internal'],'leaf_nodes':sizes['leaves'],**checks}
        foldrows.append(row)
        pd.DataFrame(foldrows).to_csv(OUT/'reconstruction_checks.csv',index=False)
        print(json.dumps(row),flush=True)
        if not(abs(checks['valid_acc_diff_pp'])<1e-10 and abs(checks['test_acc_diff_pp'])<1e-10 and checks['bytes_diff']==0):
            raise RuntimeError('Reconstruction does not reproduce archived metrics; do not publish as original models.')
        blob=tc.encode(model,tc.HARD)
        assert len(blob)==sizes['hard']
        decoded=tc.predict(blob,x[te])
        assert np.array_equal(decoded,pt)
        bpath=OUT/f'{name}_seed{seed}.bin';bpath.write_bytes(blob)
        provenance[-1][name+'_model_sha256']=hash_file(bpath)
        preds[name][te]=pt
        np.savez_compressed(OUT/f'{name}_seed{seed}_predictions.npz',valid_index=va,valid_prediction=pv,test_index=te,test_prediction=pt)
        print(name,seed,'encoded/test predictions verified; seconds',round(time.perf_counter()-started,2),flush=True)

assert np.all(seen==1) and all(np.all(p>=0) for p in preds.values())
for name, pred in preds.items():audit['pred_'+name]=pred

keys=[tuple(map(int,r)) for r in raw[:,[2,4,5]]]
lookup={}
for k,s,re,im in zip(keys,raw[:,3],raw[:,0],raw[:,1]):
    key=(*k,int(s));rho2=(re*re+im*im)/(32768.0**2)
    if key in lookup:assert abs(lookup[key]-rho2)<1e-12
    lookup[key]=rho2
loss=lambda q:-10*np.log10(1-q)
defaultq=(raw[:,4]**2+raw[:,5]**2)/(32768.0**2)
assert np.all((defaultq>=0)&(defaultq<1))
ld=loss(defaultq)
assert np.allclose(ld,audit.loss_default_dB,atol=1e-12)
for name in ['decision_tree','extra_trees','A']:
    q=np.array([lookup.get((*k,int(p)),np.nan) for k,p in zip(keys,audit['pred_'+name])])
    assert np.all((q[np.isfinite(q)]>=0)&(q[np.isfinite(q)]<1))
    calculated=loss(q)
    if name=='A':assert np.allclose(calculated,audit.loss_A,equal_nan=True,atol=1e-12,rtol=0)
    audit['loss_'+name]=calculated
    audit['MLP_obs_'+name]=calculated-audit.loss_best_observed_dB
    audit['gain_default_'+name]=ld-calculated

base=audit.common_AB.to_numpy(dtype=bool)
common=base & audit.loss_decision_tree.notna().to_numpy() & audit.loss_extra_trees.notna().to_numpy()
audit['common_three']=common
assert np.all(audit.loc[common,'MLP_obs_A']>=-1e-12)
stats=[]
for subset,mask in [('main_AB',base),('common_three',common)]:
    for name in ['decision_tree','extra_trees','A']:
        use=mask & audit['loss_'+name].notna().to_numpy()
        v=audit.loc[use];pen=v['MLP_obs_'+name];gain=v['gain_default_'+name]
        stats.append({'subset':subset,'model':name,'n':len(v),'points':v.point_id.nunique(),'pct_all':len(v)/len(audit)*100,'mean_penalty_dB':pen.mean(),'P95_penalty_dB':pen.quantile(.95),'max_penalty_dB':pen.max(),'mean_tune_gain_dB':gain.mean(),'negative_gain_count':int((gain < -1e-12).sum()),'negative_gain_pct':100*(gain < -1e-12).mean(),'RL_ge10_pct':100*(v['loss_'+name] <= -10*np.log10(.9)+1e-12).mean(),'subset_pooled_acc':100*(v['pred_'+name]==v.true).mean()})
summary=pd.DataFrame(stats);summary.to_csv(OUT/'rf_summary.csv',index=False)
foldstats=[]
for fold,part in audit.loc[common].groupby('fold'):
    for name in ['decision_tree','extra_trees','A']:
        pen=part['MLP_obs_'+name];gain=part['gain_default_'+name]
        foldstats.append({'fold':fold,'model':name,'n':len(part),'mean_penalty_dB':pen.mean(),'P95_penalty_dB':pen.quantile(.95),'mean_tune_gain_dB':gain.mean(),'negative_gain_pct':100*(gain < -1e-12).mean()})
pd.DataFrame(foldstats).to_csv(OUT/'rf_per_fold.csv',index=False)
coverage=[]
for name in ['decision_tree','extra_trees','A']:
    obs=audit['loss_'+name].notna().to_numpy()
    coverage.append({'model':name,'observable_all':int(obs.sum()),'observable_in_main_AB':int((obs&base).sum()),'common_three':int(common.sum()),'main_AB':int(base.sum()),'all_rows':len(audit),'excluded_errors':int(((audit['pred_'+name]!=audit.true)&~common).sum()),'included_errors':int(((audit['pred_'+name]!=audit.true)&common).sum())})
pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
columns=['row_index','fold','point_id','frequency_MHz','true','common_AB','common_three','loss_default_dB','loss_best_observed_dB']
for name in ['decision_tree','extra_trees','A']:columns += ['pred_'+name,'loss_'+name,'MLP_obs_'+name,'gain_default_'+name]
audit[columns].to_csv(OUT/'all_rows_three_models.csv',index=False)
audit.loc[common,columns].to_csv(OUT/'common_three_rows.csv',index=False)
report={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,
    'source_description':'Fixed validation-selected tree configurations reconstructed on archived train/valid/test indices; no new search; every fold reproduces archived validation/test accuracy and encoded size; all encoded test predictions equal hard-vote reference.',
    'n_jobs':4,'archived_int8_predictions_used':True,'all_INT8_fold_accuracies_match_logs':True,'all_tree_fold_metrics_match_archive':True,'original_main_subset_n':int(base.sum()),'common_three_n':int(common.sum()),'common_three_points':int(audit.loc[common].point_id.nunique()),'common_pct_of_main':common.sum()/base.sum()*100,'common_pct_of_all':common.mean()*100,'negative_tune_tolerance':1e-12,'quantile_method':'pandas linear interpolation',
    'data_hashes':{f:hash_file(data/f) for f in ['x_train.csv','y_train.csv']},'selected_configurations':selected[selected.model.isin(preds)].to_dict('records'),'provenance':provenance,'rf_summaries':stats}
(OUT/'analysis_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(summary.to_string(index=False),flush=True)
print('DONE',json.dumps({k:report[k] for k in ['original_main_subset_n','common_three_n','common_three_points','common_pct_of_main','common_pct_of_all']},ensure_ascii=False),flush=True)
