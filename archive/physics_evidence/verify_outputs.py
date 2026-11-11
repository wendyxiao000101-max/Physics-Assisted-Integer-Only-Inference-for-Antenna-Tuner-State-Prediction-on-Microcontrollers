"""Independent NumPy replay of exported teacher parameters and table aggregates."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[1]
x=pd.read_csv(R/'sources/x_train.csv',encoding='gbk').to_numpy(dtype=np.float64)
y=pd.read_csv(R/'sources/y_train.csv',encoding='gbk').to_numpy().ravel().astype(int)
metrics=pd.read_csv(R/'tables/C1_03_teacher_metrics_per_fold.csv')
pool=pd.read_csv(R/'tables/C1_07_test_pooled_by_band.csv')
checks=[];rows=[]
for f in sorted((R/'raw').glob('*_test_rows.csv.gz')):
    data=pd.read_csv(f);run=str(data.run.iloc[0]);ids=data.row_id.to_numpy(dtype=int);v=x[ids]
    with np.load(R/'models'/f'{run}_teacher.npz') as model:
        col=np.searchsorted(model['frequencies'],v[:,2]);z=(v[:,4]+1j*v[:,5])/32768
        magnitudes=[]
        for s in range(15):
            a,b,c=model['mobius_coefficients'][s,col].T
            magnitudes.append(np.abs((a*z+b)/(c*z+1)))
        features=np.stack(magnitudes,axis=1)
        xx=((features-model['scaler_mean'])/model['scaler_scale']).astype(np.float32)
        logits=xx@model['lr_coef'].T+model['lr_intercept']
        yp=model['lr_classes'][logits.argmax(axis=1)]
        assert np.array_equal(yp,data.teacher_pred.to_numpy())
        assert np.array_equal(y[ids],data.label.to_numpy())
        a,b,c=model['mobius_coefficients'][v[:,3].astype(int),col].T
        gp=(a*z+b)/(c*z+1)
        e=np.abs(gp-(v[:,0]+1j*v[:,1])/32768)
        np.testing.assert_allclose(e,data.reflection_error.to_numpy(),atol=1e-11,rtol=1e-10)
        m=metrics[(metrics.run==run)&(metrics.subset=='test')].iloc[0]
        acc=100*np.mean(yp==y[ids]);ba=100*np.mean([np.mean(yp[y[ids]==k]==k) for k in range(15)])
        assert abs(acc-m.teacher_acc_pct)<1e-8 and abs(ba-m.teacher_ba_pct)<1e-8
        checks.append(dict(run=run,rows=len(ids),predictions_replayed=True,max_reflection_error_roundtrip=float(np.max(np.abs(e-data.reflection_error))),teacher_acc_pct=acc,teacher_ba_pct=ba))
        rows.append(data)
data=pd.concat(rows,ignore_index=True);g=data[data.protocol=='sweep_group']
assert np.array_equal(np.sort(g.row_id.to_numpy()),np.arange(len(y)))
for band,mask in [('all',np.ones(len(g),bool)),('703-748',g.frequency_MHz<800),('824-849',g.frequency_MHz>=800)]:
    d=g.loc[mask];p=pool[(pool.protocol=='sweep_group')&(pool.band==band)].iloc[0]
    assert abs(d.reflection_error.mean()-p['mean'])<1e-11
    assert abs(d.reflection_error.quantile(.95)-p['p95'])<1e-11
manifest=json.loads((R/'manifest.json').read_text(encoding='utf-8'))
for name,digest in manifest['sources'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
report=dict(all_checks_passed=True,grouped_rows_tested_exactly_once=len(g),replay_checks=checks,
            source_hashes_unchanged=True,script_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'scripts').glob('*.py')})
(R/'independent_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS: all 10 teacher parameter exports replayed; test metrics and physical errors verified; source hashes unchanged.')
