"""Train-only refits on archived splits; no student retraining or paper edits."""
from pathlib import Path
import os
import sys, json, hashlib, importlib.util, warnings, platform
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import scipy, sklearn, torch, joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'sources'
spec = importlib.util.spec_from_file_location('c1_core_snapshot', SRC/'core_snapshot.py')
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)
torch.set_num_threads(1)

def csv(name, records):
    df = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    df.to_csv(ROOT/'tables'/name, index=False, encoding='utf-8-sig', float_format='%.12g')
    return df

def statistics(e):
    return dict(rows=len(e), mean=float(np.mean(e)), rmse=float(np.sqrt(np.mean(e*e))),
                median=float(np.median(e)), p95=float(np.percentile(e,95)),
                maximum=float(np.max(e)))

def physical_errors(physics, raw, ids):
    r=raw[ids]
    z=(r[:,4]+1j*r[:,5])/32768
    w=(r[:,0]+1j*r[:,1])/32768
    col=physics._columns(r[:,2])
    a,b,c=physics.coefficients[r[:,3].astype(int),col].T
    predicted=(a*z+b)/(c*z+1)
    eg=np.abs(predicted-w)
    ea=np.abs(a*z+b-c*z*w-w)
    np.testing.assert_allclose(ea,np.abs(c*z+1)*eg,atol=2e-15,rtol=1e-8)
    return eg,ea,predicted,np.abs(c*z+1)

def cell_diagnostics(physics,raw,ids,run):
    r=raw[ids]; out=[]
    for col,f in enumerate(physics.frequencies):
        for s in range(16):
            v=r[(r[:,2]==f)&(r[:,3]==s)]
            z=(v[:,4]+1j*v[:,5])/32768;w=(v[:,0]+1j*v[:,1])/32768
            design=np.column_stack([z,np.ones_like(z),-z*w])
            sv=np.linalg.svd(design,compute_uv=False)
            coeff=physics.coefficients[s,col]
            out.append(dict(run=run,state=s,freq=int(f),training_rows=len(v),
                rank=int(np.linalg.matrix_rank(design)),condition_number=float(sv[0]/sv[-1]),
                fallback=not bool(physics.fitted[s,col]),
                abs_a=abs(coeff[0]),abs_b=abs(coeff[1]),abs_c=abs(coeff[2]),
                map_determinant_abs=abs(coeff[0]-coeff[1]*coeff[2])))
    return out

def main():
    bundle=core.load_dataset(SRC/'x_train.csv',SRC/'y_train.csv')
    raw,y=bundle.raw,bundle.labels
    n=len(y);all_ids=np.arange(n)
    manifest={'python':sys.version,'executable':sys.executable,
        'versions':{k:v for k,v in [('numpy',np.__version__),('pandas',pd.__version__),
        ('scipy',scipy.__version__),('sklearn',sklearn.__version__),('torch',torch.__version__)]},
        'fit_definition':'PhysicsTeacher.fit on saved train indices only; C=10; max_iter=3000; Standardizer float32 output',
        'error_definition':'abs(M(Gamma2)-Gamma1); no clipping; dimensionless',
        'threadpools':threadpool_info(),
        'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in SRC.rglob('*') if p.is_file()},'dataset_audit':bundle.audit}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

    # Audit the supplied T17 independently of all held-out evaluations.
    full=core.MobiusPhysics.fit(raw,all_ids)
    eg,ea,pred,den=physical_errors(full,raw,all_ids)
    old=pd.read_csv(SRC/'T17_mobius_fit_quality.csv')
    audit=[];corrected=[]
    for rec in old.to_dict('records'):
        mask=(raw[:,3]==rec['state'])&(raw[:,2]==rec['freq'])
        g=eg[mask];a=ea[mask]
        audit.append(dict(state=rec['state'],freq=rec['freq'],
            old_res_med=rec['res_med'],algebraic_median=np.median(a),reflection_median=np.median(g),
            old_res_p95=rec['res_p95'],algebraic_p95=np.percentile(a,95),reflection_p95=np.percentile(g,95),
            old_res_max=rec['res_max'],algebraic_max=np.max(a),reflection_max=np.max(g)))
        corrected.append(dict(state=rec['state'],freq=rec['freq'],n=int(mask.sum()),
            **{f'reflection_{k}':v for k,v in statistics(g).items() if k!='rows'},
            algebraic_median=np.median(a),algebraic_p95=np.percentile(a,95),algebraic_max=np.max(a),
            abs_a=abs(full.coefficients[int(rec['state']),np.searchsorted(full.frequencies,rec['freq']),0]),
            abs_b=abs(full.coefficients[int(rec['state']),np.searchsorted(full.frequencies,rec['freq']),1]),
            abs_c=abs(full.coefficients[int(rec['state']),np.searchsorted(full.frequencies,rec['freq']),2])))
    ad=csv('T17_original_residual_audit.csv',audit)
    csv('T17_full_data_corrected_IN_SAMPLE.csv',corrected)
    checks={k:float(np.max(np.abs(ad['old_res_'+k]-ad['algebraic_'+v])))
            for k,v in [('med','median'),('p95','p95'),('max','max')]}
    assert max(checks.values())<1e-10,checks
    csv('T17_full_data_row_summary_IN_SAMPLE.csv',[
        dict(residual='reflection_prediction',**statistics(eg)),dict(residual='algebraic',**statistics(ea))])
    print('T17 verified: supplied residuals are algebraic, maximum audit difference',checks,flush=True)

    quality=[];cells=[];teachers=[];perclass=[];splitchecks=[];rowsets=[]
    original={p:pd.read_csv(SRC/f'original_{p}_runs.csv').set_index('seed') for p in ['grouped','random']}
    for sd in sorted((SRC/'splits').iterdir()):
        run=sd.name;meta=json.loads((sd/'metadata.json').read_text(encoding='utf-8'))
        protocol=meta['protocol'];seed=meta['seed']
        with np.load(sd/'split.npz') as zz: ix={k:zz[k].astype(np.int64) for k in ['train','valid','test']}
        concat=np.concatenate(list(ix.values()))
        assert np.array_equal(np.sort(concat),all_ids)
        intersections={a+'_'+b:len(np.intersect1d(bundle.sweep_groups[ix[a]],bundle.sweep_groups[ix[b]]))
            for a,b in [('train','valid'),('train','test'),('valid','test')]}
        if protocol=='sweep_group':assert sum(intersections.values())==0
        splitchecks.append(dict(run=run,train=len(ix['train']),valid=len(ix['valid']),test=len(ix['test']),
            all_rows_partition=True,**{f'group_overlap_{k}':v for k,v in intersections.items()}))
        core.set_seed(seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            teacher=core.PhysicsTeacher.fit(raw,y,ix['train'],kind='lr')
        warningtexts=[str(w.message) for w in caught]
        assert not any(issubclass(w.category,ConvergenceWarning) for w in caught),warningtexts
        phys=teacher.physics
        cellrows=cell_diagnostics(phys,raw,ix['train'],run);cells.extend(cellrows)
        assert all(c['rank']==3 and not c['fallback'] for c in cellrows)
        np.savez_compressed(ROOT/'models'/f'{run}_teacher.npz',frequencies=phys.frequencies,
            mobius_coefficients=phys.coefficients,scaler_mean=teacher.scaler.mean,scaler_scale=teacher.scaler.scale,
            lr_coef=teacher.model.coef_,lr_intercept=teacher.model.intercept_,lr_classes=teacher.model.classes_)
        log=original['grouped' if protocol=='sweep_group' else 'random'].loc[seed]
        for subset,ids in ix.items():
            e,al,pred,den=physical_errors(phys,raw,ids)
            assert np.isfinite(e).all()
            for band,mask in [('all',np.ones(len(ids),bool)),('703-748',raw[ids,2]<800),('824-849',raw[ids,2]>=800)]:
                quality.append(dict(run=run,protocol=protocol,seed=seed,subset=subset,band=band,
                    **statistics(e[mask]),predicted_abs_gt1=int((np.abs(pred[mask])>1).sum()),
                    minimum_denominator_abs=float(den[mask].min())))
            yp=teacher.predict_raw(raw[ids]);acc=accuracy_score(y[ids],yp)*100;ba=balanced_accuracy_score(y[ids],yp)*100
            record=dict(run=run,protocol=protocol,seed=seed,subset=subset,rows=len(ids),teacher_acc_pct=acc,
                teacher_ba_pct=ba,solver_iterations=int(teacher.model.n_iter_.max()),warnings=' | '.join(warningtexts))
            if subset in ['train','valid']:
                key='physics_teacher_train_accuracy' if subset=='train' else 'physics_teacher_validation_accuracy'
                record['original_logged_acc_pct']=float(log[key]*100)
                record['difference_from_log_pp']=acc-record['original_logged_acc_pct']
                assert abs(record['difference_from_log_pp'])<0.05,record
            if subset=='test':
                record['student_fp32_acc_pct']=meta['fp32_holdout']['accuracy']*100
                record['student_fp32_ba_pct']=meta['fp32_holdout']['balanced_accuracy']*100
                record['student_minus_teacher_acc_pp']=record['student_fp32_acc_pct']-acc
                record['student_minus_teacher_ba_pp']=record['student_fp32_ba_pct']-ba
                cm=confusion_matrix(y[ids],yp,labels=np.arange(15))
                pd.DataFrame(cm,index=np.arange(15),columns=np.arange(15)).to_csv(ROOT/'raw'/f'{run}_teacher_confusion.csv')
                for k in range(15):perclass.append(dict(run=run,protocol=protocol,class_id=k,support=int(cm[k].sum()),correct=int(cm[k,k]),recall_pct=100*cm[k,k]/cm[k].sum()))
                rf=pd.DataFrame(dict(run=run,protocol=protocol,row_id=ids,group_id=bundle.sweep_groups[ids],
                    frequency_MHz=raw[ids,2].astype(int),state=raw[ids,3].astype(int),label=y[ids],teacher_pred=yp,
                    gamma1_re=raw[ids,0]/32768,gamma1_im=raw[ids,1]/32768,
                    predicted_gamma1_re=pred.real,predicted_gamma1_im=pred.imag,
                    reflection_error=e,algebraic_residual=al,denominator_abs=den))
                rf.to_csv(ROOT/'raw'/f'{run}_test_rows.csv.gz',index=False,compression='gzip',float_format='%.12g')
                rowsets.append(rf)
            teachers.append(record)
        print(run,'test teacher Acc/BA',round(acc,5),round(ba,5), 'min train cell',min(c['training_rows'] for c in cellrows),flush=True)
    q=csv('C1_01_reflection_error_per_fold.csv',quality)
    c=csv('C1_02_training_cell_diagnostics.csv',cells)
    t=csv('C1_03_teacher_metrics_per_fold.csv',teachers)
    csv('C1_04_teacher_per_class.csv',perclass)
    csv('C1_05_split_audit.csv',splitchecks)
    summary=[]
    for (protocol,subset),g in t.groupby(['protocol','subset']):
        for metric in ['teacher_acc_pct','teacher_ba_pct','student_fp32_acc_pct','student_fp32_ba_pct','student_minus_teacher_acc_pp','student_minus_teacher_ba_pp']:
            vals=g[metric].dropna()
            if len(vals):summary.append(dict(protocol=protocol,subset=subset,metric=metric,runs=len(vals),mean=vals.mean(),sample_sd=vals.std(ddof=1)))
    csv('C1_06_teacher_fold_summary.csv',summary)
    rr=pd.concat(rowsets,ignore_index=True)
    grouped=rr[rr.protocol=='sweep_group']
    assert len(grouped)==n and np.array_equal(np.sort(grouped.row_id),all_ids)
    pooled=[]
    for prot,g in rr.groupby('protocol'):
        for band,mask in [('all',np.ones(len(g),bool)),('703-748',g.frequency_MHz.to_numpy()<800),('824-849',g.frequency_MHz.to_numpy()>=800)]:
            z=g.iloc[np.flatnonzero(mask)]
            pooled.append(dict(protocol=prot,band=band,aggregation='unique OOF rows' if prot=='sweep_group' else 'repeated holdout evaluations (rows may repeat)',
                **statistics(z.reflection_error.to_numpy()),teacher_acc_pct=accuracy_score(z.label,z.teacher_pred)*100,
                teacher_ba_pct=balanced_accuracy_score(z.label,z.teacher_pred)*100))
    csv('C1_07_test_pooled_by_band.csv',pooled)
    heldcells=[]
    for (run,f,s),g in rr.groupby(['run','frequency_MHz','state']):
        heldcells.append(dict(run=run,frequency_MHz=f,state=s,**statistics(g.reflection_error.to_numpy())))
    csv('C1_08_test_error_per_cell.csv',heldcells)
    checks.update(grouped_test_unique_rows=int(grouped.row_id.nunique()),
        min_train_cell_rows=int(c.training_rows.min()),rank_deficient_cells=int((c['rank']<3).sum()),
        fallback_cells=int(c.fallback.sum()),max_teacher_acc_log_difference_pp=float(t.difference_from_log_pp.abs().max()),
        full_data_true_reflection_error=statistics(eg),full_data_algebraic_residual=statistics(ea))
    (ROOT/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    # Scientific figures, with English labels and no unsupported RF-unit conversion.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for label,mask in [('703–748 MHz',grouped.frequency_MHz<800),('824–849 MHz',grouped.frequency_MHz>=800)]:
        v=np.sort(grouped.loc[mask,'reflection_error'].to_numpy());ax[0].plot(v,np.arange(1,len(v)+1)/len(v),label=label)
    ax[0].set(xlabel='Held-out reflection prediction error |M(Γ₂) − Γ₁|',ylabel='Empirical CDF',xlim=(0,np.quantile(grouped.reflection_error,.995)))
    ax[0].legend();ax[0].grid(alpha=.25)
    gt=t[(t.protocol=='sweep_group')&(t.subset=='test')].sort_values('seed');xx=np.arange(len(gt))
    ax[1].plot(xx,gt.teacher_acc_pct,'o-',label='Teacher Acc');ax[1].plot(xx,gt.teacher_ba_pct,'s-',label='Teacher BA')
    ax[1].plot(xx,gt.student_fp32_acc_pct,'o--',label='Student FP32 Acc');ax[1].plot(xx,gt.student_fp32_ba_pct,'s--',label='Student FP32 BA')
    ax[1].set(xticks=xx,xticklabels=[str(i+1) for i in xx],xlabel='Grouped fold',ylabel='Percent');ax[1].legend(fontsize=8);ax[1].grid(alpha=.25)
    fig.savefig(ROOT/'figures'/'C1_heldout_validation.png',dpi=220);fig.savefig(ROOT/'figures'/'C1_heldout_validation.pdf');plt.close(fig)
    print('COMPLETED',ROOT,flush=True)

if __name__=='__main__':main()
