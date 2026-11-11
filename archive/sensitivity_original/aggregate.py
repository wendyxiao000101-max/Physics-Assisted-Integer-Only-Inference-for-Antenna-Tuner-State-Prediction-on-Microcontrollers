"""Summarize paired, one-factor-at-a-time FP32 sensitivity runs; never edit the paper."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

O=Path(__file__).resolve().parent
variants=json.loads((O/'variants.json').read_text())
rows=[]; class_rows=[]
for variant in variants:
    for fold in range(5):
        root=O/'runs'/variant/f'fold{fold}'
        if not (root/'COMPLETE.json').exists(): continue
        paths=list((root/'ablation/runs').glob('*.json'))
        assert len(paths)==1, (root, paths)
        detail=json.loads(paths[0].read_text(encoding='utf-8-sig'))
        s=detail['summary']; cm=np.asarray(detail['test']['confusion_matrix'])
        assert cm.shape==(15,15) and cm.sum()>0 and (cm.sum(1)>0).all()
        calc={'accuracy':np.trace(cm)/cm.sum(), 'balanced_accuracy':np.mean(cm.diagonal()/cm.sum(1)),
              'minimum_class_recall':np.min(cm.diagonal()/cm.sum(1))}
        for metric, value in calc.items(): assert abs(s['test_'+metric]-value)<1e-12
        for cls in range(15):
            class_rows.append(dict(variant=variant,fold=fold,seed=42+fold,label=cls,
                test_count=int(cm[cls].sum()),correct=int(cm[cls,cls]),recall=100*cm[cls,cls]/cm[cls].sum()))
        syn=json.loads((root/'synthesis_audit.json').read_text())
        old=json.loads((O.parent/f'C1补充验证_20260918/sources/splits/sweep_group_seed{42+fold}/metadata.json').read_text())
        rows.append(dict(variant=variant,fold=fold,seed=42+fold,
                         acc=100*calc['accuracy'], ba=100*calc['balanced_accuracy'], mr=100*calc['minimum_class_recall'],
                         historical_acc=100*old['fp32_holdout']['accuracy'],
                         historical_ba=100*old['fp32_holdout']['balanced_accuracy'],
                         historical_mr=100*old['fp32_holdout']['minimum_class_recall'],
                         gate=s['physics_min_distance'],sigma=s['physics_sigma'],decay=s['physics_decay_end'],scale=s['head_scale'],
                         hybrid_alpha=s['hybrid_alpha'],epochs=s['epochs_ran'],best_epoch=s['best_epoch'],
                         checkpoint_source=s['best_weight_source'],train_seconds=s['train_seconds'],
                         train_rows=old['train_rows'],valid_rows=old['valid_rows'],test_rows=int(cm.sum()),
                         **{k:v for k,v in syn.items() if k!='sigma'}))
if not rows:
    print('No completed folds yet.'); sys.exit(0)
df=pd.DataFrame(rows)
baseline=df[df.variant=='baseline'].set_index('fold')
for m in ['acc','ba','mr']:
    df['delta_'+m+'_pp']=[r[m]-baseline.loc[r.fold,m] if r.fold in baseline.index else np.nan for _,r in df.iterrows()]
df.to_csv(O/'C13_per_fold.csv',index=False,encoding='utf-8-sig')
classes=pd.DataFrame(class_rows);cb=classes[classes.variant=='baseline'].set_index(['fold','label'])
classes['delta_recall_pp']=[r.recall-cb.loc[(r.fold,r.label),'recall'] if (r.fold,r.label) in cb.index else np.nan for _,r in classes.iterrows()]
classes.to_csv(O/'C13_per_class.csv',index=False,encoding='utf-8-sig')
summ=[]
for v in variants:
    d=df[df.variant==v]
    if d.empty:continue
    r=dict(variant=v,n_folds=len(d))
    for m in ['acc','ba','mr','delta_acc_pp','delta_ba_pp','delta_mr_pp','acceptance_rate','finite_rows','hybrid_alpha','epochs']:
        r[m+'_mean']=float(d[m].mean());r[m+'_sd']=float(d[m].std(ddof=1))
        r[m+'_min']=float(d[m].min());r[m+'_max']=float(d[m].max())
    summ.append(r)
summary=pd.DataFrame(summ)
summary.to_csv(O/'C13_summary.csv',index=False,encoding='utf-8-sig')
print(summary[['variant','n_folds','acc_mean','acc_sd','delta_acc_pp_mean','ba_mean','mr_mean','acceptance_rate_mean']].to_string(index=False))
if '--final' in sys.argv:
    assert len(df)==45 and (summary.n_folds==5).all(), 'Final report requires all 45 completed runs'
    (O/'AGGREGATION_VERIFIED.json').write_text(json.dumps({'runs':45,'confusion_matrix_metrics_verified':True,
        'averaging':'unweighted mean over 5 folds; sample SD with ddof=1',
        'paired_delta':'variant minus contemporaneously rerun baseline in the same fold; percentage points',
        'paper_modified':False},indent=2),encoding='utf8')
