"""Reload every completed model and independently recompute its held-out predictions."""
from pathlib import Path
import sys,json,hashlib,platform
import numpy as np
import pandas as pd
import torch, sklearn, scipy
O=Path(__file__).resolve().parent;R=O.parents[1]
manifest=json.loads((O/'source_and_data_manifest.json').read_text())
for name in ['x_train.csv','y_train.csv']:
    assert hashlib.sha256((R/'esp32p4_iat_demo/input/data'/name).read_bytes()).hexdigest()==manifest[name]
for name,digest in manifest['modified_source_hashes'].items():
    assert hashlib.sha256((O/'source_snapshot'/name).read_bytes()).hexdigest()==digest
sys.path.insert(0,str(O/'source_snapshot'))
from iatopt.core import load_dataset,build_features,Standardizer,IATMLP,predict_logits,metrics
torch.set_num_threads(2);torch.set_num_interop_threads(1)
bundle=load_dataset(R/'esp32p4_iat_demo/input/data/x_train.csv',R/'esp32p4_iat_demo/input/data/y_train.csv')
x=build_features(bundle.raw,'raw5');variants=json.loads((O/'variants.json').read_text())
audits=[];paths=[]
partial='--partial' in sys.argv
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def options(cmd):
    out={};key=None
    for a in cmd[2:]:
        if a.startswith('--'):key=a;out[key]=[]
        elif key is not None:out[key].append(a)
    return out
ignored={'--output','--ablate-export-dir'}
for variant,overrides in variants.items():
    oof=np.full(len(bundle.labels),-1,dtype=np.int16)
    for fold in range(5):
        root=O/'runs'/variant/f'fold{fold}'
        if partial and not (root/'COMPLETE.json').exists():continue
        assert (root/'COMPLETE.json').exists()
        dest=root/'models'/f'sweep_group_seed{42+fold}'
        expected=np.load(O/'splits'/f'fold{fold}.npz');actual=np.load(dest/'split.npz')
        assert all(np.array_equal(expected[k],actual[k]) for k in ['train','valid','test'])
        sets=[set(bundle.sweep_groups[actual[k]]) for k in ['train','valid','test']]
        assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
        a=options(read(root/'command.json')); b=options(read(O/'runs/baseline'/f'fold{fold}/command.json'))
        diffs={k:a[k] for k in a if k not in ignored and a[k]!=b[k]}
        assert diffs=={k:[v] for k,v in overrides.items()}, (variant,fold,diffs)
        meta=read(dest/'metadata.json'); sc=Standardizer.load(dest/'scaler.pkl')
        # Train-only standardization must agree with the raw training inputs.
        fit=Standardizer.fit(x[actual['train']])
        assert np.allclose(sc.mean,fit.mean,atol=1e-12,rtol=0) and np.allclose(sc.scale,fit.scale,atol=1e-12,rtol=0)
        model=IATMLP(5,meta['architecture'],dropout=meta['dropout'],head_scale=meta['head_scale'],hybrid_alpha=meta['hybrid_alpha_init'])
        model.load_state_dict(torch.load(dest/'model_fp32.pth',map_location='cpu',weights_only=True));model.cuda().eval()
        pred=predict_logits(model,sc.transform(x[actual['test']])).argmax(1)
        detail=read(next((root/'ablation/runs').glob('*.json')))
        history=detail['history']; summary=detail['summary']
        assert len(history)==summary['epochs_ran'] and 1<=len(history)<=360
        assert all(np.isfinite(h['train_loss']) and 0<=h['valid_accuracy']<=1 for h in history)
        assert 1<=summary['best_epoch']<=len(history)
        recalculated=metrics(bundle.labels[actual['test']],pred)
        assert np.array_equal(recalculated['confusion_matrix'],detail['test']['confusion_matrix']), (variant,fold)
        assert np.all(oof[actual['test']]==-1);oof[actual['test']]=pred
        np.savez_compressed(root/'verified_test_predictions.npz',row=actual['test'],truth=bundle.labels[actual['test']],prediction=pred)
        audits.append(dict(variant=variant,fold=fold,rows=len(pred),train_only_scaler=True,exact_split_match=True,
                           group_disjoint=True,one_factor_config_verified=True,confusion_exact=True,finite_training_history=True,
                           model_sha256=hashlib.sha256((dest/'model_fp32.pth').read_bytes()).hexdigest()))
        del model
    if np.all(oof>=0):
        np.savez_compressed(O/f'oof_{variant}.npz',row=np.arange(len(oof)),truth=bundle.labels,prediction=oof)
        print('Verified all folds:',variant,flush=True)
    elif not partial:raise AssertionError('Missing out-of-fold predictions')
if not partial:assert len(audits)==45
print('Verified models:',len(audits),flush=True)
(O/('verification_partial.json' if partial else 'verification.json')).write_text(json.dumps(dict(runs=audits,python=sys.version,platform=platform.platform(),
    torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,
    cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),all_passed=True),indent=2),encoding='utf8')
