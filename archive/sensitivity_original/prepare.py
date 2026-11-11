from pathlib import Path
import json,hashlib,shutil
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];old=O.parent/'C7补充验证_20260925/source_snapshot';S=O/'source_snapshot';S.mkdir(exist_ok=True)
manifest={}
for p in old.rglob('*.py'):
 if '__pycache__' in p.parts:continue
 out=S/p.relative_to(old);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out);manifest[str(p.relative_to(old))]=hashlib.sha256(p.read_bytes()).hexdigest()
# Logging only: no change to samples, RNG, losses or checkpoint selection.
p=S/'iatopt/core.py';s=p.read_text(encoding='utf-8-sig')
a='    count = len(points) * per_point * (oversample if min_distance > 0.0 else 1)'
assert s.count(a)==1;s=s.replace(a,a+'\n    candidate_count = count')
a='    return synthetic[finite], soft[finite]';assert s.count(a)==1
s=s.replace(a,'    global LAST_SYNTHESIS_AUDIT\n    LAST_SYNTHESIS_AUDIT = dict(anchors=int(len(points)), candidates=int(candidate_count), after_gate=int(count), finite_rows=int(finite.sum()), acceptance_rate=float(finite.sum()/candidate_count), sigma=float(sigma), min_distance=float(min_distance))\n'+a)
p.write_text(s,encoding='utf8')
p=S/'train.py';s=p.read_text(encoding='utf-8-sig')
a='        history.append({"epoch": epoch, "train_loss": running/len(fit_y), "valid_accuracy": valid_accuracy})';assert s.count(a)==1
s=s.replace(a,a+'\n        if epoch == 1 or epoch % 10 == 0:\n            save_json(args.output / "live.json", dict(epoch=epoch, valid_accuracy=valid_accuracy, seconds=time.perf_counter()-started))\n            print(f"epoch {epoch}: val={valid_accuracy:.6f}", flush=True)')
p.write_text(s,encoding='utf8')
(O/'splits').mkdir(exist_ok=True);alltest=[]
for fold,seed in enumerate(range(42,47)):
 p=O.parent/f'C1补充验证_20260918/sources/splits/sweep_group_seed{seed}/split.npz';out=O/'splits'/f'fold{fold}.npz';shutil.copy2(p,out)
 z=np.load(out);sets=[set(z[k].tolist()) for k in ['train','valid','test']];assert not any(sets[i]&sets[j] for i in range(3) for j in range(i));assert len(set.union(*sets))==96768;alltest.extend(z['test'].tolist())
 manifest['split'+str(fold)]=hashlib.sha256(p.read_bytes()).hexdigest()
assert len(alltest)==96768 and len(set(alltest))==96768
for name in ['x_train.csv','y_train.csv']:
 p=R/'esp32p4_iat_demo/input/data'/name;manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
manifest['modified_source_hashes']={str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in S.rglob('*.py')}
(O/'source_and_data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
variants={'baseline':{},'gate_0025':{'--physics-min-distance':'0.025'},'gate_0075':{'--physics-min-distance':'0.075'},'sigma_003':{'--physics-sigma':'0.03'},'sigma_009':{'--physics-sigma':'0.09'},'decay_035':{'--physics-decay-end':'0.35'},'decay_100':{'--physics-decay-end':'1.0'},'scale_10':{'--head-scale':'10'},'scale_30':{'--head-scale':'30'}}
(O/'variants.json').write_text(json.dumps(variants,indent=2),encoding='utf8')
print('Prepared 9 variants x 5 archived grouped folds; source changes are logging only.')
