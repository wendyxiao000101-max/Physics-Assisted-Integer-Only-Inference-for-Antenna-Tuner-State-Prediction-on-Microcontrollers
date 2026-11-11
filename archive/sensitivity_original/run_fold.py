from pathlib import Path
import sys,json,numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'source_snapshot'))
import train
from iatopt import core
def choose(bundle,protocol,run,seed):
 assert protocol=='sweep_group' and run==seed-42
 z=np.load(O/'splits'/f'fold{run}.npz')
 group_sets=[set(bundle.sweep_groups[z[k]]) for k in ['train','valid','test']]
 assert not any(group_sets[i]&group_sets[j] for i in range(3) for j in range(i))
 return z['train'],z['valid'],z['test']
train.choose_split=choose
original=train.build_physics_augmentation
def traced(*args,**kwargs):
 result=original(*args,**kwargs)
 out=Path(sys.argv[sys.argv.index('--output')+1]);(out/'synthesis_audit.json').write_text(json.dumps(core.LAST_SYNTHESIS_AUDIT,indent=2),encoding='utf8')
 return result
train.build_physics_augmentation=traced
train.main()
