from pathlib import Path
import sys,numpy as np
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'source_snapshot'))
import train
def choose(bundle,protocol,run,seed):
    assert protocol=='sweep_group'
    z=np.load(OUT/'splits'/f'fold{run}.npz')
    return z['train'],z['valid'],z['test']
train.choose_split=choose
train.main()
