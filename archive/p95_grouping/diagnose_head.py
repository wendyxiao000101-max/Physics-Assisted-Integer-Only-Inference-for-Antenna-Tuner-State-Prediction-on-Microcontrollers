from pathlib import Path
import sys,json
import numpy as np,torch
O=Path(__file__).resolve().parent;R=O.parents[1];sys.path.insert(0,str(O/'source_snapshot'))
from iatopt.quantization import load_fp32_teacher,cross_layer_equalize_block,FoldedResidual32
from iatopt.core import load_dataset
torch.set_num_threads(4)
b=load_dataset(R/'esp32p4_iat_demo/input/data/x_train.csv',R/'esp32p4_iat_demo/input/data/y_train.csv')
reports=[]
for p in sorted((O/'models').glob('sweep_group_seed*')):
    teacher,scaler,meta=load_fp32_teacher(p);h=teacher.network[-1]
    m=FoldedResidual32(5,32,h.out_features,20)
    with torch.no_grad():
        m.stem.weight.copy_(teacher.network[0].weight);m.stem.bias.copy_(teacher.network[0].bias)
        block=teacher.network[2].block
        for dest,src in [(m.block_fc1,block[0]),(m.block_fc2,block[3])]:dest.weight.copy_(src.weight);dest.bias.copy_(src.bias)
        m.head.weight.copy_(torch.nn.functional.normalize(h.weight,dim=1))
        cross_layer_equalize_block(m);m.eval();x=torch.tensor(scaler.transform(b.raw[:,[0,1,2,4,5]]),dtype=torch.float32)
        a=teacher(x).argmax(1).numpy();c=m(x).argmax(1).numpy()
    split=np.load(p/'split.npz');row=dict(seed=meta['seed'],alpha=float(h.alpha.detach()),gate_pass=bool(float(h.alpha.detach())>=.999),diagnostic_only=True)
    for k in ['train','valid','test']:
        idx=split[k];row[k+'_changes']=int((a[idx]!=c[idx]).sum());row[k+'_n']=len(idx);row[k+'_accuracy_delta_pp']=float(((c[idx]==b.labels[idx]).mean()-(a[idx]==b.labels[idx]).mean())*100)
    reports.append(row)
(O/'head_gate_diagnostics.json').write_text(json.dumps(reports,indent=2),encoding='utf8');print(json.dumps(reports,indent=2))
