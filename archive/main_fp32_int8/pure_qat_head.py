"""Gradient-only QAT on the classifier, using a pre-discrete QAT snapshot.

The frozen backbone executes exactly in integers. Trainable real-valued head
parameters pass through INT8/INT32 fake quantization, then Adam updates a smooth
CE/focal loss. Accuracy only selects checkpoints; no coordinate/threshold search.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from strict_int8_verify import IntegerModel,ROOT

def rnd(x):return x+(torch.floor(x+.5)-x).detach()
class Head(nn.Module):
    def __init__(self,model):
        super().__init__();w,b,m=model.layers[-1]
        self.weight=nn.Parameter(torch.tensor(w,dtype=torch.float32,device='cuda'))
        self.bias=nn.Parameter(torch.tensor(b,dtype=torch.float32,device='cuda'))
        self.register_buffer('mult',torch.tensor(m,dtype=torch.float64,device='cuda'))
    def forward(self,x):
        acc=F.linear(x,rnd(self.weight).clamp(-127,127),rnd(self.bias))
        return rnd(acc.double()*self.mult/(1<<24)).float()
    def export(self,source):
        data=bytearray(source);start=len(data)-604
        data[start:start+480]=torch.floor(self.weight.detach()+.5).clamp(-127,127).cpu().numpy().astype('i1').tobytes()
        data[start+480:start+540]=torch.floor(self.bias.detach()+.5).cpu().numpy().astype('<i4').tobytes()
        return bytes(data)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=600);ap.add_argument('--lr',type=float,default=.03);ap.add_argument('--focal',type=float,default=0);ap.add_argument('--temperature',type=float,default=1);ap.add_argument('--batch',type=int,default=96768);ap.add_argument('--run',default='head_ce');a=ap.parse_args()
    torch.set_num_threads(6);torch.manual_seed(20260908)
    source=ROOT/'model/strict_int8_v5_focal/model_integer.bin'
    # This exact baseline predates any discrete search: 95,692 / 96,768 correct.
    blob=source.read_bytes();assert hashlib.sha256(blob).hexdigest()=='07f8ff773e70fb92070f6e205987f546860ceceea0371efacaadbdea8b85de54'
    out=ROOT/'model/pure_qat'/a.run;out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps({**vars(a),'source_sha256':hashlib.sha256(blob).hexdigest(),'method':'gradient-only QAT; no discrete or bias search'},indent=2))
    (out/'training_source.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    m=IntegerModel(source)
    raw=pd.read_csv(ROOT.parent/'esp32p4_iat_demo/input/data/x_train.csv',encoding='gbk').values[:,[0,1,2,4,5]]
    y_np=pd.read_csv(ROOT.parent/'esp32p4_iat_demo/input/data/y_train.csv',encoding='gbk').values.ravel()
    feat=np.concatenate([m.predict(b,return_features=True) for b in np.array_split(raw,24)]).astype(np.int32)+128
    x=torch.tensor(feat,dtype=torch.float32,device='cuda');y=torch.tensor(y_np,device='cuda')
    net=Head(m);opt=torch.optim.Adam([{'params':[net.weight],'lr':a.lr},{'params':[net.bias],'lr':a.lr*64}])
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:.02+.98*(1+np.cos(np.pi*e/a.epochs))/2)
    history=[];best=-1;start=time.time()
    for epoch in range(a.epochs+1):
        if epoch:
            indices=torch.randperm(len(x),device='cuda') if a.batch<len(x) else torch.arange(len(x),device='cuda')
            for idx in indices.split(a.batch):
                opt.zero_grad(set_to_none=True);logits=net(x[idx])*.0002/a.temperature
                ce=F.cross_entropy(logits,y[idx],reduction='none');loss=(ce*(1-ce.neg().exp())**a.focal).mean();loss.backward();opt.step()
            sched.step()
        with torch.no_grad():
            correct=int((net(x).argmax(1)==y).sum())
        if correct>best:
            best=correct;(out/'model_int8.bin').write_bytes(net.export(blob))
            torch.save({'state':net.state_dict(),'epoch':epoch,'correct':correct},out/'best.pt')
        history.append({'epoch':epoch,'correct':correct,'accuracy':correct/len(y)})
        if epoch%50==0:print(a.run,epoch,correct,'best',best,flush=True)
    actual=IntegerModel(out/'model_int8.bin');pred=np.concatenate([actual.predict(b)[0] for b in np.array_split(raw,24)])
    correct=int((pred==y_np).sum());assert correct==best,(correct,best)
    report={'correct':correct,'total':len(y_np),'accuracy':correct/len(y_np),'bytes':(out/'model_int8.bin').stat().st_size,'seconds':time.time()-start,'no_discrete_optimization':True,'scope':'full training set fit','sha256':hashlib.sha256((out/'model_int8.bin').read_bytes()).hexdigest()}
    (out/'history.json').write_text(json.dumps(history,indent=2));(out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
