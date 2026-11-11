"""CPU FP32 inference. Run without arguments for full dataset evaluation."""
import time
import numpy as np
import joblib
import torch
from pc_common import arguments,inputs,finish
def main():
 a=arguments('fp32')
 if a.batch_size<1:raise ValueError('batch size must be positive')
 torch.set_num_threads(6)
 with a.model.open('rb') as f:model=torch.jit.load(f,map_location='cpu').eval()
 scaler=joblib.load(a.model.parent/'scaler.pkl')
 x,y=inputs(a);x=np.asarray(x,dtype=np.float64)
 x=torch.from_numpy(((x-scaler['mean'])/scaler['scale']).astype(np.float32))
 start=time.perf_counter()
 with torch.inference_mode():pred=torch.cat([model(b).argmax(1) for b in x.split(a.batch_size)]).tolist()
 finish(a,pred,y,time.perf_counter()-start)
if __name__=='__main__':main()
