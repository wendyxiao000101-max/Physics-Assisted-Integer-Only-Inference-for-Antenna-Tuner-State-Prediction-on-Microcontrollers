"""Batched NumPy integer reference kernel; model binary and arithmetic unchanged."""
from pathlib import Path
import numpy as np
class Predictor:
    def __init__(self,path):
        data=Path(path).read_bytes()
        if len(data)!=4656 or data[:8]!=b'I8R32V02':raise ValueError('invalid model')
        pos=8
        def get(n,dtype):
            nonlocal pos
            a=np.frombuffer(data,dtype=dtype,count=n,offset=pos).copy();pos+=a.nbytes;return a
        self.offset=get(5,'<i4');self.step=get(5,'<i4');self.hz=int(get(1,'<i4')[0]);self.lut=get(1024,'i1');self.layers=[]
        for ni,no in [(5,32),(32,32),(32,32),(32,15)]:
            self.layers.append((get(ni*no,'i1').reshape(no,ni).astype(np.int32).T.copy(),get(no,'<i4'),get(no,'<i4').astype(np.int64)))
        self.skip=int(get(1,'<i4')[0])
    def _batch(self,raw):
        def rq(v):return (v+(1<<23))>>24
        def linear(q,i,zp):
            w,b,m=self.layers[i];acc=(q.astype(np.int32)-zp)@w+b
            return acc.astype(np.int64)*m
        x=(np.clip((2*(raw-self.offset)+self.step)//(2*self.step),0,255)-128).astype(np.int8)
        h=self.lut[np.clip(rq(linear(x,0,-128)),-512,511)+512]
        a=(np.clip(rq(linear(h,1,self.hz-128)),0,255)-128).astype(np.int8)
        z=(np.clip(rq(linear(a,2,-128)+(h.astype(np.int64)-self.hz+128)*self.skip),0,255)-128).astype(np.int8)
        return rq(linear(z,3,-128)).argmax(1)
    def predict_batch(self,features,batch_size=4096):
        raw=np.asarray(features)
        if raw.ndim!=2 or raw.shape[1]!=5 or raw.dtype.kind not in 'iu':raise ValueError('expected Nx5 integer inputs')
        if batch_size<1:raise ValueError('batch_size must be positive')
        # Dataset inputs are Q15-like integers; guard before widened preprocessing.
        if raw.size and (raw.min() < -(1<<31) or raw.max() >= (1<<31)):raise ValueError('inputs must fit int32')
        raw=raw.astype(np.int64)
        return np.concatenate([self._batch(raw[i:i+batch_size]) for i in range(0,len(raw),batch_size)]) if len(raw) else np.empty(0,dtype=np.int64)
    def predict(self,features):
        return int(self.predict_batch([features])[0])
