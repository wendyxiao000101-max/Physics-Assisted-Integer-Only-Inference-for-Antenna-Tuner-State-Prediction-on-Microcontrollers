"""Standalone integer-only inference, Python standard library only.

Input: five integer features [g1Re,g1Im,freq,g2Re,g2Im].
The model blob contains all constants. No float, torch, numpy, or scaler file.
"""
import struct
from pathlib import Path

class Predictor:
    def __init__(self, model_path):
        data=Path(model_path).read_bytes()
        if len(data)!=4656 or data[:8]!=b'I8R32V02': raise ValueError('invalid model')
        pos=8
        def read(n,code):
            nonlocal pos
            value=struct.unpack_from('<'+str(n)+code,data,pos)
            pos+=struct.calcsize('<'+str(n)+code)
            return value
        self.offset=read(5,'i'); self.step=read(5,'i'); self.hz=read(1,'i')[0]
        self.lut=read(1024,'b'); self.layers=[]
        for ni,no in [(5,32),(32,32),(32,32),(32,15)]:
            flat=read(ni*no,'b'); w=[flat[r*ni:(r+1)*ni] for r in range(no)]
            self.layers.append((w,read(no,'i'),read(no,'i')))
        self.skip=read(1,'i')[0]
        assert pos==len(data)
    def predict(self,features):
        if len(features)!=5 or any(not isinstance(v,int) for v in features):
            raise ValueError('five Python integer features required')
        def clip(v,lo,hi):return min(max(v,lo),hi)
        def rq(v):return (v+(1<<23))>>24
        def linear(q,i,zp):
            w,b,m=self.layers[i]
            return [(sum((v-zp)*ww for v,ww in zip(q,row))+bb)*mm for row,bb,mm in zip(w,b,m)]
        x=[clip((2*(v-o)+s)//(2*s),0,255)-128 for v,o,s in zip(features,self.offset,self.step)]
        h=[self.lut[clip(rq(v),-512,511)+512] for v in linear(x,0,-128)]
        a=[clip(rq(v),0,255)-128 for v in linear(h,1,self.hz-128)]
        z=[clip(rq(v+(hh-self.hz+128)*self.skip),0,255)-128 for v,hh in zip(linear(a,2,-128),h)]
        scores=[rq(v) for v in linear(z,3,-128)]
        return max(range(15),key=scores.__getitem__)
