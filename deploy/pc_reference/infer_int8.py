"""CPU batched integer inference. No float model operations."""
import time
from fast_integer_runtime import Predictor
from pc_common import arguments,inputs,finish
def main():
 a=arguments('int8');model=Predictor(a.model);x,y=inputs(a)
 start=time.perf_counter();pred=model.predict_batch(x,a.batch_size).tolist()
 finish(a,pred,y,time.perf_counter()-start)
if __name__=='__main__':main()
