"""CPU warm single-call latency and full-dataset throughput; excludes file I/O."""
import argparse,json,time
from pathlib import Path
import numpy as np
from fast_integer_runtime import Predictor
from pc_common import ROOT,rows,selected
def main():
    p=argparse.ArgumentParser();p.add_argument('--repetitions',type=int,default=10000);a=p.parse_args()
    if a.repetitions<1:raise ValueError('positive repetitions required')
    x=[selected(row) for row in rows(ROOT/'data/x_train.csv')]
    model=Predictor(ROOT/'models/int8/model_int8.bin')
    for _ in range(100):model.predict(x[0])
    elapsed=[]
    for i in range(a.repetitions):
        start=time.perf_counter_ns();model.predict(x[i%len(x)]);elapsed.append((time.perf_counter_ns()-start)/1e6)
    start=time.perf_counter();model.predict_batch(x);total=time.perf_counter()-start
    report={'single_mean_ms':float(np.mean(elapsed)),'single_p50_ms':float(np.median(elapsed)),'single_p95_ms':float(np.quantile(elapsed,.95)),'single_p99_ms':float(np.quantile(elapsed,.99)),'single_repetitions':a.repetitions,'full_rows':len(x),'full_seconds':total,'scope':'CPU; includes input conversion and integer preprocessing; excludes model/CSV loading; no real-time latency guarantee'}
    print(json.dumps(report,indent=2))
    (ROOT/'latency_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
