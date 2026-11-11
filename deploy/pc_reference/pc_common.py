from pathlib import Path
import argparse,csv,json,hashlib
ROOT=Path(__file__).resolve().parent
def arguments(kind):
 p=argparse.ArgumentParser(description=kind+' PC inference; default: full dataset')
 p.add_argument('--input',nargs='+',type=int,help='5 selected or 6 raw integer features')
 p.add_argument('--x',type=Path,default=ROOT.parents[1]/'data/x_train.csv')
 p.add_argument('--y',type=Path,default=ROOT.parents[1]/'data/y_train.csv')
 p.add_argument('--no-labels',action='store_true')
 p.add_argument('--output',type=Path)
 p.add_argument('--model',type=Path,default=ROOT/'models'/kind/('model_int8.bin' if kind=='int8' else 'model_fp32_folded.pt'))
 p.add_argument('--batch-size',type=int,default=4096)
 a=p.parse_args()
 if a.output is None:a.output=ROOT/'results'/kind/('single' if a.input is not None else 'full')
 return a
def selected(row):
 if len(row)==6:return [row[i] for i in [0,1,2,4,5]]
 if len(row)==5:return row
 raise ValueError('expected 5 selected or 6 raw features')
def rows(path):
 # The supplied CSV files have one header row, encoded in GBK.
 with path.open(encoding='gbk',newline='') as f:
  reader=csv.reader(f);next(reader)
  return [[int(v) for v in row] for row in reader if row]
def inputs(a):
 if a.input is not None:return [selected(a.input)],None
 x=[selected(v) for v in rows(a.x)]
 y=None if a.no_labels else [v[0] for v in rows(a.y)]
 if y is not None and len(x)!=len(y):raise ValueError('X/Y row count mismatch')
 return x,y
def finish(a,pred,y,seconds):
 report={'rows':len(pred),'seconds':seconds,'model_bytes':a.model.stat().st_size,'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest()}
 if y is not None:
  correct=sum(int(p==t) for p,t in zip(pred,y));report.update(correct=correct,errors=len(y)-correct,accuracy_percent=100*correct/len(y),scope='full training-set refit, not held-out test accuracy')
 a.output.mkdir(parents=True,exist_ok=True)
 with (a.output/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['row_index','prediction','label','correct'])
  for i,p in enumerate(pred):w.writerow([i,p,'' if y is None else y[i],'' if y is None else int(p==y[i])])
 (a.output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False,indent=2))
 if a.input is not None:print('Predicted class:',pred[0])
 print('Results:',a.output)
