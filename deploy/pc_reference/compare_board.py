"""Compare PC reference against board CSV containing row_id,board_pred."""
import argparse,csv,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--pc',required=True,type=Path);p.add_argument('--board',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
def read(path,predkey):
 result={}
 with path.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   k=int(r['row_id']);v=int(r[predkey])
   if k in result:raise ValueError(f'duplicate row_id {k}: {path}')
   if not 0<=v<15:raise ValueError(f'invalid class {v}: {path}')
   result[k]=r
 return result
pc=read(a.pc,'pc_pred');bd=read(a.board,'board_pred')
if set(pc)!=set(bd):raise ValueError(f'row IDs differ: missing={len(set(pc)-set(bd))}, extra={len(set(bd)-set(pc))}')
if set(pc)!=set(range(96768)):raise ValueError('expected row_id 0..96767')
a.out.mkdir(parents=True,exist_ok=True)
diff=[];correct=0
for i in sorted(pc):
 r=pc[i];b=bd[i];v=int(b['board_pred']);label=int(r['label']);correct+=v==label
 for col in ['g1Re','g1Im','freq_MHz','g2Re','g2Im','label']:
  if col in b and int(b[col])!=int(r[col]):raise ValueError(f'input/label mismatch at {i}: {col}')
 if v!=int(r['pc_pred']):diff.append([i,label,r['pc_pred'],v])
with (a.out/'disagreements.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['row_id','label','pc_pred','board_pred']);w.writerows(diff)
r={'rows':len(pc),'disagreements':len(diff),'agreement_percent':100*(len(pc)-len(diff))/len(pc),'board_correct':correct,'board_accuracy_percent':100*correct/len(pc),'scope':'Top-1 agreement only; not numerical score equivalence'}
(a.out/'comparison.json').write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r,indent=2))
