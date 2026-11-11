from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,sys,json,os,time,threading
O=Path(__file__).resolve().parent;R=O.parents[1]
base=json.loads((O.parent/'C7补充验证_20260925/fp32_command.json').read_text(encoding='utf8'))[2:]
start=base.index('--seeds');end=base.index('--x');base[start:end]=[]
def option(cmd,key,val):
 if key in cmd:cmd[cmd.index(key)+1]=str(val)
 else:cmd.extend([key,str(val)])
variants=json.loads((O/'variants.json').read_text());lock=threading.Lock();state={};starttime=time.time()
def save():
 (O/'run_status.json').write_text(json.dumps({'start':starttime,'elapsed_seconds':time.time()-starttime,'jobs':state},indent=2),encoding='utf8')
def run(name,fold):
 key=f'{name}/fold{fold}';out=O/'runs'/name/f'fold{fold}';out.mkdir(parents=True,exist_ok=True)
 done=out/'COMPLETE.json'
 if done.exists():
  with lock:state[key]={'state':'complete','reused_completed_run':True};save()
  return
 cmd=[sys.executable,'-u',str(O/'run_fold.py'),*base]
 for k,v in variants[name].items():option(cmd,k,v)
 for k,v in {'--output':out,'--ablate-export-dir':out/'models','--sweep-fold':fold,'--teacher-cache-dir':O/'teacher_cache'}.items():option(cmd,k,v)
 cmd+=['--seeds',str(42+fold)]
 (out/'command.json').write_text(json.dumps(cmd,ensure_ascii=False,indent=2),encoding='utf8')
 with lock:state[key]={'state':'running','start':time.time()};save();print('START',key,flush=True)
 env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['OMP_NUM_THREADS']='6';env['MKL_NUM_THREADS']='6'
 with (out/'train.log').open('w',encoding='utf8') as f:
  proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=str(R))
  with lock:state[key]['pid']=proc.pid;save()
  rc=proc.wait()
 elapsed=time.time()-state[key]['start']
 with lock:
  state[key].update(state='complete' if rc==0 else 'failed',seconds=elapsed,exit_code=rc);save();print('DONE',key,'code',rc,'seconds',round(elapsed),flush=True)
 if rc:raise RuntimeError(f'{key} failed; see {out}/train.log')
 done.write_text(json.dumps({'seconds':elapsed,'returncode':0}),encoding='utf8')
if __name__=='__main__':
 # Each process has its own RNG. Three independent jobs fit well within GPU memory.
 for names in [['baseline'],[k for k in variants if k!='baseline']]:
  with ThreadPoolExecutor(max_workers=3) as pool:
   jobs=[pool.submit(run,n,f) for n in names for f in range(5)]
   for j in as_completed(jobs):j.result()
 (O/'TRAINING_COMPLETED.json').write_text(json.dumps({'runs':45,'seconds':time.time()-starttime}),encoding='utf8')
 print('ALL 45 RUNS COMPLETE',flush=True)
