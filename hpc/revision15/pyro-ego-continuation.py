"""CPU submission controller and bounded GPU worker for unfinished frozen Ego U/V."""
import argparse,fcntl,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
REPO=Path('/scratch/yy5919/pyro-revision15-formal');COMMIT='09abb6c0512e5e97549ad4e66d3bae05a68c42c8'
sys.path.insert(0,str(REPO))
from src.generation.outputs import final_rows
from src.generation.jsonl_recovery import read_resume_rows
from src.datasets.frozen_membership import egoschema_ids
ROOT=Path('/scratch/yy5919/pyro-resume-results/egoschema-heldout')
DERIVED=Path('/scratch/yy5919/pyro-kd-derived')
CONDITIONS=['U','V_local']
MODEL='/home/yy5919/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)
def context():
 if subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()!=COMMIT:raise ValueError('formal source changed')
 config=REPO/'configs/egoschema_pilot.json';budget=REPO/'configs/reframed_splits.json';questions=DERIVED/'datasets/egoschema/questions.json'
 ids=set(egoschema_ids(json.loads(config.read_text()),'heldout'));assert len(ids)==490
 return ids,{'pilot':sha(config),'budget':sha(budget),'questions':sha(questions)}
def input_ready(condition,expected,hashes):
 path=DERIVED/'moments'/f'egoschema-heldout-{condition}-v1.jsonl';receipt=path.with_suffix('.jsonl.preparation.json')
 if not path.exists() or not receipt.exists():return False
 rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()];r=json.loads(receipt.read_text())
 if r.get('source_commit')!=COMMIT or r.get('input_sha256')!=sha(path) or r.get('source_sha256')!=hashes:raise ValueError('input receipt provenance mismatch')
 if len(rows)!=490 or len({x['id'] for x in rows})!=490 or {x['id'] for x in rows}!=expected:raise ValueError('input frozen membership mismatch')
 if any(x.get('split')!='heldout' or x.get('dataset')!='egoschema' or x.get('selection_condition')!=condition or x.get('source_sha256')!=hashes for x in rows):raise ValueError('input condition/source mismatch')
 return True

def output_complete(path,expected,condition):
 if not path.exists():return False
 rows,recovery=read_resume_rows(path)
 if recovery is not None:return False
 if any(r.get('id') not in expected or r.get('dataset')!='egoschema' or r.get('split')!='heldout' or r.get('selection_condition')!=condition or not r.get('run_fingerprint') for r in rows):raise ValueError('output membership/condition mismatch')
 if len({r['run_fingerprint'] for r in rows})>1:raise ValueError('mixed output fingerprints')
 final=final_rows(rows)
 if len({r['id'] for r in final})!=len(final):raise ValueError('duplicate successful outputs')
 return len(final)==len(expected) and {r['id'] for r in final}==expected

def job_active(job):
 return bool(subprocess.check_output(['squeue','-h','-j',str(job),'-o','%i'],text=True).strip())
def terminal(job):
 if job_active(job):return False
 rows=subprocess.check_output(['sacct','-X','-n','-P','-j',str(job),'--format=JobIDRaw,State'],text=True).splitlines()
 states=[line.split('|')[1].split()[0].rstrip('+') for line in rows if line.split('|')[0]==str(job)]
 return bool(states) and all(s in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED'} for s in states)

def worker(conditions):
 expected,hashes=context();job=os.environ['SLURM_JOB_ID'];stages=[];report=ROOT/f'continuation-gpu-{job}.json'
 with Path('/scratch/yy5919/pyro-secondary-gpu.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for condition in conditions:
   output=ROOT/(condition+'.jsonl')
   if output_complete(output,expected,condition):
    stages.append({'condition':condition,'status':'complete coverage already present; preserved'});write_json(report,{'job_id':job,'stages':stages});continue
   if not input_ready(condition,expected,hashes):raise ValueError('inputs no longer ready; exit without GPU waiting')
   inp=DERIVED/'moments'/f'egoschema-heldout-{condition}-v1.jsonl'
   args=['python','-u','scripts/run_qa_vlm.py','--input',str(inp),'--output',str(output),'--model',MODEL,'--max-side','768','--max-new-tokens','16']
   cmd='OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '+shlex.join(args)
   shell='source hpc/config.sh; in_env ro '+shlex.quote(cmd)
   started=time.monotonic();print(json.dumps({'starting':condition,'input_sha256':sha(inp),'expected':490}),flush=True)
   result=subprocess.run(['timeout','--kill-after=30','5400','bash','-c',shell],cwd=REPO)
   stages.append({'condition':condition,'command':args,'exit_code':result.returncode,'seconds':round(time.monotonic()-started,3),'coverage_complete':output_complete(output,expected,condition),'input_sha256':sha(inp)})
   write_json(report,{'job_id':job,'source_commit':COMMIT,'stages':stages})
 raise SystemExit(1 if any(s.get('exit_code',0) for s in stages) else 0)

def controller():
 expected,hashes=context();status=ROOT/'continuation-controller.json';deadline=time.monotonic()+2*3600
 with Path('/scratch/yy5919/pyro-ego-continuation-submit.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if status.exists():
   prior=json.loads(status.read_text())
   if prior.get('submitted_job_id'):
    print(json.dumps({'status':'prior continuation already registered; no duplicate submission','record':prior}),flush=True);return
  registered=subprocess.check_output(['squeue','-h','--name=pyro-ego-continue-41197','-o','%i'],text=True).strip().splitlines()
  if registered:
   write_json(status,{'status':'existing named continuation found; no duplicate submission','submitted_job_id':registered[0],'source_commit':COMMIT});return
  while time.monotonic()<deadline:
   done=terminal(41197)
   ready={c:input_ready(c,expected,hashes) for c in CONDITIONS}
   write_json(status,{'status':'waiting on CPU only','source_commit':COMMIT,'controller_job_id':os.environ.get('SLURM_JOB_ID'),'primary_secondary_job_terminal':done,'inputs_ready':ready,'api_cost_usd':0})
   if done and all(ready.values()):break
   time.sleep(30)
  else:raise TimeoutError('2h CPU bound before source job terminal and inputs ready')
  unfinished=[c for c in CONDITIONS if not output_complete(ROOT/(c+'.jsonl'),expected,c)]
  if not unfinished:
   write_json(status,{'status':'all U/V outputs already have490 expected IDs; no GPU submitted','source_commit':COMMIT});return
  # Recheck immediately before submission; the GPU worker independently locks
  # the same writer namespace and rechecks coverage before loading any model.
  if not terminal(41197):raise RuntimeError('source GPU job became active again')
  env=dict(os.environ);env.pop('SLURM_GET_USER_ENV',None);env['SLURM_EXPORT_ENV']='ALL'
  cmd=['sbatch','--parsable','--export=ALL','/scratch/yy5919/pyro-ego-continuation-gpu.sbatch',*unfinished]
  raw=subprocess.check_output(cmd,text=True,env=env).strip();job=raw.split(';')[0]
  if not job.isdigit():raise ValueError('unexpected sbatch response')
  write_json(status,{'status':'submitted bounded GPU continuation','source_commit':COMMIT,'submitted_job_id':job,'conditions':unfinished,'controller_job_id':os.environ.get('SLURM_JOB_ID'),'submit_command':cmd,'expected_gpu_minutes_per_condition':35,'max_gpu_minutes_per_condition':90,'slurm_gpu_minutes_cap':150,'api_cost_usd':0})
  print(status.read_text(),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['controller','worker','plan']);p.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=CONDITIONS);a=p.parse_args()
 if a.mode=='worker':worker(a.conditions)
 elif a.mode=='controller':controller()
 else:
  expected,hashes=context();print(json.dumps({'source_commit':COMMIT,'expected':len(expected),'source_sha256':hashes,'source_job41197_active':job_active(41197),'inputs_ready':{c:input_ready(c,expected,hashes) for c in CONDITIONS},'gpu_submission':False,'cost_usd':0}))

if __name__=='__main__':main()
