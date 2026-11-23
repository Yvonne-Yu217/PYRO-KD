"""Independent prepared tasks on one additional GPU; exit instead of idling on data."""
import fcntl,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
longrepo=Path('/scratch/yy5919/pyro-revision15-long-prep');formal=Path('/scratch/yy5919/pyro-revision15-formal')
root=Path('/scratch/yy5919/pyro-resume-results');job=os.environ['SLURM_JOB_ID'];report=root/f'secondary-gpu-{job}.json'
model='/home/yy5919/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'
stages=[]
def save():report.write_text(json.dumps({'job_id':job,'stages':stages,'scope':'no paid API calls; separate outputs from primary GPU'},indent=2)+'\n')
def run(name,repo,args,seconds,python='python'):
 command='OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '+shlex.join([python,'-u',*args])
 shell='source hpc/config.sh; in_env ro '+shlex.quote(command)
 started=time.monotonic();print(json.dumps({'starting':name,'utc':time.strftime('%FT%TZ',time.gmtime())}),flush=True)
 result=subprocess.run(['timeout','--kill-after=30',str(seconds),'bash','-c',shell],cwd=repo)
 stages.append({'name':name,'exit_code':result.returncode,'seconds':round(time.monotonic()-started,3)});save()
 print(json.dumps(stages[-1]),flush=True)
def ready(path,n):
 if not path.exists():return False
 rows=[json.loads(x) for x in path.read_text().splitlines()];return len(rows)==len({r['id'] for r in rows})==n
with Path('/scratch/yy5919/pyro-secondary-gpu.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for condition in ['AV-flat-timeline','AV-hierarchical-timeline']:
  inp=Path('/scratch/yy5919/pyro-kd-derived/moments')/f'video-mme-long-{condition}-v1.jsonl';assert ready(inp,36)
  out=root/'long-qa-frozen'/f'{condition}-224.jsonl'
  if ready(out,36):stages.append({'name':condition,'status':'already complete; preserved'});save();continue
  audit={'condition':condition,'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),'source_commit':'a7e0e1a','job_id':job,'expected':36}
  (out.parent/f'{condition}.secondary-freeze.json').write_text(json.dumps(audit,indent=2)+'\n')
  run(condition,longrepo,['scripts/run_qa_vlm.py','--input',str(inp),'--output',str(out),'--model',model,'--max-side','224','--max-new-tokens','16'],2700)
 for label,pattern,limit,python in [('formal-no-context','ad-A_cue-P0-qwen-tl0-*',3600,'python'),('formal-native','ad-A_cue-P0-omni-*',5400,'/scratch/yy5919/pyro-audio-env/bin/python')]:
  run(label,formal,['scripts/run_experiment.py','--config','configs/experiments/reframed_test_local.toml','--stage','ablation5','--runs',pattern],limit,python)
 for condition in ['U','V_local']:
  inp=Path('/scratch/yy5919/pyro-kd-derived/moments')/f'egoschema-heldout-{condition}-v1.jsonl'
  if not ready(inp,490):
   stages.append({'name':'EgoSchema-'+condition,'status':'deferred: CPU inputs not ready; no GPU waiting'});save();continue
  out=root/'egoschema-heldout'/f'{condition}.jsonl'
  if ready(out,490):stages.append({'name':'EgoSchema-'+condition,'status':'already complete; preserved'});save();continue
  run('EgoSchema-'+condition,formal,['scripts/run_qa_vlm.py','--input',str(inp),'--output',str(out),'--model',model,'--max-side','768','--max-new-tokens','16'],5400)
save()
raise SystemExit(1 if any(s.get('exit_code',0) for s in stages) else 0)
