import json,sys,time,subprocess
from pathlib import Path
repo=Path('/scratch/yy5919/pyro-revision15-formal');sys.path.insert(0,str(repo))
from src.experiments.config import load_config
from src.experiments.matrix import ablation5_runs
from src.experiments.io import read_jsonl
from src.experiments.generation_stage import moments_path,generations_path
cfg=load_config(repo/'configs/experiments/reframed_test_local.toml')
exp=Path(cfg['paths']['output_root'])/cfg['experiment']['name']
runs=[r for r in ablation5_runs(cfg) if not r.timeline]
root=Path('/scratch/yy5919/pyro-kd-derived/tts/formal-local-v1')
voice='/scratch/yy5919/pyro-kd-derived/models/piper-ljspeech-medium'
print(json.dumps({'watching':[r.run_id for r in runs],'cache':'/scratch/yy5919/pyro-kd-derived/cache/reframed-test','device':'cpu'}),flush=True)
done=set();deadline=time.monotonic()+21600
while len(done)<len(runs) and time.monotonic()<deadline:
 for run in runs:
  if run.run_id in done:continue
  expected={r['id'] for r in read_jsonl(moments_path(exp,run.selection))}
  source=generations_path(exp,run);rows=read_jsonl(source)
  if not expected or not expected<={r['id'] for r in rows}:continue
  out=root/run.run_id
  rendered=out/'rendered';overlap=out/'overlap'
  if not (rendered/'report.json').exists():
   subprocess.run(['/scratch/yy5919/pyro-tts-env/bin/python','scripts/render_ad_piper.py','--input',str(source),'--voice-dir',voice,'--output-dir',str(rendered)],check=True,cwd=repo)
  if not (overlap/'report.json').exists():
   subprocess.run([sys.executable,'scripts/evaluate_rendered_overlap.py','--input',str(rendered/'rendered.jsonl'),'--cache-dir','/scratch/yy5919/pyro-kd-derived/cache/reframed-test','--output-dir',str(overlap)],check=True,cwd=repo)
  done.add(run.run_id);print(json.dumps({'finished':run.run_id,'cells':len(done)}),flush=True)
 if len(done)<len(runs):time.sleep(20)
print(json.dumps({'complete':len(done)==len(runs),'cells':len(done)}),flush=True)
