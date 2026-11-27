import json,time,subprocess,sys,shutil
from pathlib import Path
repo=Path('/scratch/yy5919/pyro-revision15-records');sys.path.insert(0,str(repo))
from src.experiments.config import load_config
from src.experiments.matrix import core_generation_runs
cfgpath=Path('/scratch/yy5919/pyro-revision15-formal/configs/experiments/reframed_test_local.toml');cfg=load_config(cfgpath)
exp=Path('/scratch/yy5919/pyro-resume-results/experiments/reframed_test_local_v1');tts=Path('/scratch/yy5919/pyro-kd-derived/tts/formal-local-v1');root=Path('/scratch/yy5919/pyro-resume-results')
runs=[r for r in core_generation_runs(cfg) if r.model_label=='qwen'];deadline=time.monotonic()+10800
while time.monotonic()<deadline:
 if all((exp/'eval'/f'generation_{r.run_id}.json').exists() and (tts/r.run_id/'overlap/report.json').exists() for r in runs):break
 time.sleep(20)
else:raise RuntimeError('all10formal reports/TTS not ready')
def run(args):
 print(json.dumps({'command':list(map(str,args))}),flush=True);subprocess.run([sys.executable,*map(str,args)],cwd=repo,check=True)
core=repo/'docs/results/reframed-formal-core'
run(['scripts/export_core_results.py','--config',cfgpath,'--tts-root',tts,'--output-dir',core,'--deferred-model','api','--scope','Frozen30clip REFRAMED test subset; all20planned cells listed,10paidAPI cells deferred. Textmetrics are conditional on valid temporal matches; failures remain in coverage.'])
run(['scripts/plot_core_coverage.py','--input',core/'core-results.json','--output',core/'coverage','--title','REFRAMED formal: coverage across10local cells'])
paired=root/'formal-core-temporal/paired.jsonl';contrast=paired.parent/'contrasts.json'
if not paired.exists():run(['scripts/export_core_temporal_metrics.py','--config',cfgpath,'--experiment-dir',exp,'--output',paired])
if not contrast.exists():run(['scripts/analyze_core_temporal.py','--input',paired,'--plan',repo/'configs/core_temporal_contrasts.json','--output',contrast])
dest=repo/'docs/results/reframed-core-temporal';dest.mkdir(parents=True,exist_ok=True);shutil.copyfile(contrast,dest/'contrasts.json')
run(['scripts/plot_paired_results.py','--input',contrast,'--output-dir',dest,'--title','Formal caption placement: movie-clustered F1','--scope','All30clips and10movies;17newlydeclared structural contrasts. Oracleprompt contrast is an upper-bound diagnostic. No matched-pair text significance claim.'])
ablation=root/'formal-ablation5-temporal/contrasts.json'
if not ablation.exists():run(['scripts/analyze_ablation5_temporal.py','--config',cfgpath,'--experiment-dir',exp,'--core-provenance',paired.with_suffix('.jsonl.provenance.json'),'--plan',repo/'configs/ablation5_temporal_contrasts.json','--output',ablation])
adest=repo/'docs/results/reframed-ablation5-temporal';adest.mkdir(parents=True,exist_ok=True);shutil.copyfile(ablation,adest/'contrasts.json')
run(['scripts/plot_paired_results.py','--input',ablation,'--output-dir',adest,'--title','Audio-context ablation: movie-clustered F1','--scope','All30clips and10movies;3structural contrasts, Holm3. Native comparisons change model andmodality; no humanquality claim.'])
gap=Path('/scratch/yy5919/pyro-kd-derived/metrics/measured-gap-formal-core-v1')
if not gap.exists():run(['scripts/export_measured_gap_utilization.py','--config',cfgpath,'--tts-root',tts,'--cache-root','/scratch/yy5919/pyro-kd-derived/cache/reframed-test','--output-dir',gap])
gdest=repo/'docs/results/reframed-formal-gap-utilization'
if not gdest.exists():run(['scripts/compact_gap_report.py','--input-dir',gap,'--output-dir',gdest])
print(json.dumps({'complete':True,'core_cells':10,'planned_cells':20,'paired_contrasts':17,'ablation_contrasts':3}),flush=True)
