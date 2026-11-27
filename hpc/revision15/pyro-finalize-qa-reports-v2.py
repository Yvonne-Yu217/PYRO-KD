import json,time,sys,subprocess,shutil
from pathlib import Path
repo=Path('/scratch/yy5919/pyro-revision15-records');root=Path('/scratch/yy5919/pyro-resume-results');inputs=Path('/scratch/yy5919/pyro-kd-derived/moments')
def run(args):
 print(json.dumps({'command':list(map(str,args))}),flush=True);subprocess.run([sys.executable,*map(str,args)],cwd=repo,check=True)
for dataset,folder,scope in [('video-mme','video-mme-long','12audio-bearing long videos,36frozen questions percondition. A/AV are constrained-budget diagnostics; frames differ fromvisualcontrols. No whole-benchmark or equal-budget claim.'),('egoschema','egoschema-heldout','490frozen heldout questions,10pilot questions excluded. Visual-only diagnostic; no audio-aware claim.')]:
 isego=dataset=='egoschema';conditions=['U','V_local','V_memory'] if isego else json.loads((repo/'configs/long_qa_comparisons.json').read_text())['conditions'];evaluation=root/('egoschema-heldout' if isego else 'long-qa-frozen')/'evaluation';dest=repo/'docs/results'/folder
 reports={c:evaluation/(c+('.evaluation.json' if isego else '.json')) for c in conditions};stats=evaluation/'comparisons.json';deadline=time.monotonic()+10800
 while time.monotonic()<deadline:
  if stats.exists() and all(p.exists() for p in reports.values()):break
  time.sleep(20)
 else:raise RuntimeError(dataset+' scored reports not ready')
 args=['scripts/export_qa_results.py','--dataset',dataset,'--output-dir',dest,'--scope',scope]
 for condition,path in reports.items():args+=['--report',condition,path]
 if not isego:
  for condition in conditions:args+=['--qa-input',condition,inputs/f'video-mme-long-{condition}-v1.jsonl']
 run(args);shutil.copyfile(stats,dest/'comparisons.json')
 run(['scripts/plot_paired_results.py','--input',stats,'--output-dir',dest,'--title',dataset+' paired accuracy contrasts','--scope',scope])
 print(json.dumps({'complete':dataset,'conditions':len(conditions)}),flush=True)
