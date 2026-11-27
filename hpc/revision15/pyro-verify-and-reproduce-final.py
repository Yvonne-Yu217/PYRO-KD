import json, subprocess, sys, time
from pathlib import Path
repo=Path('/scratch/yy5919/pyro-revision15-records')
root=Path('/scratch/yy5919/pyro-resume-results')
status=root/'egoschema-heldout/evaluation/watcher-status.json'
deadline=time.monotonic()+7200
while time.monotonic()<deadline:
    if status.exists() and json.loads(status.read_text()).get('state')=='complete':
        break
    time.sleep(20)
else:
    raise TimeoutError('Ego evaluation did not complete within two hours')
commands=[
 ['scripts/verify_revision15_completion.py','--gap-evidence','docs/results/reframed-formal-gap-utilization/measured-gap-utilization.json','--gap-evidence','docs/results/reframed-formal-ablation5-gap-utilization/measured-gap-utilization.json','--temporal-evidence',str(root/'formal-core-temporal/paired.jsonl.provenance.json'),'--output',str(root/'revision15-completion.json')],
 ['scripts/reproduce_revision15_tables.py','--completion-manifest',str(root/'revision15-completion.json'),'--output-root',str(root/'reproduced-tables-v1')]
]
for args in commands:
    print(json.dumps({'command':args}),flush=True)
    subprocess.run([sys.executable,*args],cwd=repo,check=True)
print('Final completion and cached-table reproduction passed',flush=True)
