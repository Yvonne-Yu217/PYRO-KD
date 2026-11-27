import fcntl,json,shlex,subprocess,time
from pathlib import Path
repo=Path('/scratch/yy5919/pyro-revision15-formal')
inp=Path('/scratch/yy5919/pyro-kd-derived/moments/egoschema-heldout-V_memory-v1.jsonl')
out=Path('/scratch/yy5919/pyro-resume-results/egoschema-heldout/V_memory.jsonl')
model='/home/yy5919/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'
deadline=time.monotonic()+14400
while not inp.exists() and time.monotonic()<deadline:time.sleep(20)
if not inp.exists():raise SystemExit('Inputs unavailable; no GPU lock acquired')
rows=[json.loads(x) for x in inp.read_text().splitlines()];assert len(rows)==len({r['id'] for r in rows})==490
with Path('/scratch/yy5919/pyro-resume-gpu.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if out.exists():
  done=[json.loads(x) for x in out.read_text().splitlines()]
  if len(done)==len({r['id'] for r in done})==490:raise SystemExit('Already complete')
 cmd='OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '+shlex.join(['python','-u','scripts/run_qa_vlm.py','--input',str(inp),'--output',str(out),'--model',model,'--max-side','768','--max-new-tokens','16'])
 print('Starting frozen visual-only EgoSchema V_memory after primary core generation',flush=True)
 result=subprocess.run(['timeout','--kill-after=30','5400','bash','-c','source hpc/config.sh; in_env ro '+shlex.quote(cmd)],cwd=repo)
 raise SystemExit(result.returncode)
