import fcntl,json,shlex,subprocess,hashlib
from pathlib import Path
repo=Path('/scratch/yy5919/pyro-revision15-long-prep')
model='/home/yy5919/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'
for selector in ['A_cue','AV']:
 p=Path('/scratch/yy5919/pyro-kd-derived/moments')/f'video-mme-long-{selector}-flat-none-v1.jsonl'
 rows=[json.loads(x) for x in p.read_text().splitlines()];assert len(rows)==len({r['id'] for r in rows})==36
with Path('/scratch/yy5919/pyro-resume-gpu.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for selector in ['A_cue','AV']:
  inp=Path('/scratch/yy5919/pyro-kd-derived/moments')/f'video-mme-long-{selector}-flat-none-v1.jsonl'
  out=Path('/scratch/yy5919/pyro-resume-results/long-qa-frozen')/f'{selector}-flat-none-224.jsonl'
  command='OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 '+shlex.join(['python','-u','scripts/run_qa_vlm.py','--input',str(inp),'--output',str(out),'--model',model,'--max-side','224','--max-new-tokens','16'])
  print(json.dumps({'starting':selector,'source':'a7e0e1a','input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest()}),flush=True)
  result=subprocess.run(['timeout','--kill-after=30','2700','bash','-c','source hpc/config.sh; in_env ro '+shlex.quote(command)],cwd=repo)
  print(json.dumps({'finished':selector,'exit_code':result.returncode}),flush=True)
