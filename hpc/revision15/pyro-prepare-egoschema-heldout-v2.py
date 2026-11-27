"""CPU-only frozen heldout media + visual preparation, resumable per-video caches."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
repo=Path('/scratch/yy5919/pyro-revision15-formal-prep')
sys.path.insert(0,str(repo))
from src.datasets.frozen_membership import egoschema_ids
from src.utils.media import preprocess_video,cache_is_complete,cache_is_current
from src.utils import visual_features as vf
from scripts.cache_visual_features import process_video
from scripts.verify_video_mme_cache import check_frame

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=4)
 p.add_argument('--report',type=Path,required=True);p.add_argument('--plan-only',action='store_true')
 a=p.parse_args()
 if not 0<=a.shard<a.shards:raise ValueError('invalid shard')
 config=repo/'configs/egoschema_pilot.json';allids=egoschema_ids(json.loads(config.read_text()),'heldout');ids=allids[a.shard::a.shards]
 if a.plan_only:
  print(json.dumps({'shard':a.shard,'total_shards':a.shards,'video_ids':ids,'count':len(ids)}));return
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only driver requires CUDA_VISIBLE_DEVICES empty')
 import torch,numpy as np
 torch.set_num_threads(4)
 root=Path('/scratch/yy5919/pyro-kd-derived');cache=root/'cache/egoschema';raw=root/'datasets/egoschema/videos'
 lockroot=cache/'.heldout-locks';lockroot.mkdir(parents=True,exist_ok=True)
 report={'complete':False,'subset':'heldout','expected':len(ids),'shard':a.shard,'shards':a.shards,
         'config_sha256':sha(config),'driver_sha256':sha(__file__),
         'source_commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
         'scope':'CPU media and visual features only; no question or answer text loaded','results':[]}
 a.report.parent.mkdir(parents=True,exist_ok=True)
 if a.report.exists():raise FileExistsError('use a new attempt report; existing caches are reusable')
 embedder=None
 def get_embedder():
  nonlocal embedder
  if embedder is None:embedder=vf.clip_embedder(batch_size=8,device='cpu')
  return embedder
 def save():
  temporary=a.report.with_suffix('.tmp');temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(a.report)
 for ident in ids:
  start=time.monotonic()
  try:
   with (lockroot/(ident+'.lock')).open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=raw/(ident+'.mp4');manifest=preprocess_video(source,cache/ident,fps=2.,max_side=768)
    if not cache_is_current(manifest) or not cache_is_complete(manifest):raise ValueError('incomplete/stale media')
    if manifest['audio_path'] is not None or manifest['media_info']['has_audio']:raise ValueError('unexpected audio')
    frames=manifest['frames']
    if not frames or abs(len(frames)/2-manifest['media_info']['duration'])>1:raise ValueError('incomplete duration coverage')
    if any(f['index']!=i or f['timestamp']!=i/2 for i,f in enumerate(frames)):raise ValueError('incorrect frame grid')
    for frame in frames:check_frame(frame)
    status,features=process_video(cache/ident,get_embedder)
    # process_video's freshly computed record predates atomic publication.
    # Reload the committed record containing its immutable embedding reference.
    features,problem=vf.load_validated_features(cache/ident)
    if problem or features is None:raise ValueError(f'invalid published features: {problem}')
    embeddings=vf.load_embeddings(cache/ident,features)
    if embeddings is None or embeddings.shape!=(len(frames),512):raise ValueError('invalid visual embedding coverage')
    if not np.allclose(np.linalg.norm(embeddings,axis=1),1,atol=.001):raise ValueError('invalid normalization')
    if not np.allclose(vf.novelty_curve(embeddings),features['novelty'],atol=.001):raise ValueError('novelty mismatch')
    row={'video_id':ident,'complete':True,'frames':len(frames),'has_audio':False,'visual_status':status,
         'source_sha256':sha(source),'manifest_sha256':sha(cache/ident/'manifest.json'),
         'visual_sha256':sha(cache/ident/'visual_features.json')}
  except Exception as exc:row={'video_id':ident,'complete':False,'error':repr(exc)}
  row['seconds']=round(time.monotonic()-start,3);report['results'].append(row);save();print(json.dumps(row),flush=True)
 report['complete']=len(report['results'])==len(ids) and all(x['complete'] for x in report['results']);save()
 raise SystemExit(0 if report['complete'] else 1)

if __name__=='__main__':main()
