"""Publish three frozen label-free EgoSchema inputs after validated CPU caches."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,uuid
from pathlib import Path
REPO=Path('/scratch/yy5919/pyro-revision15-formal')
COMMIT='09abb6c0512e5e97549ad4e66d3bae05a68c42c8'
sys.path.insert(0,str(REPO))
from src.datasets.frozen_membership import egoschema_ids
from src.generation.qa import build_qa_prompt,reject_labels
from src.utils.media import load_manifest,cache_is_current,cache_is_complete,frame_window
from src.utils.selection import k_for_duration,uniform_timestamps
from src.utils.visual_features import load_validated_features,load_embeddings,VisualCueSelector

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate(path,condition,ids,hashes,questions,cache,rate):
 rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
 if len(rows)!=len(ids) or len({r['id'] for r in rows})!=len(ids) or {r['id'] for r in rows}!=set(ids):
  raise ValueError('input membership is not exactly490 frozen unique heldout IDs')
 for row in rows:
  ident=row['id'];question=questions[ident]
  if (row['video_id']!=ident or row['dataset']!='egoschema' or row['split']!='heldout'
      or row['selection_condition']!=condition or row['selector']!=condition.split('_')[0]
      or row['audio_context']!='none' or row['source_sha256']!=hashes):raise ValueError('input provenance/condition mismatch')
  if row['question']!=question['question'] or row['options']!=[question[f'option {i}'] for i in range(5)]:
   raise ValueError('question/option content or order changed')
  manifest=load_manifest(cache/ident)
  if not cache_is_current(manifest) or not cache_is_complete(manifest) or manifest['audio_path'] is not None:
   raise ValueError('media cache is stale/incomplete or no longer silent')
  duration=manifest['media_info']['duration'];k=k_for_duration(duration,rate)
  if condition=='U':
   times=uniform_timestamps(duration,k);feature_hash=None
  else:
   features,problem=load_validated_features(cache/ident)
   if features is None:raise ValueError(problem)
   selector=VisualCueSelector()
   if condition=='V_memory':
    embeddings=load_embeddings(cache/ident,features)
    if embeddings is None:raise ValueError('invalid visual embeddings')
    selected=selector.select_memory(features,embeddings,k)
   else:selected=selector.select(features,k)
   times=[candidate.time for candidate in selected]
   feature_hash=sha(cache/ident/'visual_features.json')
  frames=[f for t in times for f in frame_window(manifest,t,4,2.)]
  expected={'selected_times':times,'frames':[f.path for f in frames],
            'frame_times':[f.timestamp for f in frames],'k_requested':k,'k_realized':len(times),
            'k_shortfall':k-len(times),'rate_per_minute':rate,'frames_per_moment':4,
            'window_seconds':2.,'duration':duration,'visual_features_sha256':feature_hash}
  if any(row.get(key)!=value for key,value in expected.items()):raise ValueError('cached selection/frame protocol differs')
  reject_labels(row);build_qa_prompt(row)
 return {'questions':len(rows),'moments':sum(r['k_realized'] for r in rows),
         'shortfall':sum(r['k_shortfall'] for r in rows),'frame_references':sum(len(r['frames']) for r in rows)}

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan-only',action='store_true');args=parser.parse_args()
 actual=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
 if actual!=COMMIT:raise ValueError('formal source commit changed')
 root=Path('/scratch/yy5919/pyro-kd-derived');membership=REPO/'configs/egoschema_pilot.json';budget=REPO/'configs/reframed_splits.json'
 question_path=root/'datasets/egoschema/questions.json';cache=root/'cache/egoschema';output=root/'moments'
 ids=egoschema_ids(json.loads(membership.read_text()),'heldout')
 if len(ids)!=490:raise ValueError('expected frozen490heldout questions')
 conditions=['U','V_local','V_memory']
 targets={c:output/f'egoschema-heldout-{c}-v1.jsonl' for c in conditions}
 hashes={'questions':sha(question_path),'pilot':sha(membership),'budget':sha(budget)}
 if args.plan_only:
  print(json.dumps({'source_commit':COMMIT,'questions_per_condition':len(ids),'source_sha256':hashes,'outputs':{k:str(v) for k,v in targets.items()},'model_loaded':False}));return
 if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only builder requires CUDA_VISIBLE_DEVICES empty')
 questions={}
 for row in json.loads(question_path.read_text()):
  if row['q_uid'] in ids:
   if row['q_uid'] in questions:raise ValueError('duplicate question')
   questions[row['q_uid']]={'question':row['question'],**{f'option {i}':row[f'option {i}'] for i in range(5)}}
 if set(questions)!=set(ids):raise ValueError('missing source questions')
 rate=json.loads(budget.read_text())['k_rule']['rate_per_minute']
 output.mkdir(parents=True,exist_ok=True)
 with (output/'.egoschema-heldout-inputs.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for condition,path in targets.items():
   started=time.monotonic();reused=path.exists()
   if not reused:
    staging=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.staging')
    subprocess.run([sys.executable,'-u','scripts/build_egoschema_qa_inputs.py',
        '--questions',str(question_path),'--pilot-config',str(membership),'--budget-config',str(budget),
        '--cache-dir',str(cache),'--output',str(staging),'--split','heldout','--selector',condition],cwd=REPO,check=True)
    summary=validate(staging,condition,ids,hashes,questions,cache,rate)
    os.link(staging,path);staging.unlink()  # Atomic publication, refuses overwrite.
   else:summary=validate(path,condition,ids,hashes,questions,cache,rate)
   receipt={**summary,'condition':condition,'reused':reused,'source_commit':COMMIT,
            'driver_sha256':sha(__file__),'source_sha256':hashes,'input_sha256':sha(path),
            'seconds':round(time.monotonic()-started,3),'scope':'label-free CPU input preparation; no model inference'}
   sidecar=path.with_suffix('.jsonl.preparation.json');tmp=sidecar.with_suffix('.tmp')
   tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(sidecar)
   print(json.dumps({'published':str(path),**receipt}),flush=True)

if __name__=='__main__':main()
