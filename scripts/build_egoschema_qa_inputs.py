"""Prepare label-free U/V QA inputs for the frozen silent EgoSchema pilot only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.generation.qa import build_qa_prompt
from src.utils.media import load_manifest,cache_is_current,cache_is_complete,frame_window
from src.utils.selection import k_for_duration,uniform_timestamps
from src.utils.visual_features import load_validated_features,load_embeddings,VisualCueSelector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--questions',type=Path,required=True)
    p.add_argument('--pilot-config',type=Path,default=Path('configs/egoschema_pilot.json'))
    p.add_argument('--budget-config',type=Path,default=Path('configs/reframed_splits.json'))
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selector',choices=('U','V_local','V_memory'),default='U')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('preserve prior generation inputs')
    pilot=json.loads(args.pilot_config.read_text());ids=pilot['pilot_q_uids']
    if len(ids)!=10 or len(set(ids))!=10 or set(ids)&set(pilot['heldout_q_uids']): raise ValueError('invalid frozen pilot')
    questions={}
    for row in json.loads(args.questions.read_text()):
        if row['q_uid'] in ids:
            if row['q_uid'] in questions: raise ValueError('duplicate pilot question')
            questions[row['q_uid']]=row
    if set(questions)!=set(ids): raise ValueError('missing pilot questions')
    rate=json.loads(args.budget_config.read_text())['k_rule']['rate_per_minute']
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
            [('questions',args.questions),('pilot',args.pilot_config),('budget',args.budget_config)]}
    records=[]
    for ident in ids:
        manifest=load_manifest(args.cache_dir/ident)
        if not cache_is_current(manifest) or not cache_is_complete(manifest): raise ValueError('incomplete/stale media')
        if manifest['audio_path'] is not None: raise ValueError('this protocol requires the verified silent dataset')
        duration=manifest['media_info']['duration'];k=k_for_duration(duration,rate)
        feature_hash=None
        if args.selector=='U':
            times=uniform_timestamps(duration,k)
        else:
            features,reason=load_validated_features(args.cache_dir/ident)
            if features is None: raise ValueError(f'{ident}: {reason}')
            selector=VisualCueSelector()
            if args.selector=='V_memory':
                embeddings=load_embeddings(args.cache_dir/ident,features)
                if embeddings is None: raise ValueError('invalid visual embeddings')
                selected=selector.select_memory(features,embeddings,k)
            else:
                selected=selector.select(features,k)
            times=[candidate.time for candidate in selected]
            feature_hash=hashlib.sha256((args.cache_dir/ident/'visual_features.json').read_bytes()).hexdigest()
        if any(b-a<2 for a,b in zip(times,times[1:])): raise ValueError('K violates2s separation')
        frames=[f for time in times for f in frame_window(manifest,time,4,2.)]
        if len(frames)!=4*len(times): raise ValueError('four frames per moment required')
        source=questions[ident]
        record={'id':ident,'video_id':ident,'dataset':'egoschema','split':'pilot','selector':args.selector.split('_')[0],
                'selection_condition':args.selector,'visual_features_sha256':feature_hash,
                'protocol_status':'exploratory; GPU smoke pending','question':source['question'],
                'options':[source[f'option {i}'] for i in range(5)],
                'frames':[f.path for f in frames],'frame_times':[f.timestamp for f in frames],
                'selected_times':times,'k_requested':k,'k_realized':len(times),'k_shortfall':k-len(times),'rate_per_minute':rate,
                'frames_per_moment':4,'window_seconds':2.,'duration':duration,'audio_context':'none',
                'source_sha256':hashes}
        build_qa_prompt(record)
        records.append(record)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        for row in records: handle.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps({'questions':len(records),'frames_per_question':[len(r['frames']) for r in records],
                      'scope':'pilot only; no answer labels loaded; no model inference'}))


if __name__=='__main__':main()
