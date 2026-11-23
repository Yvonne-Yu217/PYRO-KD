"""Inspect untuned A/AV trigger/suppress selections on frozen development clips."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.build_moments import load_targets
from src.utils.audio_features import load_current_features,resolve_audio_path
from src.utils.visual_features import load_validated_features,load_embeddings
from src.utils.media import load_manifest,cache_is_current
from src.utils.selection import k_for_duration
from src.utils.fused_candidates import make_candidates
from src.utils.placed_selection import select_placed


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--speech-cues',type=Path,required=True)
    p.add_argument('--split-config',type=Path,default=Path('configs/reframed_splits.json'))
    p.add_argument('--required-words',type=int,required=True)
    p.add_argument('--max-advance',type=float,required=True)
    p.add_argument('--max-lag',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('preserve prior diagnostic; use a new output path')
    split=json.loads(args.split_config.read_text())
    speech=json.loads(args.speech_cues.read_text())
    if not speech['complete'] or speech['split_sha256']!=sha(args.split_config): raise ValueError('speech cue split mismatch')
    by_id={r['video_id']:r for r in speech['videos']}
    weights={'event':1.,'semantic':1.,'deictic':1.,'silence':1.,'shot':.5,'novelty':1.}
    records=[]
    for video_id,duration,_ in load_targets('reframed','dev',split,args.data_dir):
        cache=args.cache_dir/video_id
        manifest=load_manifest(cache)
        if not cache_is_current(manifest): raise ValueError('stale source')
        audio,reason=load_current_features(cache,manifest)
        if audio is None: raise ValueError(f'{video_id}: {reason}')
        cues=by_id[video_id]
        if cues['speech_sha256']!=sha(cache/'speech_features.json') or cues['vad_config_hash']!=audio['config_hash']:
            raise ValueError('stale speech cues')
        wavhash=sha(resolve_audio_path(manifest,cache))
        asr=json.loads((cache/'speech_features.json').read_text())
        meta=json.loads((cache/'event_features.json').read_text())
        if not meta['complete'] or any(x['provenance']['audio_sha256']!=wavhash for x in (asr,meta)):
            raise ValueError('stale audio streams')
        if sha(cache/'event_features.npz')!=meta['arrays_sha256']: raise ValueError('event arrays changed')
        with np.load(cache/'event_features.npz',allow_pickle=False) as arrays:
            event={name:arrays[name] for name in arrays.files}
        visual,reason=load_validated_features(cache)
        if visual is None: raise ValueError(f'{video_id}: {reason}')
        embeddings=load_embeddings(cache,visual)
        if embeddings is None: raise ValueError('invalid visual embeddings')
        k=k_for_duration(duration,split['k_rule']['rate_per_minute'])
        conditions={}
        for mode in ('A','AV'):
            for sign in (1,-1):
                candidates,vectors,components=make_candidates(mode,event,meta['labels'],cues['tracks'],sign=sign,
                    weights=weights,visual=visual,visual_embeddings=embeddings)
                selected=select_placed(candidates,[(g['start'],g['end']) for g in audio['gaps']],k,
                    required_words=args.required_words,max_advance=args.max_advance,max_lag=args.max_lag,
                    novelty_weight=weights['novelty'] if mode=='AV' else 0,embeddings=vectors)
                name=mode+('_trigger' if sign==1 else '_suppress')
                conditions[name]={**asdict(selected),'shortfall':selected.shortfall,
                    'candidate_components':[{'event_time':c.event_time,**scores} for c,scores in zip(candidates,components)]}
        records.append({'video_id':video_id,'conditions':conditions,'source_sha256':{
            name:sha(cache/name) for name in ('audio_features.json','speech_features.json','event_features.json','visual_features.json')}})
        print(json.dumps({'video_id':video_id,'selected':{name:len(r['selected']) for name,r in conditions.items()}}),flush=True)
    names=list(records[0]['conditions'])
    report={'scope':'exploratory mechanics only; no tuned weights, validated detector, GF/OP score or final matrix',
            'weights':weights,'required_words':args.required_words,'max_advance':args.max_advance,'max_lag':args.max_lag,
            'speech_cues_sha256':sha(args.speech_cues),'split_sha256':sha(args.split_config),'videos':records,
            'totals':{name:{'requested':sum(r['conditions'][name]['requested'] for r in records),
                            'selected':sum(len(r['conditions'][name]['selected']) for r in records)} for name in names}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(report,handle,indent=2,allow_nan=False)
    print(json.dumps(report['totals']))


if __name__=='__main__': main()
