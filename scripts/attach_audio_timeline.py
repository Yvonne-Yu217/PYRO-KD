#!/usr/bin/env python
"""Attach derived Whisper/event context to moments without changing selected frames/times."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.generation.audio_timeline import contextual_timeline,timeline_hash
from src.utils.audio_features import load_current_features,resolve_audio_path
from src.utils.event_features import EVENT_VERSION,EXCLUDED_LABELS
from src.utils.media import load_manifest,cache_is_current
from src.utils.speech_features import SPEECH_FEATURES_VERSION


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--min-score',type=float,required=True)
    p.add_argument('--min-rise',type=float,required=True)
    args=p.parse_args()
    if not 0<=args.min_score<=1 or not 0<args.min_rise<=1: raise ValueError('invalid event thresholds')
    if args.output.exists(): raise FileExistsError('output exists; preserve frozen inputs with a new output path')
    rows=[json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    sources={}
    for video_id in sorted({r['video_id'] for r in rows}):
        cache=args.cache_dir/video_id
        manifest=load_manifest(cache)
        if not cache_is_current(manifest): raise ValueError(f'{video_id}: stale preprocessed cache')
        vad,reason=load_current_features(cache,manifest)
        if vad is None: raise ValueError(f'{video_id}: {reason}')
        wav=resolve_audio_path(manifest,cache)
        digest=hashlib.sha256(wav.read_bytes()).hexdigest()
        speech=json.loads((cache/'speech_features.json').read_text())
        event=json.loads((cache/'event_features.json').read_text())
        if not speech['complete'] or not event['complete']: raise ValueError('incomplete audio streams')
        if speech['provenance']['version']!=SPEECH_FEATURES_VERSION or event['provenance']['version']!=EVENT_VERSION: raise ValueError('outdated feature version')
        if any(feature['provenance']['audio_sha256']!=digest for feature in (speech,event)): raise ValueError('audio features belong to different/stale WAV')
        if speech['provenance']['vad_config_hash']!=vad['config_hash'] or speech['provenance']['speech']!=vad['speech']: raise ValueError('stale speech gates')
        raw=(cache/'event_features.npz').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=event['arrays_sha256']: raise ValueError('event arrays changed')
        with np.load(cache/'event_features.npz',allow_pickle=False) as arrays:
            times=arrays['times']; scores=arrays['probabilities']; onsets=arrays['class_onsets']
            if scores.shape!=onsets.shape or scores.shape!=(len(times),len(event['labels'])): raise ValueError('event dimensions mismatch')
            events=[]
            for i,t in enumerate(times):
                for c,label in enumerate(event['labels']):
                    rise=float(onsets[i,c]);score=float(scores[i,c])
                    if label in EXCLUDED_LABELS or score<args.min_score or rise<args.min_rise: continue
                    # Strict earlier / non-strict later comparison picks the first point of a plateau.
                    if i and rise<=float(onsets[i-1,c]): continue
                    if i+1<len(times) and rise<float(onsets[i+1,c]): continue
                    events.append({'kind':'event','start':float(t),'end':float(t),'text':label,'score':score,'rise':rise})
        sources[video_id]={'words':speech['words'],'events':events,
                           'provenance':{'speech_hash':speech['config_hash'],'event_hash':event['config_hash'],'arrays_sha256':event['arrays_sha256']}}
    output=[]
    for row in rows:
        source=sources[row['video_id']]
        center=row.get('frame_center_time',row['time'])
        entries,truncated=contextual_timeline(source['words'],source['events'],center)
        output.append({**row,'audio_timeline':entries,'audio_timeline_sha256':timeline_hash(entries),
                       'audio_timeline_provenance':{**source['provenance'],'min_score':args.min_score,'min_rise':args.min_rise,
                                                    'context_center_time':center,'context_seconds_each_side':15.0,'max_entries':40,'omitted_entries':truncated}})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        for row in output: handle.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps({'moments':len(output),'videos':len(sources),'entries':sum(len(r['audio_timeline']) for r in output),
                      'output':str(args.output),'thresholds_status':'exploratory; not tuned or detector-validated'}))


if __name__=='__main__': main()
