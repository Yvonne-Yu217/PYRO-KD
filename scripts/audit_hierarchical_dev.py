"""Short-development mechanics check for long-video hierarchical allocation."""
import argparse
from bisect import bisect_right
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.utils.hierarchical_budget import segment_video,event_densities,allocate_budget
from src.utils.placed_selection import Candidate,select_placed
from src.utils.fused_candidates import nearest_indices
from src.utils.visual_features import load_validated_features,load_embeddings
from src.utils.audio_features import load_current_features
from src.utils.media import load_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-segment',type=float,required=True)
    p.add_argument('--max-segment',type=float,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('use a new audit output')
    source=json.loads(args.input.read_text());records=[]
    for video in source['videos']:
        video_id=video['video_id'];cache=args.cache_dir/video_id
        for name,digest in video['source_sha256'].items():
            if hashlib.sha256((cache/name).read_bytes()).hexdigest()!=digest: raise ValueError('source changed')
        manifest=load_manifest(cache)
        visual,reason=load_validated_features(cache)
        if visual is None: raise ValueError(reason)
        embeddings=load_embeddings(cache,visual)
        audio,reason=load_current_features(cache,manifest)
        if embeddings is None or audio is None: raise ValueError('invalid cached cues')
        event=json.loads((cache/'event_features.json').read_text())
        raw=(cache/'event_features.npz').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=event['arrays_sha256']: raise ValueError('event arrays changed')
        with np.load(cache/'event_features.npz',allow_pickle=False) as arrays:
            times=arrays['times'];onsets=arrays['class_onsets'];probs=arrays['probabilities']
        music=event['labels'].index('Music')
        boundaries=visual['shots']+[g['start'] for g in audio['gaps'] if g['end']-g['start']>=5]
        boundaries += [float(t) for t,rise,prob in zip(times,onsets[:,music],probs[:,music]) if rise>=.1 and prob>=.5]
        segments=segment_video(manifest['media_info']['duration'],boundaries,
                               min_seconds=args.min_segment,max_seconds=args.max_segment)
        density=event_densities(segments,times.tolist(),onsets.max(axis=1).tolist())
        flat=video['conditions']['AV_trigger'];weights=source['weights']
        allocation=allocate_budget(density,flat['requested'],floor=1)
        components=flat['candidate_components']
        candidates=[Candidate(c['event_time'],sum(weights[k]*c[k] for k in weights if k!='novelty'),c['novelty']) for c in components]
        starts=[s[0] for s in segments]
        ids=[min(bisect_right(starts,c.event_time)-1,len(segments)-1) for c in candidates]
        vectors=embeddings[nearest_indices(visual['frame_times'],[c.event_time for c in candidates])]
        result=select_placed(candidates,[(g['start'],g['end']) for g in audio['gaps']],flat['requested'],
            required_words=source['required_words'],max_advance=source['max_advance'],max_lag=source['max_lag'],
            novelty_weight=weights['novelty'],embeddings=vectors,segment_ids=ids,segment_quotas=allocation.quotas)
        used=[sum(ids[s.candidate_index]==i for s in result.selected) for i in range(len(segments))]
        if any(n>q for n,q in zip(used,allocation.quotas)): raise ValueError('quota violated')
        records.append({'video_id':video_id,'segments':segments,'densities':density,'allocation':asdict(allocation),
                        'used_per_segment':used,'flat_selected':len(flat['selected']),
                        'hierarchical':{**asdict(result),'shortfall':result.shortfall}})
    report={'scope':'short-clip mechanics check only; not long-form evaluation or tuned comparison',
            'source_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
            'settings':{'min_segment':args.min_segment,'max_segment':args.max_segment,'quiet_floor':1,
                        'long_silence_boundary_seconds':5,'music_min_rise':.1,'music_min_probability':.5},
            'videos':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(report,handle,indent=2,allow_nan=False)
    print(json.dumps({'videos':len(records),'requested':sum(r['allocation']['requested'] for r in records),
                      'selected':sum(len(r['hierarchical']['selected']) for r in records)}))


if __name__=='__main__': main()
