"""Label-blind long-video visual hierarchy mechanics; no QA or AD accuracy."""
import argparse
from bisect import bisect_right
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.hierarchical_budget import segment_video,event_densities,allocate_budget
from src.utils.visual_features import load_validated_features,load_embeddings,VisualCueSelector
from src.utils.media import load_manifest
from src.utils.selection import k_for_duration


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--subset',type=Path,default=Path('configs/video_mme_long_subset.json'))
    p.add_argument('--budget',type=Path,default=Path('configs/reframed_splits.json'))
    p.add_argument('--dev-hierarchy',type=Path,required=True)
    p.add_argument('--flat-audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve prior mechanics output')
    ids=[r['video_id'] for r in json.loads(args.subset.read_text())['videos']]
    flat=json.loads(args.flat_audit.read_text())
    if (not flat['complete'] or flat['subset_config_sha256']!=sha(args.subset)
            or flat['budget_config_sha256']!=sha(args.budget)):
        raise ValueError('matching complete flat visual audit required')
    previous={r['video_id']:r for r in flat['videos']}
    if len(previous)!=len(ids) or set(previous)!=set(ids):raise ValueError('flat coverage differs')
    settings=json.loads(args.dev_hierarchy.read_text())['settings']
    rate=json.loads(args.budget.read_text())['k_rule']['rate_per_minute']
    records=[]
    for ident in ids:
        cache=args.cache_dir/ident;features,problem=load_validated_features(cache)
        if features is None:raise ValueError(problem)
        if sha(cache/'visual_features.json')!=previous[ident]['features_sha256']:
            raise ValueError('features changed since flat audit')
        vectors=load_embeddings(cache,features)
        if vectors is None:raise ValueError('invalid embeddings')
        duration=load_manifest(cache)['media_info']['duration'];k=k_for_duration(duration,rate)
        segments=segment_video(duration,features['shots'],min_seconds=settings['min_segment'],max_seconds=settings['max_segment'])
        selector=VisualCueSelector();candidates=selector.candidates(features)
        densities=event_densities(segments,[c.time for c in candidates],[c.score for c in candidates])
        allocation=allocate_budget(densities,k,floor=settings['quiet_floor'])
        starts=[a for a,b in segments]
        segment_ids=[min(bisect_right(starts,c.time)-1,len(segments)-1) for c in candidates]
        kwargs=dict(segment_ids=segment_ids,segment_quotas=allocation.quotas)
        selected={'local':selector.select(features,k,**kwargs),
                  'memory':selector.select_memory(features,vectors,k,**kwargs)}
        conditions={}
        for name,chosen in selected.items():
            times=[c.time for c in chosen];counts=[0]*len(segments)
            for time in times:counts[min(bisect_right(starts,time)-1,len(segments)-1)]+=1
            if (len(times)>k or any(b-a<2 for a,b in zip(times,times[1:]))
                    or any(n>q for n,q in zip(counts,allocation.quotas))
                    or not set(times)<=set(features['frame_times'])):
                raise ValueError('hierarchical selection violates constraints')
            baseline=previous[ident]['selected'][name]
            conditions[name]=dict(selected_times=times,selected=len(times),shortfall=k-len(times),
                segment_selected=counts,unfilled_quotas=[q-n for n,q in zip(counts,allocation.quotas)],
                flat_selected=len(baseline),shared_with_flat=len(set(times)&set(baseline)))
        records.append(dict(video_id=ident,duration=duration,segments=segments,densities=densities,
                            allocation=asdict(allocation),conditions=conditions))
        print(json.dumps({'video_id':ident,'segments':len(segments),'requested':k,
                          'selected':{name:r['selected'] for name,r in conditions.items()}}),flush=True)
    report=dict(complete=True,scope='visual-only hierarchy mechanics on long videos; not tuned or frozen QA budget; no content/answers inspected',
        source_sha256={key:sha(path) for key,path in [('subset',args.subset),('budget',args.budget),
            ('dev_hierarchy',args.dev_hierarchy),('flat_audit',args.flat_audit)]},
        settings={'min_segment':settings['min_segment'],'max_segment':settings['max_segment'],
            'quiet_floor':settings['quiet_floor'],'boundaries':'cached visual shots only',
            'density':'local visual candidate score mass per second','rate_per_minute':rate,
            'memory':'global selected frames across segments','min_separation':2.0},videos=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
