#!/usr/bin/env python
"""Development-only diagnosis of Uniform gap snapping; does not change selections."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.build_moments import load_targets
from src.utils.audio_features import load_current_features
from src.utils.gap_snapping import snap_to_gap
from src.utils.media import load_manifest
from src.utils.selection import k_for_duration,uniform_timestamps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('mtg-ad','reframed'),required=True)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    splits=json.loads(args.splits.read_text())
    targets=load_targets(args.dataset,'dev',splits,args.data_dir)
    results=[]
    for required_words in (1,6,15):
        for bound in (0.0,1.0,2.0,5.0,10.0):
            clips=[]
            for video_id,duration,_ in targets:
                cache=args.cache_dir/video_id
                manifest=load_manifest(cache)
                features,reason=load_current_features(cache,manifest)
                if features is None: raise ValueError(f'{video_id}: {reason}')
                k=k_for_duration(duration,splits['k_rule']['rate_per_minute'])
                candidates=uniform_timestamps(duration,k)
                gaps=[(g['start'],g['end']) for g in features['gaps']]
                placements=[snap_to_gap(t,gaps,required_words,max_advance=bound,max_lag=bound) for t in candidates]
                found=[x for x in placements if x is not None]
                # Count distinct nearest placements respecting two-second separation.
                # This greedy count is diagnostic, not an optimal joint assignment.
                distinct=[]
                for m in sorted(found,key=lambda x:x.time):
                    if not distinct or m.time-distinct[-1]>=2:
                        distinct.append(m.time)
                clips.append({'video_id':video_id,'requested':k,'individually_snappable':len(found),
                              'nearest_placements_after_2s_separation':len(distinct),
                              'absolute_lag_sum':sum(abs(x.lag_seconds) for x in found),
                              'annotation_duration':duration,'media_duration':manifest['media_info']['duration']})
            results.append({'required_words':required_words,'max_advance':bound,'max_lag':bound,
                            'requested':sum(c['requested'] for c in clips),
                            'individually_snappable':sum(c['individually_snappable'] for c in clips),
                            'nearest_placements_after_2s_separation':sum(c['nearest_placements_after_2s_separation'] for c in clips),
                            'clips':clips})
    report={'dataset':args.dataset,'split':'dev','split_sha256':hashlib.sha256(args.splits.read_bytes()).hexdigest(),
            'scope':'feasibility diagnosis only; nearest-placement deduplication is not optimal assignment; no policy frozen',
            'gaps':'current VAD gaps with existing minimum-gap and safety-margin settings',
            'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='clips'} for r in results],indent=2))


if __name__=='__main__': main()
