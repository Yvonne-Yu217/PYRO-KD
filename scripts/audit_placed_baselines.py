"""Development Uniform/oracle anchors under explicitly shared placement constraints."""
import argparse
from dataclasses import asdict
import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.datasets import reframed
from src.utils.audio_features import load_current_features
from src.utils.media import load_manifest,cache_is_current
from src.utils.selection import k_for_duration,uniform_timestamps
from src.utils.placed_selection import Candidate,select_placed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--splits',type=Path,default=Path('configs/reframed_splits.json'))
    p.add_argument('--required-words',type=int,required=True)
    p.add_argument('--max-advance',type=float,required=True)
    p.add_argument('--max-lag',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('use a new diagnostic output')
    split=json.loads(args.splits.read_text())
    clips={c.video_id:c for c in reframed.load_split(args.data_dir,'validation')}
    records=[]
    for item in split['clips']:
        video_id=item['video_id'];clip=clips[video_id];cache=args.cache_dir/video_id
        manifest=load_manifest(cache)
        if not cache_is_current(manifest): raise ValueError('stale media source')
        audio,reason=load_current_features(cache,manifest)
        if audio is None: raise ValueError(reason)
        k=k_for_duration(clip.video_duration,split['k_rule']['rate_per_minute'])
        # Ground truth is supplied only to the explicitly oracle condition.
        anchors={'U_snapped':uniform_timestamps(clip.video_duration,k),
                 'O_snapped':[c.start for c in reframed.merge_cues(clip.ad_us,max_gap=.5)]}
        conditions={}
        for name,times in anchors.items():
            result=select_placed([Candidate(t,0) for t in times],[(g['start'],g['end']) for g in audio['gaps']],k,
                required_words=args.required_words,max_advance=args.max_advance,max_lag=args.max_lag,novelty_weight=0)
            conditions[name]={**asdict(result),'shortfall':result.shortfall,'anchor_times':times}
        records.append({'video_id':video_id,'conditions':conditions,
            'source_sha256':{'audio_features.json':hashlib.sha256((cache/'audio_features.json').read_bytes()).hexdigest()}})
    report={'scope':'exploratory common-placement baselines; U event anchors uniform, O uses merged US human starts',
            'oracle_policy':'all reference starts as candidates; equal scores, earliest feasible playback tie-break, shared2s separation; not a deployable method',
            'required_words':args.required_words,'max_advance':args.max_advance,'max_lag':args.max_lag,
            'split_sha256':hashlib.sha256(args.splits.read_bytes()).hexdigest(),'videos':records,
            'totals':{name:{'requested':sum(r['conditions'][name]['requested'] for r in records),
                            'selected':sum(len(r['conditions'][name]['selected']) for r in records)} for name in anchors}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(report,handle,indent=2,allow_nan=False)
    print(json.dumps(report['totals']))


if __name__=='__main__': main()
