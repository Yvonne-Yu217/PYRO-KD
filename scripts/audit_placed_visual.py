"""Development-only comparison of visual ranking after explicit gap snapping."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.build_moments import load_targets
from src.utils.audio_features import load_current_features as load_audio
from src.utils.visual_features import load_validated_features,load_embeddings,VisualCueSelector
from src.utils.placed_selection import Candidate,select_placed
from src.utils.media import load_manifest
from src.utils.selection import k_for_duration


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
    splits=json.loads(args.splits.read_text())
    records=[]
    for video_id,duration,_ in load_targets('reframed','dev',splits,args.data_dir):
        cache=args.cache_dir/video_id
        visual,reason=load_validated_features(cache)
        if visual is None: raise ValueError(f'{video_id}: {reason}')
        embeddings=load_embeddings(cache,visual)
        if embeddings is None: raise ValueError(f'{video_id}: invalid embeddings')
        audio,reason=load_audio(cache,load_manifest(cache))
        if audio is None: raise ValueError(f'{video_id}: {reason}')
        selector=VisualCueSelector()
        old=selector.candidates(visual)
        peak=max(visual['novelty']) or 1
        candidates=[Candidate(c.time,selector.shot_weight*c.shot_boundary,c.novelty/peak) for c in old]
        k=k_for_duration(duration,splits['k_rule']['rate_per_minute'])
        modes={}
        for mode in ('local','memory'):
            result=select_placed(candidates,[(g['start'],g['end']) for g in audio['gaps']],k,
                                required_words=args.required_words,max_advance=args.max_advance,
                                max_lag=args.max_lag,novelty_weight=selector.novelty_weight,
                                embeddings=embeddings if mode=='memory' else None)
            modes[mode]={**asdict(result),'shortfall':result.shortfall}
        records.append({'video_id':video_id,'modes':modes,
            'source_sha256':{name:hashlib.sha256((cache/name).read_bytes()).hexdigest()
                             for name in ('manifest.json','audio_features.json','visual_features.json')}})
    report={'scope':'exploratory dev ranking; no protocol frozen, no AD generation or accuracy claim',
            'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'split_sha256':hashlib.sha256(args.splits.read_bytes()).hexdigest(),'videos':records,
            'totals':{mode:{'requested':sum(r['modes'][mode]['requested'] for r in records),
                            'selected':sum(len(r['modes'][mode]['selected']) for r in records),
                            'shortfall':sum(r['modes'][mode]['shortfall'] for r in records)}
                      for mode in ('local','memory')}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['totals']))


if __name__=='__main__': main()
