"""Per-class descriptive event/AD association on frozen REFRAMED development only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.datasets import reframed
from src.evaluation.event_alignment import alignment
from src.utils.event_features import EXCLUDED_LABELS
from src.utils.audio_features import resolve_audio_path
from src.utils.media import load_manifest,cache_is_current


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--split-config',type=Path,default=Path('configs/reframed_splits.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve prior development diagnostic')
    split=json.loads(args.split_config.read_text());ids=[r['video_id'] for r in split['clips']]
    if split['split']!='validation' or not ids or len(set(ids))!=len(ids):
        raise ValueError('frozen validation development IDs required')
    clips={c.video_id:c for c in reframed.load_split(args.data_dir,'validation')}
    if len({clips[i].movie_id for i in ids})!=len(ids):raise ValueError('one development clip per movie required')
    pins={r['repo_id']:r['revision'] for r in json.loads(Path('configs/audio_models.json').read_text())['models']}
    models={key:pins[key] for key in ('MIT/ast-finetuned-audioset-10-10-0.4593','laion/clap-htsat-unfused')}
    records=[];vocabulary=None
    for ident in ids:
        cache=args.cache_dir/ident;clip=clips[ident];manifest=load_manifest(cache)
        if not cache_is_current(manifest):raise ValueError('stale source')
        meta=json.loads((cache/'event_features.json').read_text())
        if (not meta['complete'] or meta['provenance']['models']!=models
                or meta['provenance']['hop_seconds']!=1.0 or meta['provenance']['window_seconds']!=10.0
                or sha(resolve_audio_path(manifest,cache))!=meta['provenance']['audio_sha256']
                or sha(cache/'event_features.npz')!=meta['arrays_sha256']):
            raise ValueError('event provenance mismatch')
        labels=meta['labels']
        if len(set(labels))!=len(labels) or (vocabulary is not None and labels!=vocabulary):
            raise ValueError('inconsistent event vocabulary')
        vocabulary=labels
        truth=[c.start for c in reframed.merge_cues(clip.ad_us,max_gap=.5)]
        if any(not 0<=t<=manifest['media_info']['duration'] for t in truth):
            raise ValueError('out-of-bounds AD timestamps')
        with np.load(cache/'event_features.npz',allow_pickle=False) as arrays:
            if not np.array_equal(arrays['times'],np.arange(int(np.ceil(meta['duration'])))):
                raise ValueError('event time grid differs from declared protocol')
            values=alignment(arrays['times'],arrays['probabilities'],arrays['class_onsets'],truth)
        if len(values)!=len(labels):raise ValueError('event vocabulary/array mismatch')
        records.append(dict(video_id=ident,movie_id=clip.movie_id,ad_starts=len(truth),
            event_metadata_sha256=sha(cache/'event_features.json'),
            per_class={label:value for label,value in zip(labels,values) if label not in EXCLUDED_LABELS}))
    summaries={}
    for label in records[0]['per_class']:
        rows=[r['per_class'][label] for r in records]
        phis=[r['phi'] for r in rows if r['phi'] is not None]
        summaries[label]=dict(event_windows=sum(r['event_windows'] for r in rows),
            videos_with_event=sum(r['event_windows']>0 for r in rows),
            videos_with_defined_phi=len(phis),mean_video_phi=sum(phis)/len(phis) if phis else None)
    report=dict(complete=True,scope='development-only descriptive association; no causal sign, detector accuracy, held-out result or tuned weights',
        settings={'probability':.5,'positive_rise':.1,'ad_tolerance_seconds':2.,'ad_track':'US','ad_merge_gap':.5,
                  'event_grid_seconds':1.,'tagger_window_seconds':10.,'multiple_testing':'no inferential p-values or significance claims'},
        split_sha256=sha(args.split_config),annotations_sha256=sha(args.data_dir/'validation.jsonl'),
        models=models,videos=records,per_class=summaries,
        caveat='Overlapping tagger windows and dense AD starts are correlated. Undefined per-video phi is explicit; do not infer trigger/suppress policy from sparse classes.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'videos':len(records),'classes':len(summaries),'classes_with_events':sum(r['event_windows']>0 for r in summaries.values())}))


if __name__=='__main__':main()
