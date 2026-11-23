"""Audit frozen onset caches, with optional explicit full-WAV recomputation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from src.utils.onset_features import SETTINGS,compute_onsets
from src.utils.audio_features import resolve_audio_path
from src.utils.media import load_manifest,cache_is_current


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--subset',type=Path,required=True)
    p.add_argument('--recompute-video-id',action='append',default=[])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve prior audit')
    ids=[r['video_id'] for r in json.loads(args.subset.read_text())['videos']]
    if not ids or len(set(ids))!=len(ids) or not set(args.recompute_video_id)<=set(ids):
        raise ValueError('unique frozen IDs and in-subset recomputations required')
    records=[]
    for ident in ids:
        try:
            cache=args.cache_dir/ident;manifest=load_manifest(cache)
            if not cache_is_current(manifest):raise ValueError('stale source')
            path=cache/'onset_features.json';payload=json.loads(path.read_text())
            provenance=payload['provenance']
            if (not payload['complete'] or provenance['settings']!=SETTINGS
                    or payload['config_hash']!=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()):
                raise ValueError('onset settings/config mismatch')
            audio=resolve_audio_path(manifest,cache)
            if hashlib.sha256(audio.read_bytes()).hexdigest()!=provenance['audio_sha256']:
                raise ValueError('changed WAV')
            info=sf.info(audio)
            if info.samplerate!=16000 or info.channels!=1:raise ValueError('audio format mismatch')
            expected=np.arange(int(np.ceil(info.frames/info.samplerate)),dtype=float)
            times,raw,normalized=[np.asarray(payload[key],dtype=float) for key in ('times','strength','normalized')]
            if (not np.array_equal(times,expected) or raw.shape!=times.shape or normalized.shape!=times.shape
                    or not all(np.isfinite(x).all() for x in (raw,normalized))
                    or (raw<0).any() or (normalized<0).any() or (normalized>1).any()):
                raise ValueError('invalid onset grid or values')
            if not np.allclose(normalized,raw/(raw.max() or 1),rtol=1e-12,atol=1e-12):
                raise ValueError('normalization mismatch')
            recomputed=ident in args.recompute_video_id
            if recomputed:
                wave,sr=sf.read(audio,dtype='float32');fresh=compute_onsets(wave,sr)
                for key in ('times','strength','normalized'):
                    if not np.allclose(fresh[key],payload[key],rtol=1e-7,atol=1e-8):
                        raise ValueError('full-WAV recomputation differs: '+key)
            records.append(dict(video_id=ident,verified=True,windows=len(times),
                full_wave_recomputed=recomputed,features_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        except Exception as exc:
            records.append(dict(video_id=ident,verified=False,error=repr(exc)))
    report=dict(complete=all(r['verified'] for r in records),expected=len(ids),
                verified=sum(r['verified'] for r in records),videos=records,
                subset_sha256=hashlib.sha256(args.subset.read_bytes()).hexdigest(),
                scope='source/config/grid/value/normalization checks; full-WAV recomputation only where explicitly marked; no semantic onset accuracy')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='videos'}))
    raise SystemExit(0 if report['complete'] else 1)


if __name__=='__main__':main()
