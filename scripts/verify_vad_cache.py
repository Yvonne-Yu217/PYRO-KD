"""Validate frozen-subset VAD cache identity and speech/gap interval arithmetic.

This checks mechanics, not detector accuracy against human speech labels.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import wave
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.audio_features import load_current_features,resolve_audio_path,DEFAULT_CONFIG
from src.utils.media import load_manifest,cache_is_current


def validate_intervals(features, duration):
    if not math.isfinite(duration) or duration<=0 or abs(features['duration']-duration)>.001:
        raise ValueError('audio duration mismatch')
    speech=features['speech']
    previous=0.
    for start,end in speech:
        if not all(math.isfinite(x) for x in (start,end)) or start<previous or not 0<=start<end<=duration+.001:
            raise ValueError('invalid, overlapping or out-of-bounds speech intervals')
        previous=end
    if abs(sum(end-start for start,end in speech)-features['speech_seconds'])>.001:
        raise ValueError('speech duration sum mismatch')
    # Independent complement from persisted intervals; allow only serialization precision.
    expected=[]
    cursor=0.
    cfg=features['config']
    for index,(start,end) in enumerate(speech+[[duration,duration]]):
        left=cursor+(cfg['edge_margin_seconds'] if cursor>0 else 0)
        right=start-(cfg['edge_margin_seconds'] if index<len(speech) else 0)
        if right-left>=cfg['min_gap_seconds']:
            expected.append((left,right))
        cursor=max(cursor,end)
    if len(expected)!=len(features['gaps']):
        raise ValueError('usable gap count differs from speech complement')
    for gap,(start,end) in zip(features['gaps'],expected):
        if abs(gap['start']-start)>.001 or abs(gap['end']-end)>.001:
            raise ValueError('gap boundaries differ from speech complement')
        if any(not math.isfinite(gap[key]) or gap[key]<0 for key in ('rms','onset')):
            raise ValueError('nonfinite or negative acoustic features')
    return {'speech_intervals':len(speech),'speech_seconds':features['speech_seconds'],
            'usable_gaps':len(expected),'usable_gap_seconds':sum(e-s for s,e in expected)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('use a new audit path')
    subset=json.loads(args.subset.read_text())
    ids=[r['video_id'] for r in subset['videos']]
    if not ids or len(ids)!=len(set(ids)): raise ValueError('unique frozen subset required')
    records=[]
    for ident in ids:
        if Path(ident).name!=ident or ident in ('.','..'): raise ValueError('invalid video ID')
        cache=args.cache_dir/ident
        manifest=load_manifest(cache)
        if not cache_is_current(manifest): raise ValueError('stale source media')
        features,reason=load_current_features(cache,manifest)
        if features is None: raise ValueError(f'{ident}: {reason}')
        if features['config']!=DEFAULT_CONFIG: raise ValueError('unexpected VAD protocol')
        with wave.open(str(resolve_audio_path(manifest,cache)),'rb') as wav:
            duration=wav.getnframes()/wav.getframerate()
        record={'video_id':ident,'verified':True,'duration':duration,
                'feature_sha256':hashlib.sha256((cache/'audio_features.json').read_bytes()).hexdigest(),
                **validate_intervals(features,duration)}
        records.append(record)
        print(json.dumps(record),flush=True)
    result={'complete':True,'scope':__doc__,'subset_sha256':hashlib.sha256(args.subset.read_bytes()).hexdigest(),
            'config':DEFAULT_CONFIG,'videos':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(result,handle,indent=2,allow_nan=False)


if __name__=='__main__': main()
