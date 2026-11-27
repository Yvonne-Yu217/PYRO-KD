"""Prepare the frozen long-video subset without reading QA text or labels.

Requires the successful independent download audit. Existing matching caches
are reusable; raw MP4 hashes are checked before preprocessing. The report is
checkpointed after each video and only complete when every frozen ID passes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import wave
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.media import preprocess_video, cache_is_complete, cache_is_current


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validated_ids(config, audit, config_hash):
    ids=[row['video_id'] for row in config['videos']]
    if not ids or len(set(ids))!=len(ids) or any(not isinstance(i,str) or not i.isdigit() for i in ids):
        raise ValueError('unique numeric frozen video IDs required')
    if config['official_split']!='test' or any(row['duration']!='long' for row in config['videos']):
        raise ValueError('expected frozen long-form test subset')
    if not audit['complete'] or not audit['audio_gate_passed'] or audit['subset_sha256']!=config_hash:
        raise ValueError('matching completed audio-gated download audit required')
    records={row['video_id']:row for row in audit['videos']}
    if len(records)!=len(audit['videos']) or set(records)!=set(ids) or any(not row['verified'] for row in records.values()):
        raise ValueError('audit IDs must exactly cover frozen subset')
    return ids,records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset',type=Path,default=Path('configs/video_mme_long_subset.json'))
    p.add_argument('--verification',type=Path,required=True)
    p.add_argument('--video-dir',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    subset_hash=sha(args.subset)
    ids,verified=validated_ids(json.loads(args.subset.read_text()),json.loads(args.verification.read_text()),subset_hash)
    raw=args.video_dir.resolve()
    for destination in (args.cache_dir.resolve(),args.report.resolve()):
        if destination==raw or raw in destination.parents:
            raise ValueError('derived data cannot be written into raw video directory')
    report={'complete':False,'subset_sha256':subset_hash,'verification_sha256':sha(args.verification),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'implementation_sha256':sha(__file__),'fps':2.0,'max_side':768,'sample_rate':16000,
            'scope':'media preprocessing only; no QA text, answer labels or semantic inspection',
            'results':[]}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    def checkpoint():
        temporary=args.report.with_suffix('.incoming.json')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(args.report)
    checkpoint()
    for ident in ids:
        source=raw/f'{ident}.mp4'
        print(json.dumps({'video_id':ident,'state':'checking_source_and_preparing'}),flush=True)
        try:
            if sha(source)!=verified[ident]['sha256']:
                raise ValueError('raw video hash differs from verified download')
            manifest=preprocess_video(source,args.cache_dir/ident,fps=2.0,max_side=768,sample_rate=16000)
            duration=manifest['media_info']['duration']
            with wave.open(manifest['audio_path'],'rb') as wav:
                if wav.getnchannels()!=1 or wav.getframerate()!=16000 or wav.getsampwidth()!=2:
                    raise ValueError('unexpected PCM audio format')
                audio_seconds=wav.getnframes()/wav.getframerate()
            frames=manifest['frames']
            if not cache_is_complete(manifest) or not cache_is_current(manifest):
                raise ValueError('incomplete or stale cache')
            if abs(audio_seconds-duration)>1.0 or abs(len(frames)/2-duration)>1.0:
                raise ValueError('audio/frame coverage differs from source duration by >1s')
            if not all(math.isclose(frame['timestamp'],i/2,abs_tol=1e-6) for i,frame in enumerate(frames)):
                raise ValueError('unexpected sampled frame timestamps')
            row={'video_id':ident,'complete':True,'frames':len(frames),'duration':duration,
                 'audio_seconds':audio_seconds,'source_sha256':verified[ident]['sha256'],
                 'manifest_sha256':sha(args.cache_dir/ident/'manifest.json'),
                 'audio_sha256':sha(manifest['audio_path'])}
        except Exception as exc:
            row={'video_id':ident,'complete':False,'error':repr(exc)}
        report['results'].append(row)
        checkpoint()
        print(json.dumps(row),flush=True)
    report['complete']=len(report['results'])==len(ids) and all(row['complete'] for row in report['results'])
    checkpoint()
    raise SystemExit(0 if report['complete'] else 1)


if __name__=='__main__':
    main()
