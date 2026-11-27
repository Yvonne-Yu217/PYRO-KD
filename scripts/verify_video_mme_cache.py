"""Independently decode every cached JPEG and verify full WAV/hash coverage."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import wave
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from src.utils.media import load_manifest, cache_is_current
from scripts.preprocess_video_mme import sha


def check_frame(frame):
    path=Path(frame['path'])
    digest=sha(path)
    with Image.open(path) as image:
        image.load()
        if image.format!='JPEG' or min(image.size)<=0 or max(image.size)>768:
            raise ValueError('invalid JPEG or unexpected dimensions')
    return {'index':frame['index'],'timestamp':frame['timestamp'],'path':str(path),'sha256':digest}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset',type=Path,default=Path('configs/video_mme_long_subset.json'))
    p.add_argument('--preprocessing-report',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior verification; use a new path')
    config=json.loads(args.subset.read_text())
    source=json.loads(args.preprocessing_report.read_text())
    ids=[r['video_id'] for r in config['videos']]
    prepared={r['video_id']:r for r in source['results']}
    if not source['complete'] or source['subset_sha256']!=sha(args.subset) or set(ids)!=set(prepared):
        raise ValueError('complete matching preparation required')
    videos=[]
    for ident in ids:
        cache=args.cache_dir/ident
        manifest=load_manifest(cache)
        if not cache_is_current(manifest) or sha(cache/'manifest.json')!=prepared[ident]['manifest_sha256']:
            raise ValueError('stale or modified manifest')
        if sha(manifest['source'])!=prepared[ident]['source_sha256']:
            raise ValueError('source changed')
        frames=manifest['frames']
        if len(frames)!=prepared[ident]['frames'] or not frames:
            raise ValueError('frame coverage mismatch')
        if any(f['index']!=i or f['timestamp']!=i/2 for i,f in enumerate(frames)):
            raise ValueError('invalid frame grid')
        with ThreadPoolExecutor(max_workers=4) as pool:
            checked=list(pool.map(check_frame,frames))
        frame_digest=hashlib.sha256(json.dumps(checked,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        wav=Path(manifest['audio_path'])
        if sha(wav)!=prepared[ident]['audio_sha256']:
            raise ValueError('audio changed')
        with wave.open(str(wav),'rb') as stream:
            samples=stream.getnframes()
            if stream.getnchannels()!=1 or stream.getsampwidth()!=2 or stream.getframerate()!=16000:
                raise ValueError('unexpected WAV format')
            read_samples=0
            while chunk:=stream.readframes(16000*10):
                read_samples+=len(chunk)//2
            if read_samples!=samples:
                raise ValueError('truncated PCM data')
        duration=manifest['media_info']['duration']
        if abs(samples/16000-duration)>1 or abs(len(frames)/2-duration)>1:
            raise ValueError('duration coverage mismatch')
        row={'video_id':ident,'verified':True,'decoded_jpegs':len(frames),
             'decoded_frame_manifest_sha256':frame_digest,'pcm_samples_read':samples,'duration':duration}
        videos.append(row)
        print(json.dumps(row),flush=True)
    report={'complete':True,'scope':'all cached JPEGs decoded; full PCM read and hashed; source hashes checked; no semantic QA inspection',
            'subset_sha256':sha(args.subset),'preprocessing_report_sha256':sha(args.preprocessing_report),'videos':videos}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
        handle.write('\n')


if __name__=='__main__':
    main()
