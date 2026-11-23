"""Verify frozen selected media against archive identity and actual audio probes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.download_video_mme_subset import checksum


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset',type=Path,default=Path('configs/video_mme_long_subset.json'))
    p.add_argument('--members',type=Path,required=True)
    p.add_argument('--videos',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--decode-samples',action='store_true',help='decode10s audio at10%,50%,90%; not full-media decode')
    args=p.parse_args()
    subset=json.loads(args.subset.read_text());members=json.loads(args.members.read_text())
    if subset['revision']!=members['revision']: raise ValueError('revision mismatch')
    mapped={r['video_id']:r for r in members['videos']}
    if set(mapped)!={r['video_id'] for r in subset['videos']}: raise ValueError('frozen membership mismatch')
    records=[]
    for selected in subset['videos']:
        video_id=selected['video_id'];path=args.videos/(video_id+'.mp4')
        try:
            member=mapped[video_id]['member']
            if member is None: raise ValueError('archive member unlocated')
            if mapped[video_id]['videoID']!=selected['videoID']: raise ValueError('source video identity mismatch')
            size,crc,digest=checksum(path)
            if size!=member['size'] or crc!=member['crc']: raise ValueError('archive size/CRC mismatch')
            previous=json.loads(path.with_suffix('.verified.json').read_text())
            if previous['sha256']!=digest: raise ValueError('download SHA mismatch')
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],text=True))
            duration=float(probe['format']['duration'])
            if not math.isfinite(duration) or duration<=0: raise ValueError('invalid duration')
            kinds=[s['codec_type'] for s in probe['streams']]
            audio='audio' in kinds;video='video' in kinds;samples=[]
            if args.decode_samples and audio:
                import numpy as np
                for fraction in (.1,.5,.9):
                    raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(duration*fraction),'-i',str(path),
                        '-map','0:a:0','-t','10','-ac','1','-ar','16000','-f','f32le','pipe:1'])
                    wave=np.frombuffer(raw,dtype='<f4')
                    if len(wave)<159000 or not np.isfinite(wave).all(): raise ValueError('short/nonfinite decoded audio sample')
                    samples.append({'fraction':fraction,'samples':len(wave),'rms':float(np.sqrt(np.mean(wave.astype(float)**2)))})
            records.append({'video_id':video_id,'verified':True,'bytes':size,'sha256':digest,'duration_seconds':duration,
                'has_video':video,'has_audio':audio,'within_30_to_60_minutes':1800<=duration<=3600,
                'audio_samples':samples,'sampled_nonzero_audio':any(s['rms']>1e-6 for s in samples) if samples else None})
        except Exception as exc:
            records.append({'video_id':video_id,'verified':False,'error_type':type(exc).__name__,'error':str(exc)})
        print(json.dumps(records[-1]),flush=True)
    complete=all(r['verified'] for r in records)
    report={'complete':complete,'expected':len(subset['videos']),'verified':sum(r['verified'] for r in records),
            'audio_gate_passed':complete and all(r['has_audio'] and r['has_video'] and r['within_30_to_60_minutes'] for r in records),
            'decode_scope':'three10s audio windows, not full audio/video decode' if args.decode_samples else 'stream metadata only',
            'subset_sha256':hashlib.sha256(args.subset.read_bytes()).hexdigest(),'videos':records}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if complete else 1


if __name__=='__main__': raise SystemExit(main())
