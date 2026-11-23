#!/usr/bin/env python3
"""Mix existing timestamped narration WAVs into full-clip videos; preserve failures and overlaps."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
from review_tool import CONTRASTS, dump, load


def render(input_path, output):
    if output.exists():
        raise ValueError('Output exists; use a new directory.')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise ValueError('ffmpeg and ffprobe are required.')
    data = load(input_path)
    labels = sorted({x for pair in CONTRASTS.values() for x in pair})
    checked = []
    for clip in data['clips']:
        def resolve(value):
            p = Path(value)
            p = p if p.is_absolute() else input_path.parent/p
            if not p.is_file():
                raise ValueError(f'Missing input: {p}')
            return p.resolve()
        source = resolve(clip['original'])
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(source)]))
        duration = float(probe['format']['duration'])
        if not math.isfinite(duration) or duration <= 0 or not any(s['codec_type']=='video' for s in probe['streams']):
            raise ValueError('Source must have video and a positive finite duration.')
        has_audio = any(s['codec_type']=='audio' for s in probe['streams'])
        variants = {}
        for label in labels:
            entries = clip['narrations'][label]  # [] explicitly means no valid rendered captions
            variants[label] = []
            for row in entries:
                t = row['time']
                if type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t < duration:
                    raise ValueError('Narration time must be within the clip.')
                speed = row.get('playback_speed', 1.2)
                if type(speed) not in (int, float) or not math.isfinite(speed) or not 0.5 <= speed <= 2:
                    raise ValueError('Playback speed must be in [0.5, 2].')
                variants[label].append((t, resolve(row['wav']), speed))
        checked.append((clip, source, duration, has_audio, variants))
    output.mkdir(parents=True)
    clips = []
    for n, (clip, source, duration, has_audio, variants) in enumerate(checked):
        record = {'clip_id':clip['clip_id'],'movie_id':clip['movie_id'],'original':str(source),'variants':{}}
        for label, events in variants.items():
            target = output/f'{n:03d}-{label}.mp4'
            cmd = ['ffmpeg','-nostdin','-v','error','-n','-i',str(source)]
            for _, wav, _ in events:
                cmd += ['-i',str(wav)]
            filters = [('[0:a]aresample=48000,aformat=channel_layouts=stereo,volume=0.5[base]' if has_audio else
                        f'anullsrc=r=48000:cl=stereo,atrim=duration={duration}[base]')]
            for i, (t, _, speed) in enumerate(events,1):
                filters.append(f'[{i}:a]aresample=48000,aformat=channel_layouts=stereo,atempo={speed},volume=0.5,adelay={round(t*1000)}:all=1[a{i}]')
            inputs='[base]'+''.join(f'[a{i}]' for i in range(1,len(events)+1))
            filters.append(f'{inputs}amix=inputs={len(events)+1}:duration=first:normalize=0,alimiter=limit=0.95:level=false:latency=true[out]')
            cmd += ['-filter_complex',';'.join(filters),'-map','0:v:0','-map','[out]',
                    '-map_metadata','-1','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
                    '-c:a','aac','-b:a','192k','-t',str(duration),'-movflags','+faststart',str(target)]
            subprocess.run(cmd,check=True)
            record['variants'][label]=str(target.resolve())
        clips.append(record)
    dump(output/'clips.json',{'clips':clips})
    dump(output/'render-receipt.json',{'source_manifest':str(input_path),'clips':len(clips),
         'original_gain':0.5,'narration_gain':0.5,'limiter':0.95,'note':'Per-row playback_speed applied; default 1.2 for the original unaccelerated Piper WAVs. No gap repair or per-track loudness normalization. Overlaps retained; tails beyond clip end trimmed.'})
    return output/'clips.json'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        print(render(a.input.resolve(),a.output.resolve()))
    except (ValueError,KeyError,OSError,TypeError,subprocess.CalledProcessError) as exc:
        p.exit(1,f'Error: {exc}\n')
