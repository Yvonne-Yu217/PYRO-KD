#!/usr/bin/env python
"""Preprocess an existing frozen REFRAMED validation selection without downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.reframed import load_split
from src.utils.media import cache_is_complete, cache_is_current, preprocess_video


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/reframed_splits.json'))
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    if config.get('split') != 'validation':
        raise ValueError('This preparation accepts only the frozen validation selection')
    ids = [row['video_id'] for row in config['clips']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Selection must contain unique nonempty development IDs')
    clips = {c.video_id:c for c in load_split(args.data_dir,'validation')}
    if any(i not in clips for i in ids):
        raise ValueError('Selection contains an ID outside validation')
    raw = (args.data_dir/'videos').resolve()
    for destination in (args.cache_dir.resolve(), args.report.resolve()):
        if destination == raw or raw in destination.parents:
            raise ValueError('Derived artifacts must not be written into raw videos')
    report = {'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              'config_sha256':hashlib.sha256(config_bytes).hexdigest(),
              'data_root':str(args.data_dir),'cache_root':str(args.cache_dir),
              'fps':2.0,'results':[],'complete':False}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    for video_id in ids:
        try:
            manifest = preprocess_video(args.data_dir/'videos'/f'{video_id}.mp4',args.cache_dir/video_id,fps=2.0)
            current = cache_is_current(manifest)
            complete = cache_is_complete(manifest)
            info = manifest['media_info']
            result = {'video_id':video_id,'complete':complete,'current':current,
                      'frames':len(manifest['frames']),'has_audio':bool(manifest['audio_path']),
                      'duration':info['duration'],
                      'annotation_duration':clips[video_id].video_duration,
                      'duration_drift':info['duration']-clips[video_id].video_duration}
        except Exception as exc:
            result = {'video_id':video_id,'complete':False,'error':repr(exc)}
        report['results'].append(result)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    report['complete'] = all(r['complete'] and r.get('current') and r.get('has_audio') for r in report['results'])
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['complete'] else 1)


if __name__ == '__main__':
    main()
