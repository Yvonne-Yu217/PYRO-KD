"""Cache frames for the frozen silent EgoSchema pilot without loading QA text."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.media import preprocess_video,cache_is_complete,cache_is_current
from scripts.preprocess_video_mme import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/egoschema_pilot.json'))
    p.add_argument('--video-dir',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    config=json.loads(args.config.read_text())
    ids=config['pilot_q_uids']
    if len(ids)!=10 or len(set(ids))!=10 or set(ids)&set(config['heldout_q_uids']):
        raise ValueError('expected10 distinct pilot IDs excluded from held-out IDs')
    if any(Path(i).name!=i or i in ('.','..') for i in ids): raise ValueError('invalid ID')
    raw=args.video_dir.resolve()
    for destination in (args.cache_dir.resolve(),args.report.resolve()):
        if destination==raw or raw in destination.parents: raise ValueError('raw media is read-only')
    report={'complete':False,'config_sha256':sha(args.config),'scope':__doc__,'results':[]}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    for ident in ids:
        source=raw/(ident+'.mp4')
        try:
            digest=sha(source)
            manifest=preprocess_video(source,args.cache_dir/ident,fps=2.,max_side=768)
            if not cache_is_current(manifest) or not cache_is_complete(manifest): raise ValueError('stale/incomplete cache')
            if manifest['audio_path'] is not None or manifest['media_info']['has_audio']:
                raise ValueError('unexpected audio; re-evaluate silent-EgoSchema protocol')
            if abs(len(manifest['frames'])/2-manifest['media_info']['duration'])>1:
                raise ValueError('frame duration coverage mismatch')
            result={'video_id':ident,'complete':True,'frames':len(manifest['frames']),
                    'duration':manifest['media_info']['duration'],'has_audio':False,
                    'source_sha256':digest,'manifest_sha256':sha(args.cache_dir/ident/'manifest.json')}
        except Exception as exc: result={'video_id':ident,'complete':False,'error':repr(exc)}
        report['results'].append(result)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    report['complete']=all(row['complete'] for row in report['results'])
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['complete'] else 1)


if __name__=='__main__':main()
