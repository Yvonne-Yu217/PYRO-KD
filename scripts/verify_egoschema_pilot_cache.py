"""Decode all frozen EgoSchema pilot frames and recheck source identity/silence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.preprocess_video_mme import sha
from scripts.verify_video_mme_cache import check_frame
from src.utils.media import load_manifest,cache_is_current,probe


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/egoschema_pilot.json'))
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--preprocessing-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('preserve prior audits')
    config=json.loads(args.config.read_text());prepared=json.loads(args.preprocessing_report.read_text())
    by_id={r['video_id']:r for r in prepared['results']}
    if not prepared['complete'] or prepared['config_sha256']!=sha(args.config) or set(by_id)!=set(config['pilot_q_uids']):
        raise ValueError('incomplete or mismatched frozen pilot')
    records=[]
    for ident in config['pilot_q_uids']:
        cache=args.cache_dir/ident;manifest=load_manifest(cache)
        if not cache_is_current(manifest) or sha(cache/'manifest.json')!=by_id[ident]['manifest_sha256'] or sha(manifest['source'])!=by_id[ident]['source_sha256']:
            raise ValueError('stale or modified media/cache')
        info=probe(manifest['source'])
        if info.has_audio or manifest['audio_path'] is not None: raise ValueError('unexpected audio')
        frames=manifest['frames']
        if not frames or abs(len(frames)/2-info.duration)>1 or any(f['index']!=i or f['timestamp']!=i/2 for i,f in enumerate(frames)):
            raise ValueError('invalid frame grid or coverage')
        with ThreadPoolExecutor(max_workers=4) as pool: checked=list(pool.map(check_frame,frames))
        records.append({'video_id':ident,'verified':True,'decoded_jpegs':len(checked),'has_audio':False})
    result={'complete':True,'config_sha256':sha(args.config),'scope':__doc__,'videos':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(result,handle,indent=2)
    print(json.dumps({'complete':True,'videos':len(records),'frames':sum(r['decoded_jpegs'] for r in records)}))


if __name__=='__main__':main()
