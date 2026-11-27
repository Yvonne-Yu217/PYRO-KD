"""Freeze a label-blind, domain-stratified Video-MME long evaluation subset."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-domain',type=int,default=2)
    p.add_argument('--seed',type=int,default=0)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('subset already frozen; do not silently resample')
    if args.per_domain<1: raise ValueError('positive domain sample required')
    # Never materialize question text, options or answer columns during selection.
    rows=pq.read_table(args.annotations,columns=['video_id','videoID','duration','domain']).to_pylist()
    videos={}
    for row in rows:
        if row['duration']!='long': continue
        if row['video_id'] in videos and videos[row['video_id']]!=row:
            raise ValueError('inconsistent video metadata')
        videos[row['video_id']]=row
    selected=[]
    for domain in sorted({r['domain'] for r in videos.values()}):
        candidates=[r for r in videos.values() if r['domain']==domain]
        if len(candidates)<args.per_domain: raise ValueError('insufficient videos in domain')
        selected.extend(sorted(candidates,key=lambda r:(hashlib.sha256(f"{args.seed}:{r['video_id']}".encode()).hexdigest(),r['video_id']))[:args.per_domain])
    release=json.loads(args.release.read_text())
    record={'dataset':release['repo'],'revision':release['revision'],'official_split':'test',
            'use':'small held-out long-form QA evaluation subset; not a tuning set',
            'selection':'duration long; per-domain ascending SHA256(seed:video_id); no question/answer columns read',
            'seed':args.seed,'per_domain':args.per_domain,'eligible_videos':len(videos),
            'annotations_sha256':hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
            'videos':selected,'availability_policy':'retain selected IDs and report missing/unusable audio; no performance-based replacement'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(record,handle,indent=2)
    print(json.dumps({'selected':len(selected),'domains':len({r['domain'] for r in selected}),'eligible':len(videos)}))


if __name__=='__main__': main()
