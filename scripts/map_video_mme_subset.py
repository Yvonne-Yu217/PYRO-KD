"""Map frozen Video-MME IDs to a pinned ZIP inventory without resampling."""
import argparse
import hashlib
import json
from pathlib import Path


def map_subset(subset,inventory,allow_missing=False):
    if subset['dataset']!=inventory['repo'] or subset['revision']!=inventory['revision']:
        raise ValueError('subset and archive release identities differ')
    rows=[]
    for selected in subset['videos']:
        matches=[dict(member,archive=archive['archive'],archive_bytes=archive['archive_bytes'])
                 for archive in inventory['archives'] for member in archive['members']
                 if Path(member['name']).suffix.lower()=='.mp4' and Path(member['name']).stem==selected['videoID']]
        if len(matches)>1: raise ValueError('ambiguous archive member for frozen ID')
        if not matches and not allow_missing: raise ValueError(f"frozen ID {selected['video_id']} is not located")
        rows.append({**selected,'member':matches[0] if matches else None})
    return {'repo':inventory['repo'],'revision':inventory['revision'],'videos':rows,
            'complete':all(row['member'] is not None for row in rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subset',type=Path,default=Path('configs/video_mme_long_subset.json'))
    p.add_argument('--inventory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-missing',action='store_true')
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('preserve prior member map')
    result=map_subset(json.loads(args.subset.read_text()),json.loads(args.inventory.read_text()),args.allow_missing)
    result['source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.subset,args.inventory)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(result,handle,indent=2)
    print(json.dumps({'complete':result['complete'],'videos':len(result['videos'])}))


if __name__=='__main__': main()
