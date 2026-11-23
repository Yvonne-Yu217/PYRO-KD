"""Freeze a question-blind pilot from public label membership, without using values."""
import argparse
import hashlib
import json
from pathlib import Path
import random


def freeze(ids, count=10, seed=0):
    ordered=sorted(ids)
    if len(ordered)!=len(set(ordered)) or any(not isinstance(i,str) or not i for i in ordered):
        raise ValueError('unique nonempty IDs required')
    if isinstance(count,bool) or not isinstance(count,int) or not 0<count<len(ordered):
        raise ValueError('pilot must leave a nonempty held-out set')
    chosen=set(random.Random(seed).sample(ordered,count))
    return [i for i in ordered if i in chosen],[i for i in ordered if i not in chosen]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--answers',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('preserve the frozen pilot')
    raw=args.answers.read_bytes()
    membership=json.loads(raw)
    if not isinstance(membership,dict) or len(membership)!=500:
        raise ValueError('expected public500 membership')
    pilot,heldout=freeze(membership.keys())
    record={'dataset':'egoschema public subset','seed':0,'pilot_q_uids':pilot,
            'heldout_q_uids':heldout,'num_heldout':len(heldout),
            'answers_file_sha256':hashlib.sha256(raw).hexdigest(),
            'selection':'sorted public label membership; random.Random(0).sample10; no question or label values used',
            'note':'pilot items are excluded from reported local evaluation; silent videos support visual-only QA'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(record,handle,indent=2)
    print(json.dumps({'pilot':len(pilot),'heldout':len(heldout),'output':str(args.output)}))


if __name__=='__main__':main()
