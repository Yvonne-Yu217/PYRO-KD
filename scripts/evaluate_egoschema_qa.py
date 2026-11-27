"""Score actual EgoSchema predictions against a frozen pilot or held-out partition.

Answer labels are loaded only here, never by the QA input builder or runner.
No score is produced for an absent/empty prediction file. Partial coverage is
reported explicitly and exits with status1 rather than claiming completion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.qa import score_qa


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--answers',type=Path,required=True)
    p.add_argument('--pilot-config',type=Path,default=Path('configs/egoschema_pilot.json'))
    p.add_argument('--split',choices=('pilot','heldout'),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('use a new evaluation output path')
    # Fail before loading labels if no actual predictions exist. Malformed lines
    # are not silently skipped; repair/recover interrupted files explicitly.
    predictions=[json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not predictions: raise ValueError('no predictions to evaluate')
    if any(not isinstance(row,dict) for row in predictions): raise ValueError('prediction records must be objects')
    config=json.loads(args.pilot_config.read_text())
    pilot,heldout=config['pilot_q_uids'],config['heldout_q_uids']
    if len(pilot)!=10 or len(set(pilot))!=10 or len(heldout)!=490 or len(set(heldout))!=490 or set(pilot)&set(heldout):
        raise ValueError('invalid frozen10/490 partition')
    if sha(args.answers)!=config['answers_file_sha256']: raise ValueError('label file changed since partition freeze')
    labels=json.loads(args.answers.read_text())
    if set(labels)!=set(pilot)|set(heldout): raise ValueError('frozen partition does not match public labels')
    ids=pilot if args.split=='pilot' else heldout
    if any(row.get('dataset')!='egoschema' or row.get('split')!=args.split or not row.get('run_fingerprint') for row in predictions):
        raise ValueError('dataset/split/provenance mismatch')
    result=score_qa(predictions,{ident:labels[ident] for ident in ids},5)
    result.update(dataset='egoschema',split=args.split,pilot_excluded=args.split=='heldout',
                  input_sha256=sha(args.input),config_sha256=sha(args.pilot_config),answers_sha256=sha(args.answers))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps({key:result[key] for key in ('complete','expected','attempted','correct','accuracy','status_counts')}))
    raise SystemExit(0 if result['complete'] else 1)


if __name__=='__main__':main()
