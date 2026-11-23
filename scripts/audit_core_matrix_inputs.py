"""Inventory development input coverage for all 20 core cells, without freezing variants."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

VARIANTS = {
    'U': ['U_snapped'], 'V': ['V_local','V_memory'],
    'A': ['A_trigger','A_suppress'], 'AV': ['AV_trigger','AV_suppress'],
    'O': ['O_snapped'],
}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--moments-dir',type=Path,required=True)
    p.add_argument('--split-config',type=Path,default=Path('configs/reframed_splits.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve prior readiness audit')
    split=json.loads(args.split_config.read_text())
    if split['split']!='validation':raise ValueError('development inventory only')
    ids={r['video_id'] for r in split['clips']}
    variants={}
    for selector,names in VARIANTS.items():
        for name in names:
            path=args.moments_dir/f'reframed_{name}_placed_event_dev_exploratory.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('unique nonempty input required')
            counts={ident:0 for ident in ids}
            requested={}
            for row in rows:
                if row['video_id'] not in ids:raise ValueError('input outside frozen development membership')
                if len(row['frames'])!=4 or len(row['frame_times'])!=4:raise ValueError('four-frame protocol required')
                if any(not Path(frame).is_file() for frame in row['frames']):raise ValueError('missing selected frame')
                if row['word_limit']<1:raise ValueError('invalid word budget')
                ident=row['video_id'];counts[ident]+=1
                if ident in requested and requested[ident]!=row['k_requested']:raise ValueError('inconsistent K')
                requested[ident]=row['k_requested']
            variants[name]=dict(selector=selector,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                moments=len(rows),videos_with_moments=sum(v>0 for v in counts.values()),
                                counts=counts,requested_for_present_videos=requested,
                                missing_video_ids=sorted(k for k,v in counts.items() if not v),
                                status='exploratory development inputs; not final frozen selector')
    cells=[]
    for selector,prior,model in itertools.product(VARIANTS,('P0','P1'),('open','api')):
        gates=['real end-to-end GPU smoke','final dev-tuned selector and fixed-K protocol',
               'frozen evaluation inputs','actual predictions and evaluated outputs']
        if model=='api':gates+=['explicit API spending cap','frozen API model and budget-enforcing runner']
        cells.append(dict(condition=f'{selector}-{prior}-{model}',selector=selector,prior=prior,model_family=model,
                          development_variants=VARIANTS[selector],execution_status='not assessed by input inventory',required_gates=gates))
    assert len(cells)==20 and len({r['condition'] for r in cells})==20
    report=dict(scope='development input inventory; no selector choice, experiment execution or results',
                selection_config_sha256=hashlib.sha256(args.split_config.read_bytes()).hexdigest(),
                expected_development_videos=len(ids),variants=variants,cells=cells,
                conditions=20,experiment_outputs_inspected=False,
                note='API conditions are retained; native Omni is an ablation, not a replacement API condition')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'conditions':20,'variants':{k:{'moments':v['moments'],'videos':v['videos_with_moments']} for k,v in variants.items()}}))


if __name__=='__main__':main()
