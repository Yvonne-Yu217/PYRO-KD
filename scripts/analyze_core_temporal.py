"""Execute the explicitly declared core temporal family after complete export."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_paired import analyze
from src.evaluation.significance import holm

def run_family(rows,plan):
    conditions={f'{s}-{p}' for s in ('U','V','A_cue','AV','O') for p in ('P0','P1')}
    if plan['metric']!='temporal_f1' or plan['unit']!='cluster' or len(plan['contrasts'])!=17:
        raise ValueError('declared17 movie-clustered temporalF1 contrasts required')
    memberships={c:[r['example_id'] for r in rows if r['condition']==c] for c in conditions}
    reference=set(next(iter(memberships.values())))
    if len(reference)!=30 or any(len(ids)!=30 or set(ids)!=reference for ids in memberships.values()) or len(rows)!=300:
        raise ValueError('complete30 clips x10 core conditions required')
    results=[]
    for contrast in plan['contrasts']:
        if contrast['a'] not in conditions or contrast['b'] not in conditions:raise ValueError('unknown planned condition')
        result=analyze(rows,contrast['a'],contrast['b'],plan['metric'],plan['unit'],plan['resamples'],plan['seed'])
        if result['excluded']:raise ValueError('undefined/missing temporal metrics prevent complete family')
        result['oracle_upper_bound']=contrast['oracle_upper_bound'];results.append(result)
    adjusted=holm([r['wilcoxon']['pvalue'] for r in results])
    for result,p in zip(results,adjusted):
        result['wilcoxon']['holm_pvalue']=p
        result['wilcoxon']['pvalue_adjustment']='Holm across all17 declared contrasts'
    return {'plan':plan,'results':results,'scope':'temporalF1 only; no matched-pair text significance claims'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--plan',type=Path,default=Path('configs/core_temporal_contrasts.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve prior family')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    side=args.input.with_suffix(args.input.suffix+'.provenance.json')
    provenance=json.loads(side.read_text())
    if provenance['output_sha256']!=digest(args.input):raise ValueError('export provenance mismatch')
    result=run_family([json.loads(line) for line in args.input.read_text().splitlines() if line.strip()],json.loads(args.plan.read_text()))
    result['input_sha256']=digest(args.input);result['export_provenance_sha256']=digest(side)
    result['plan_sha256']=digest(args.plan);result['analyzer_sha256']=digest(Path(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'contrasts':len(result['results']),'scope':result['scope']}))

if __name__=='__main__':main()
