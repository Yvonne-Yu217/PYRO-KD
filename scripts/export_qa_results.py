"""Publish fixed-denominator QA coverage and video-bootstrap accuracy tables."""
import argparse,csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.export_qa_paired_metrics import export_reports
from src.evaluation.stats import bootstrap_mean_ci

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['egoschema','video-mme'],required=True);p.add_argument('--report',nargs=2,action='append',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--scope',required=True);a=p.parse_args()
    reports=[];sources=[]
    for name,path in a.report:
        path=Path(path);raw=path.read_bytes();reports.append((name,json.loads(raw)));sources.append(dict(condition=name,path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest()))
    export_reports(reports,a.dataset)
    rows=[];failures={}
    for name,report in reports:
        groups=defaultdict(list)
        for r in report['rows']:groups[r['video_id']].append(float(r['correct']))
        if len({len(v) for v in groups.values()})!=1:raise ValueError('unequal questions/video: report both estimands explicitly')
        interval=bootstrap_mean_ci([sum(v)/len(v) for v in groups.values()],resamples=10000,seed=0)
        status=report['status_counts'];rows.append(dict(condition=name,videos=len(groups),expected=report['expected'],attempted=report['attempted'],valid=report['valid_predictions'],correct=report['correct'],accuracy=report['accuracy'],ci_low=interval['ci_low'],ci_high=interval['ci_high'],generation_errors=status['generation_error'],invalid_answers=status['invalid_answer'],missing=status['missing']))
        failures[name]=[dict(id=r['id'],video_id=r['video_id'],status=r['status']) for r in report['rows'] if r['status']!='valid']
    a.output_dir.mkdir(parents=True,exist_ok=True)
    payload=dict(dataset=a.dataset,scope=a.scope,rows=rows,sources=sources,failed_or_missing=failures,interval='Pointwise95% percentile bootstrap over videos,10000 resamples,seed0; not simultaneous intervals. Failed/missing/invalid predictions count incorrect.')
    (a.output_dir/'qa-results.json').write_text(json.dumps(payload,indent=2)+'\n')
    with (a.output_dir/'qa-results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    lines=['# QA results','',a.scope,'',payload['interval'],'','| Condition | Correct / expected | Accuracy | 95% CI | Generation error / invalid / missing |','|---|---:|---:|---|---|']
    for r in rows:lines.append(f"| {r['condition']} | {r['correct']}/{r['expected']} | {r['accuracy']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['generation_errors']} / {r['invalid_answers']} / {r['missing']} |")
    (a.output_dir/'qa-results.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,max(3,0.5*len(rows)+1)));means=[r['accuracy'] for r in rows]
    ax.errorbar(means,range(len(rows)),xerr=[[max(0,r['accuracy']-r['ci_low']) for r in rows],[max(0,r['ci_high']-r['accuracy']) for r in rows]],fmt='o',capsize=4,color='#176b91');ax.set_yticks(range(len(rows)),[r['condition'] for r in rows]);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Accuracy over all frozen questions (pointwise95% video-bootstrap CI)');ax.set_title(a.dataset+' QA: fixed denominator');ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(a.output_dir/f'qa-results.{suffix}',dpi=180)
    print(json.dumps({'dataset':a.dataset,'conditions':len(rows),'expected_per_condition':rows[0]['expected']}))
if __name__=='__main__':main()
