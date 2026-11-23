"""Publish fixed-denominator QA coverage and video-bootstrap accuracy tables."""
import argparse,csv,hashlib,json,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.export_qa_paired_metrics import export_reports
from src.evaluation.stats import bootstrap_mean_ci

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',choices=['egoschema','video-mme'],required=True);p.add_argument('--report',nargs=2,action='append',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--scope',required=True);p.add_argument('--qa-input',nargs=2,action='append',default=[]);a=p.parse_args()
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
    if a.qa_input:
        import math
        input_paths=dict(a.qa_input)
        if len(input_paths)!=len(a.qa_input) or set(input_paths)!={name for name,_ in reports}:raise ValueError('one input file per condition required')
        length_rows=[];input_sources=[]
        for name,report in reports:
            path=Path(input_paths[name]);raw=path.read_bytes();inputs=[json.loads(x) for x in raw.splitlines()];by_id={r['id']:r for r in inputs}
            if len(by_id)!=len(inputs) or set(by_id)!={r['id'] for r in report['rows']}:raise ValueError('QA input membership mismatch')
            grouped=defaultdict(list)
            for r in report['rows']:grouped[r['video_id']].append(r)
            for video_id,items in grouped.items():
                meta=[by_id[r['id']] for r in items];durations={r['duration'] for r in meta};framecounts={len(r['frames']) for r in meta}
                if len(durations)!=1 or len(framecounts)!=1 or any(r['video_id']!=video_id for r in meta):raise ValueError('within-video input mismatch')
                duration=durations.pop()
                if not math.isfinite(duration) or duration<=0:raise ValueError('positive duration required')
                length_rows.append(dict(condition=name,video_id=video_id,duration_minutes=duration/60,frames=framecounts.pop(),questions=len(items),correct=sum(r['correct'] for r in items),accuracy=sum(r['correct'] for r in items)/len(items)))
            input_sources.append(dict(condition=name,path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest()))
        with (a.output_dir/'qa-by-length.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=length_rows[0]);w.writeheader();w.writerows(length_rows)
        (a.output_dir/'qa-by-length.json').write_text(json.dumps(dict(rows=length_rows,inputs=input_sources,scope='Descriptive per-video accuracy versus measured duration; no fitted trend or length-dependent causal claim. Frame counts exposed for constrained-budget conditions.'),indent=2)+'\n')
        columns=min(3,len(reports));nrows=math.ceil(len(reports)/columns);fig,axes=plt.subplots(nrows,columns,figsize=(4.5*columns,2.8*nrows),squeeze=False,sharex=True,sharey=True)
        for ax,(name,_) in zip(axes.flat,reports):
            subset=[r for r in length_rows if r['condition']==name];ax.scatter([r['duration_minutes'] for r in subset],[r['accuracy'] for r in subset],color='#176b91');ax.set_title(name);ax.set_ylim(-.05,1.05);ax.set_xlabel('Video duration (minutes)');ax.set_ylabel('Accuracy');ax.spines[['top','right']].set_visible(False)
        for ax in list(axes.flat)[len(reports):]:ax.set_visible(False)
        fig.tight_layout()
        for suffix in ['png','pdf']:fig.savefig(a.output_dir/f'qa-by-length.{suffix}',dpi=180)
    print(json.dumps({'dataset':a.dataset,'conditions':len(rows),'expected_per_condition':rows[0]['expected']}))
if __name__=='__main__':main()
