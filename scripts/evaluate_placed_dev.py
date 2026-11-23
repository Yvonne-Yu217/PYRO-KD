"""Exploratory REFRAMED development placement metrics from frozen selector reports.

No test annotations or model inference. Preserve shortfalls and raw matches;
these untuned diagnostics cannot establish final selector superiority.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.datasets import reframed
from src.evaluation.metrics import temporal_precision_recall_f1
from src.evaluation.stats import bootstrap_mean_ci,paired_bootstrap_diff


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def score_condition(times, truth, requested):
    if len(times)>requested: raise ValueError('selection exceeds requested budget')
    metrics=temporal_precision_recall_f1(times,truth,2.0)
    return {**metrics,'requested':requested,'selected':len(times),'shortfall':requested-len(times),
            'matched_per_requested':metrics['tp']/requested if requested else None,
            'times':times,'f1_by_tolerance':{str(t):temporal_precision_recall_f1(times,truth,t)['f1'] for t in (0,1,2,3,5)}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--fused-report',type=Path,required=True)
    p.add_argument('--visual-report',type=Path,required=True)
    p.add_argument('--baseline-report',type=Path)
    p.add_argument('--split-config',type=Path,default=Path('configs/reframed_splits.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('preserve prior evaluation; use a new path')
    splits=json.loads(args.split_config.read_text())
    ids=[r['video_id'] for r in splits['clips']]
    fused=json.loads(args.fused_report.read_text()); visual=json.loads(args.visual_report.read_text())
    baseline=json.loads(args.baseline_report.read_text()) if args.baseline_report else None
    for report in ([fused,visual]+([baseline] if baseline else [])):
        if report['split_sha256']!=digest(args.split_config): raise ValueError('selection split mismatch')
        if len(report['videos'])!=len(ids) or {r['video_id'] for r in report['videos']}!=set(ids):
            raise ValueError('incomplete or duplicate selection coverage')
    if baseline and any(baseline[k]!=fused[k] for k in ('required_words','max_advance','max_lag')):
        raise ValueError('baseline placement settings differ')
    baseline_by={r['video_id']:r for r in baseline['videos']} if baseline else {}
    vsettings=visual['settings']
    if any(fused[k]!=vsettings[k] for k in ('required_words','max_advance','max_lag')):
        raise ValueError('placement settings differ between reports')
    fused_by={r['video_id']:r for r in fused['videos']}
    visual_by={r['video_id']:r for r in visual['videos']}
    clips={c.video_id:c for c in reframed.load_split(args.data_dir,'validation')}
    movie_ids=[clips[i].movie_id for i in ids]
    if len(set(movie_ids))!=len(movie_ids):
        raise ValueError('this dev bootstrap requires one clip per movie; use clustered analysis otherwise')
    rows=[]
    for video_id in ids:
        clip=clips[video_id]
        truth=[item.start for item in reframed.merge_cues(clip.ad_us,max_gap=.5)]
        conditions={**fused_by[video_id]['conditions'],
                    **{'V_'+name:value for name,value in visual_by[video_id]['modes'].items()}}
        if baseline:
            conditions.update(baseline_by[video_id]['conditions'])
        if len({c['requested'] for c in conditions.values()})!=1: raise ValueError('requested K differs')
        for name,selected in conditions.items():
            times=[s['placement']['time'] for s in selected['selected']]
            rows.append({'video_id':video_id,'movie_id':clip.movie_id,'condition':name,
                         'ground_truth_times':truth,'ground_truth_out_of_bounds':[t for t in truth if not 0<=t<=clip.video_duration],
                         **score_condition(times,truth,selected['requested'])})
    names=sorted({r['condition'] for r in rows})
    summaries={}
    for name in names:
        group=[r for r in rows if r['condition']==name]
        tp,fp,fn=(sum(r[key] for r in group) for key in ('tp','fp','fn'))
        summaries[name]={'videos':len(group),'requested':sum(r['requested'] for r in group),
            'selected':sum(r['selected'] for r in group),'shortfall':sum(r['shortfall'] for r in group),
            'tp':tp,'fp':fp,'fn':fn,'micro_precision':tp/(tp+fp) if tp+fp else 0,
            'micro_recall':tp/(tp+fn) if tp+fn else 0,'micro_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'macro_matched_per_requested':bootstrap_mean_ci([r['matched_per_requested'] for r in group],resamples=10000,seed=0),
            'macro_f1':bootstrap_mean_ci([r['f1'] for r in group],resamples=10000,seed=0)}
    pairs={}
    for a,b in (('A_trigger','A_suppress'),('AV_trigger','AV_suppress'),('V_memory','V_local')):
        keyed={name:{r['video_id']:r['f1'] for r in rows if r['condition']==name} for name in (a,b)}
        pairs[a+' minus '+b]=paired_bootstrap_diff([keyed[a][i] for i in ids],[keyed[b][i] for i in ids],resamples=10000,seed=0)
    report={'scope':'exploratory development-only diagnostics; untuned selectors; not confirmatory results',
            'reference':'US AD starts after merging cue gaps <=0.5s; retain all annotations; report bounds anomalies',
            'tolerance_seconds':2,'bootstrap':'10000 resamples, seed0; one clip per unique movie verified',
            'source_sha256':{str(path):digest(path) for path in ([args.fused_report,args.visual_report,args.split_config]+([args.baseline_report] if args.baseline_report else []))},
            'summaries':summaries,'exploratory_paired_differences':pairs,'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(report,handle,indent=2,allow_nan=False)
    with args.output.with_suffix('.csv').open('w') as handle:
        fields=['video_id','movie_id','condition','requested','selected','shortfall','tp','fp','fn','precision','recall','f1','matched_per_requested']
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__': main()
