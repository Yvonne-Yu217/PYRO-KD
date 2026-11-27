"""Checked movie-clustered inference for a fixed selector-ablation family."""
import argparse,csv,json,os,sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.export_core_temporal_metrics import sha,digest,strict_rows,same_number
from scripts.analyze_paired import analyze
from src.evaluation.metrics import temporal_precision_recall_f1
from src.evaluation.stats import holm_adjust


def checked_rows(data,selections,references):
    ids=set(references)
    if set(data['common_video_ids'])!=ids or len(data['common_video_ids'])!=len(ids) or data['n_videos']!=len(ids):
        raise ValueError('fixed complete membership required')
    rows=[]
    for name,run in data['runs'].items():
        videos=run['videos'];by_id={v['video_id']:v for v in videos}
        raw=selections[name];selected={v['video_id']:v for v in raw}
        if (not run['complete'] or run['missing_video_ids'] or run['unexpected_video_ids']
                or len(by_id)!=len(videos) or set(by_id)!=ids or len(selected)!=len(raw) or set(selected)!=ids):
            raise ValueError('all frozen clips, including zero picks, required')
        for ident in sorted(ids):
            v=by_id[ident];r=selected[ident];ref=references[ident]
            if r['run_id']!=run['run_id'] or r['selector']!=name or r['status']!='ok' or v['status']!='ok':
                raise ValueError('selection identity/status mismatch')
            if not ref['movie_id'] or v['group']!=ref['movie_id']:raise ValueError('movie cluster mismatch')
            if r['k']!=ref['requested_k'] or v['k']!=r['k'] or len(r['times'])!=v['num_selected']:
                raise ValueError('selection counts or fixedK mismatch')
            metrics=temporal_precision_recall_f1(r['times'],ref['starts'],data['tolerance_seconds'])
            for key in ('tp','fp','fn','precision','recall','f1'):
                if not same_number(v[key],metrics[key]):raise ValueError('report/reference temporal mismatch')
            if v['num_gt']!=len(ref['starts']) or v['shortfall']!=r['k']-len(r['times']):
                raise ValueError('reference count or shortfall mismatch')
            rows.append({'example_id':ident,'condition':name,'cluster_id':ref['movie_id'],
                         'temporal_f1':metrics['f1'],'temporal_precision':metrics['precision'],
                         'temporal_recall':metrics['recall'],'tp':metrics['tp'],'fp':metrics['fp'],'fn':metrics['fn'],
                         'requested_k':r['k'],'selected':len(r['times']),'reference_sha256':ref['reference_sha256']})
    return rows


def family_analysis(rows,plan):
    if plan['metric']!='temporal_f1' or plan['unit']!='cluster':raise ValueError('movie-clustered temporalF1 only')
    if len(plan['contrasts'])!=16 or len({(r['a'],r['b']) for r in plan['contrasts']})!=16:
        raise ValueError('exact declared16 contrasts required')
    results=[]
    for c in plan['contrasts']:
        if 'O' in (c['a'],c['b']):raise ValueError('no oracle inferential contrasts')
        r=analyze(rows,c['a'],c['b'],plan['metric'],plan['unit'],plan['resamples'],plan['seed'])
        if r['excluded'] or len(r['included_example_ids'])!=30 or len(r['units'])!=10:
            raise ValueError('all30clips/10movies required for every contrast')
        r['ablation']=c['ablation'];results.append(r)
    for r,p in zip(results,holm_adjust([r['wilcoxon']['pvalue'] for r in results])):
        r['wilcoxon']['holm_pvalue']=p;r['wilcoxon']['pvalue_adjustment']=plan['adjustment']
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--experiment-dir',required=True,type=Path)
    p.add_argument('--plan',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args();config=a.config.resolve();exp=a.experiment_dir.resolve();plan_path=a.plan.resolve();out=a.output_dir.resolve()
    if out.exists():raise FileExistsError('preserve prior analysis')
    os.chdir(config.parents[2])
    from src.experiments.config import load_config
    from src.experiments.data import load_records,_subset_ids,ground_truth,video_duration,k_for
    from src.experiments.matrix import all_selection_runs,NON_RESULT_MODEL_KEYS
    from src.utils.media import load_manifest
    cfg=load_config(config);ds=cfg['dataset']
    if ds['name']!='reframed' or ds['split']!='test' or ds['subset']!='test_sample':raise ValueError('frozen REFRAMEDtest required')
    guard=exp/'config.identity.json';identity=json.loads(guard.read_text())
    expected=json.loads(json.dumps({k:v for k,v in cfg.items() if k not in ('_source','evaluation')}))
    for model in expected.get('models',{}).values():
        for key in NON_RESULT_MODEL_KEYS:model.pop(key,None)
    if expected!={k:v for k,v in identity.items() if k!='source_and_input_hashes'}:raise ValueError('frozen config mismatch')
    for path,h in identity['source_and_input_hashes'].items():
        if sha(path)!=h:raise ValueError('frozen source changed: '+path)
    ids=_subset_ids(ds,ds['subset']);records=load_records(cfg)
    if len(ids)!=30 or len(set(ids))!=30 or {r.video_id for r in records}!=set(ids):raise ValueError('all30 frozen clips required')
    report_path=exp/'eval/selection_metrics.json';data=json.loads(report_path.read_text())
    runs=all_selection_runs(cfg)
    if len(runs)!=31 or {r.name:r.run_id for r in runs}!={n:r['run_id'] for n,r in data['runs'].items()}:
        raise ValueError('all31 frozen selector identities required')
    if data['tolerance_seconds']!=cfg['selection']['tolerance_seconds']:raise ValueError('tolerance mismatch')
    sources={str(p):sha(p) for p in [config,guard,report_path,plan_path,Path(ds['root'])/'test.jsonl']}
    refs={}
    for record in records:
        cache=Path(cfg['paths']['cache_root'])/record.video_id;manifest=load_manifest(cache)
        sources[str(cache/'manifest.json')]=sha(cache/'manifest.json')
        duration=video_duration(record,manifest)
        starts,excluded=ground_truth(record,duration,float(cfg['selection']['duration_slack_seconds']))
        ref={'movie_id':record.group,'ad_version':record.meta.get('ad_version'),'merged_ad':[asdict(e) for e in record.ad],
             'duration':duration,'starts':starts,'out_of_bounds':excluded,'requested_k':k_for(record,duration,cfg)}
        refs[record.video_id]={**ref,'reference_sha256':digest(ref)}
    selections={}
    for run in runs:
        path=exp/'selections'/f'{run.run_id}.jsonl';sources[str(path)]=sha(path)
        # Selection runner and evaluator use latest-per-video semantics on resumable files.
        selections[run.name]=list({r['video_id']:r for r in strict_rows(path)}.values())
    rows=checked_rows(data,selections,refs);plan=json.loads(plan_path.read_text());results=family_analysis(rows,plan)
    for path,h in sources.items():
        if sha(path)!=h:raise ValueError('source changed during analysis')
    out.mkdir(parents=True)
    provenance={'sources':sources,'references':{vid:{'movie_id':ref['movie_id'],'reference_sha256':ref['reference_sha256'],'requested_k':ref['requested_k'],'num_ground_truth':len(ref['starts'])} for vid,ref in refs.items()},'script_sha256':sha(__file__),
                'implementation':{str(Path(m.__file__).resolve()):sha(m.__file__) for n,m in sys.modules.items()
                                  if n in ('scripts.analyze_paired','src.evaluation.stats','src.evaluation.metrics','src.datasets.adapters','src.datasets.reframed')},
                'reference_limitation':'historical report lacks original reference digest; every metric was recomputed and matched against currently bound references'}
    payload={'plan':plan,'provenance':provenance,'results':results,'coverage':{'selectors':31,'clips':30,'movies':10},
             'scope':'selector placement only; no generated-text quality, rendered-speech overlap, or human detector accuracy inference'}
    (out/'selector-ablation-stats.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    with (out/'paired-metrics.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    summaries=[{'ablation':r['ablation'],'a':r['condition_a'],'b':r['condition_b'],'movies':len(r['units']),
                **r['bootstrap'],'wilcoxon_p':r['wilcoxon']['pvalue'],'holm_p':r['wilcoxon']['holm_pvalue']} for r in results]
    with (out/'selector-ablation-stats.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    notes=plan['declaration']+'. All30clips,10movies; movie-mean differences; 95% paired bootstrap CIs; two-sided Wilcoxon; Holm across16. Selector placement only, not human detector accuracy. RequestedK may exceed realizedK.'
    lines=['# Selector ablation statistics','',notes,'','| Ablation | A − B | Mean F1 difference | 95% CI | Holm p |','|---|---|---:|---|---:|']
    for r in summaries:lines.append(f"| {r['ablation']} | {r['a']} − {r['b']} | {r['mean_diff']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['holm_p']:.4f} |")
    (out/'selector-ablation-stats.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,8));means=[r['mean_diff'] for r in summaries]
    ax.errorbar(means,range(len(means)),xerr=[[max(0.,r['mean_diff']-r['ci_low']) for r in summaries],[max(0.,r['ci_high']-r['mean_diff']) for r in summaries]],fmt='o',capsize=3,color='#176b91')
    ax.set_yticks(range(len(means)),[r['a']+' − '+r['b'] for r in summaries]);ax.invert_yaxis();ax.axvline(0,color='gray',linewidth=1)
    ax.set_xlabel('Movie-mean temporal F1 difference, 95% paired bootstrap CI');ax.set_title('REFRAMED selector ablations: 30 clips / 10 movies');fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(out/f'selector-ablation-stats.{suffix}',dpi=180)
    print(json.dumps({'complete':True,'selectors':31,'contrasts':len(results),'clips':30,'movies':10,'output':str(out)}))

if __name__=='__main__':main()
