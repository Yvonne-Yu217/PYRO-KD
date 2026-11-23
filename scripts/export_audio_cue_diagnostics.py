"""Descriptive detector-defined audio proxies on frozen REFRAMED references.

Reuses existing diagnostics defaults; never estimates human detector accuracy.
"""
import argparse,csv,json,os,sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.audio_diagnostics import trigger_suppress,silent_gt_times,silent_visual_coverage
from src.evaluation.metrics import temporal_precision_recall_f1
from src.audio.taxonomy import SUPERCLASSES
from scripts.export_core_temporal_metrics import sha,digest,strict_rows,same_number
from scripts.analyze_selector_ablations import checked_rows

WINDOW=2.0
LOW_CHANGE_THRESHOLD=0.2
CORE=('U','V','A_cue','AV','O')

def summarize(videos):
    """Pure descriptive aggregation, with fixed core selector and proxy definitions."""
    if len({v['video_id'] for v in videos})!=len(videos):raise ValueError('duplicate video')
    association=trigger_suppress(videos,window=WINDOW,classes=SUPERCLASSES)
    labels=sorted({e['label'] for v in videos for e in v['events']})
    label_videos=[{**v,'events':[{**e,'superclass':e['label']} for e in v['events']]} for v in videos]
    by_label=trigger_suppress(label_videos,window=WINDOW,classes=labels)
    per_video=[];totals={name:{'tp':0,'fn':0} for name in CORE}
    for v in videos:
        if set(v['selections'])!=set(CORE):raise ValueError('all five core selectors required')
        low=silent_gt_times(v['gt_times'],v['grid'],v['audio_change'],threshold=LOW_CHANGE_THRESHOLD,window=WINDOW)
        coverage=silent_visual_coverage(v['selections'],low,v['gt_times'],tolerance=WINDOW)
        row={'video_id':v['video_id'],'movie_id':v['movie_id'],'n_gt':len(v['gt_times']),
             'n_low_change_gt':len(low),'low_change_fraction':coverage['silent_fraction'],
             'recall_on_low_change':coverage['recall_on_silent']}
        per_video.append(row)
        for name,times in v['selections'].items():
            metrics=temporal_precision_recall_f1(times,low,WINDOW)
            totals[name]['tp']+=metrics['tp'];totals[name]['fn']+=metrics['fn']
    total_gt=sum(r['n_gt'] for r in per_video);total_low=sum(r['n_low_change_gt'] for r in per_video)
    for name,counts in totals.items():
        recalls=[v['recall_on_low_change'][name] for v in per_video if v['recall_on_low_change'][name] is not None]
        counts.update(pooled_recall=counts['tp']/total_low if total_low else None,
                      clip_macro_recall=sum(recalls)/len(recalls) if recalls else None,
                      clips_with_low_change_gt=len(recalls),oracle_upper_bound=name=='O')
    return {'association_by_superclass':association,'association_by_detected_label':by_label,
            'low_change_reference_coverage':{'n_gt':total_gt,'n_low_change_gt':total_low,
              'fraction':total_low/total_gt if total_gt else None,'selectors':totals,'videos':per_video}}


def association_rows(result,videos,key):
    output=[]
    for label,values in result['classes'].items():
        counts=[sum(e[key]==label for e in v['events']) for v in videos]
        output.append({'class':label,'detected_onsets':sum(counts),'clips_with_detected_onsets':sum(c>0 for c in counts),
                       **{k:values[k] for k in ('n_gt','hit_rate','chance_rate','lift')}})
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--experiment-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args();config=a.config.resolve();exp=a.experiment_dir.resolve();out=a.output_dir.resolve()
    if out.exists():raise FileExistsError('preserve prior diagnostics')
    os.chdir(config.parents[2])
    from src.experiments.config import load_config
    from src.experiments.data import load_records,_subset_ids,load_context,ground_truth,k_for
    from src.experiments.matrix import core_selections,NON_RESULT_MODEL_KEYS
    cfg=load_config(config);ds=cfg['dataset'];guard=exp/'config.identity.json'
    if ds['name']!='reframed' or ds['split']!='test' or ds['subset']!='test_sample':raise ValueError('frozen test subset required')
    identity=json.loads(guard.read_text());expected=json.loads(json.dumps({k:v for k,v in cfg.items() if k not in ('_source','evaluation')}))
    for model in expected.get('models',{}).values():
        for key in NON_RESULT_MODEL_KEYS:model.pop(key,None)
    if expected!={k:v for k,v in identity.items() if k!='source_and_input_hashes'}:raise ValueError('frozen configuration mismatch')
    for path,h in identity['source_and_input_hashes'].items():
        if sha(path)!=h:raise ValueError('frozen source/input changed')
    records=load_records(cfg);ids=_subset_ids(ds,ds['subset'])
    if len(ids)!=30 or len(set(ids))!=30 or {r.video_id for r in records}!=set(ids):raise ValueError('all30clips required')
    report_path=exp/'eval/selection_metrics.json';data=json.loads(report_path.read_text())
    if data['tolerance_seconds']!=WINDOW:raise ValueError('unchanged2s selection tolerance required')
    runs=core_selections(cfg)
    if {r.name for r in runs}!=set(CORE):raise ValueError('five frozen core selectors required')
    raw={};selected={};sources={str(p):sha(p) for p in (config,guard,report_path,Path(ds['root'])/'test.jsonl')}
    for run in runs:
        path=exp/'selections'/f'{run.run_id}.jsonl';sources[str(path)]=sha(path)
        raw[run.name]=list({v['video_id']:v for v in strict_rows(path)}.values())
        selected[run.name]={r['video_id']:r['times'] for r in raw[run.name]}
        if data['runs'][run.name]['run_id']!=run.run_id:raise ValueError('core selector identity mismatch')
    videos=[];refs={};cache_status={}
    for record in records:
        context,available=load_context(record,cfg,need_visual=False)
        if context.audio_cues is None or context.audio_features is None or available.get('audio_cues')!='ok':
            raise ValueError('validated complete audio cues required')
        if any(not str(context.audio_cues.get('streams',{}).get(s,'')).startswith('ok') for s in ('asr','tagger','clap','acoustic')):
            raise ValueError('required detector streams unavailable')
        bundle=context.bundle(use_visual=False)
        truth,excluded=ground_truth(record,context.duration,cfg['selection']['duration_slack_seconds'])
        reference={'movie_id':record.group,'ad_version':record.meta.get('ad_version'),'merged_ad':[asdict(e) for e in record.ad],
                   'duration':context.duration,'starts':truth,'excluded':excluded,'requested_k':k_for(record,context.duration,cfg)}
        refs[record.video_id]={**reference,'reference_sha256':digest(reference)}
        video={'video_id':record.video_id,'movie_id':record.group,'duration':context.duration,'gt_times':truth,
               'events':bundle.events,'grid':bundle.grid,'audio_change':bundle.audio_change(),
               'selections':{name:selected[name][record.video_id] for name in CORE}}
        videos.append(video);cache_status[record.video_id]={'available':available,'missing_streams':bundle.missing}
        cache=Path(cfg['paths']['cache_root'])/record.video_id
        for name in ('manifest.json','audio_features.json','audio_cues.json'):
            path=cache/name;sources[str(path)]=sha(path)
        sources.update(context.audio_cues.get('bridge_sources',{}))
    checked_rows({**data,'runs':{name:data['runs'][name] for name in CORE}},raw,refs)
    result=summarize(videos)
    # Confirm the newly exported low-change proxy exactly agrees with the prior selector evaluation.
    for row in result['low_change_reference_coverage']['videos']:
        for name in CORE:
            prior=next(v for v in data['runs'][name]['videos'] if v['video_id']==row['video_id'])
            if prior['n_silent_gt']!=row['n_low_change_gt'] or not same_number(prior['recall_on_silent_gt'],row['recall_on_low_change'][name]):
                raise ValueError('low-change diagnostic differs from existing frozen report')
    for path,h in sources.items():
        if sha(path)!=h:raise ValueError('source changed during diagnostics')
    result['parameters']={'onset_window_seconds':WINDOW,'low_change_window_seconds':WINDOW,
                          'low_change_threshold':LOW_CHANGE_THRESHOLD,'chance_grid_step_seconds':0.1,
                          'low_change_definition':'maximum over modeled NSE tracks below threshold; not actual acoustic silence'}
    result['coverage']={'clips':30,'movies':len({v['movie_id'] for v in videos}),'core_selectors':5}
    result['scope']='Detector-defined descriptive associations only; not causation, actual silence, human detector precision/recall, or rendered speech.'
    result['provenance']={'sources':sources,'cache_status':cache_status,
                         'reference_hashes':{v:r['reference_sha256'] for v,r in refs.items()},
                         'implementation':{str(Path(m.__file__).resolve()):sha(m.__file__) for n,m in sys.modules.items()
                                           if n in ('src.evaluation.audio_diagnostics','src.selection.cues','src.experiments.data','src.datasets.adapters','src.datasets.reframed')},
                         'exporter_sha256':sha(__file__)}
    superclass_rows=association_rows(result['association_by_superclass'],videos,'superclass')
    label_rows=association_rows(result['association_by_detected_label'],videos,'label')
    out.mkdir(parents=True)
    (out/'audio-cue-diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,rows in [('superclass-association',superclass_rows),('detected-label-association',label_rows)]:
        with (out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=['class','detected_onsets','clips_with_detected_onsets','n_gt','hit_rate','chance_rate','lift']);w.writeheader();w.writerows(rows)
    low=result['low_change_reference_coverage'];fmt=lambda x:'NA' if x is None else f'{x:.4f}'
    lines=['# Detector-defined audio-cue diagnostics','',result['scope'],'',
           'Frozen parameters: onset window ±2 s; low modeled change <0.2 within ±2 s; timeline chance sampled every0.1 s. No threshold tuning. All30clips retained.','',
           '| Superclass | Detected onsets | Clips with onset | AD-onset hit rate | Timeline chance | Lift |','|---|---:|---:|---:|---:|---:|']
    for r in superclass_rows:lines.append(f"| {r['class']} | {r['detected_onsets']} | {r['clips_with_detected_onsets']} | {fmt(r['hit_rate'])} | {fmt(r['chance_rate'])} | {fmt(r['lift'])} |")
    lines += ['',f"Low modeled audio-change subset: {low['n_low_change_gt']}/{low['n_gt']} reference starts ({fmt(low['fraction'])}); this is not a count of genuinely silent events.",'',
              '| Selector | Matched low-change references | Pooled recall | Clip-macro recall | Eligible clips |','|---|---:|---:|---:|---:|']
    for name,r in low['selectors'].items():lines.append(f"| {name} | {r['tp']} | {fmt(r['pooled_recall'])} | {fmt(r['clip_macro_recall'])} | {r['clips_with_low_change_gt']} |")
    lines += ['', 'O is an oracle upper bound. Pooled association hit rates weight reference starts; chance rates weight clip duration. Lift is descriptive and does not establish trigger/suppress causation. Human event annotations remain unavailable.']
    (out/'audio-cue-diagnostics.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete':True,**result['coverage'],'detected_labels':len(label_rows),'low_change_references':low['n_low_change_gt'],'reference_starts':low['n_gt']}))

if __name__=='__main__':main()
