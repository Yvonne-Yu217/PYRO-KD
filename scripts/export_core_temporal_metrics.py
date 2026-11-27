"""Export all frozen core AD temporal metrics with raw-output/reference checks.

Unlike matched-pair text scores, temporal metrics retain every selected clip,
including clips with no feasible moments or no valid generations. Does not run
statistical inference or alter generation/evaluation artifacts.
"""
import argparse,hashlib,json,math,os,sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.metrics import temporal_precision_recall_f1
from src.generation.outputs import final_rows,is_success


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def strict_rows(path):
    raw=Path(path).read_bytes()
    if raw and not raw.endswith(b'\n'):raise ValueError('incomplete JSONL tail')
    return [json.loads(line) for line in raw.splitlines() if line.strip()]

def same_number(actual,expected):
    if expected is None:return actual is None
    return (not isinstance(actual,bool) and isinstance(actual,(int,float)) and math.isfinite(actual)
            and math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12))

def export_condition(report,generations,moments,references,run):
    """Pure checked export. references maps every frozen video to movie/starts/hash."""
    condition=f'{run.selection.name}-{run.prior}'
    for key,value in {'run_id':run.run_id,'selector':run.selection.name,'prior':run.prior,
                      'model':run.model_label,'timeline':run.timeline,'oracle':run.selection.is_oracle}.items():
        if report.get(key)!=value:raise ValueError(f'{condition}: report identity mismatch {key}')
    ids=set(references)
    videos=report['videos'];by_video={r['video_id']:r for r in videos}
    if len(by_video)!=len(videos) or set(by_video)!=ids or report['n_videos']!=len(ids) or report['expected_videos']!=len(ids):
        raise ValueError('complete unique frozen video membership required')
    if report.get('unexpected_video_ids') or report.get('missing_moment_ids'):
        raise ValueError('incomplete/unexpected generation coverage')
    moment_map={m['id']:m for m in moments}
    if len(moment_map)!=len(moments) or any(m['video_id'] not in ids for m in moments):
        raise ValueError('invalid moment membership')
    if report['expected_moments']!=len(moments):raise ValueError('moment count mismatch')
    final=final_rows(generations);pred={r['id']:r for r in final}
    if len(pred)!=len(final) or set(pred)!=set(moment_map):raise ValueError('incomplete moment attempts')
    for ident,row in pred.items():
        m=moment_map[ident]
        if m.get('k_requested')!=references[m['video_id']]['requested_k']:
            raise ValueError('moment requestedK differs from frozen protocol')
        if row.get('run_id')!=run.run_id or row.get('video_id')!=m['video_id'] or not same_number(row.get('time'),m['time']):
            raise ValueError('prediction identity/time differs from frozen moment')
    missing=sorted(ids-{r['video_id'] for r in final})
    if sorted(report['missing_video_ids'])!=missing:raise ValueError('zero-output coverage disagrees')
    out=[]
    for ident in sorted(ids):
        ref=references[ident];v=by_video[ident]
        if not ref['movie_id'] or not ref['reference_sha256']:raise ValueError('movie/reference provenance required')
        rows=[r for r in final if r['video_id']==ident]
        if len(rows)>ref['requested_k']:raise ValueError('prepared moments exceed fixedK')
        ok=[r for r in rows if is_success(r) and r.get('output')]
        metrics=temporal_precision_recall_f1([r['time'] for r in ok],ref['starts'],ref['tolerance_seconds'])
        for key in ('precision','recall','f1'):
            if not same_number(v[key],metrics[key]):raise ValueError('report differs from recomputed temporal metrics')
        if (v['n_expected']!=len(rows) or v['n_moments']!=len(rows) or v['n_ok']!=len(ok)
                or v['n_error']!=len(rows)-len(ok) or v['missing_ids'] or v['unexpected_ids']
                or v['no_generation_rows']!= (not rows)):
            raise ValueError('per-video generation coverage disagrees')
        out.append({'example_id':ident,'cluster_id':ref['movie_id'],'condition':condition,
                    'oracle_upper_bound':run.selection.is_oracle,'reference_sha256':ref['reference_sha256'],
                    **{'temporal_'+k:metrics[k] for k in ('precision','recall','f1','tp','fp','fn')},
                    'num_generated':len(ok),'num_generation_errors':len(rows)-len(ok),
                    'num_requested_moments':ref['requested_k'],'num_prepared_moments':len(rows),
                    'num_preparation_shortfall':ref['requested_k']-len(rows),'num_ground_truth':len(ref['starts'])})
    expected_macro=[r['temporal_f1'] for r in out if r['temporal_f1'] is not None]
    if not same_number(report['macro_f1']['mean'],sum(expected_macro)/len(expected_macro) if expected_macro else None):
        raise ValueError('macro summary differs from all-video metrics')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--experiment-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve();sidecar=out.with_suffix(out.suffix+'.provenance.json')
    if out.exists() or sidecar.exists():raise FileExistsError('preserve prior export')
    config=args.config.resolve();exp=args.experiment_dir.resolve()
    # Resolve config-relative repository paths as the original frozen runner did.
    frozen_repo=config.parents[2];os.chdir(frozen_repo)
    from src.experiments.config import load_config
    from src.experiments.data import load_records,ground_truth,video_duration,_subset_ids,k_for
    from src.experiments.matrix import core_generation_runs
    from src.experiments.generation_stage import moments_path,generations_path
    from src.utils.media import load_manifest
    cfg=load_config(config)
    if cfg['dataset']['name']!='reframed' or cfg['dataset']['split']!='test' or cfg['dataset']['subset']!='test_sample':
        raise ValueError('frozen REFRAMED test_sample required')
    identity_path=exp/'config.identity.json';identity=json.loads(identity_path.read_text())
    from src.experiments.matrix import NON_RESULT_MODEL_KEYS
    expected_identity=json.loads(json.dumps({k:v for k,v in cfg.items() if k not in ('_source','evaluation')}))
    for model in expected_identity.get('models',{}).values():
        for key in NON_RESULT_MODEL_KEYS:model.pop(key,None)
    if expected_identity!={k:v for k,v in identity.items() if k!='source_and_input_hashes'}:
        raise ValueError('experiment configuration identity mismatch')
    for source,expected in identity['source_and_input_hashes'].items():
        if sha(source)!=expected:raise ValueError('frozen source/input changed: '+source)
    records=load_records(cfg);ids=_subset_ids(cfg['dataset'],cfg['dataset']['subset'])
    if len(ids)!=30 or len(set(ids))!=30 or {r.video_id for r in records}!=set(ids):
        raise ValueError('all30 frozen test clips required')
    annotations=Path(cfg['dataset']['root'])/'test.jsonl'
    refs={};reference_sources={}
    for record in records:
        manifest_path=Path(cfg['paths']['cache_root'])/record.video_id/'manifest.json'
        manifest=load_manifest(manifest_path.parent)
        duration=video_duration(record,manifest)
        if duration is None or record.ad is None:raise ValueError('complete temporal references required')
        starts,excluded=ground_truth(record,duration,float(cfg['selection']['duration_slack_seconds']))
        reference={'video_id':record.video_id,'movie_id':record.group,'ad_version':record.meta.get('ad_version'),
                   'merged_ad':[asdict(e) for e in record.ad],'duration':duration,'starts':starts,
                   'out_of_bounds_starts':excluded,'requested_k':k_for(record,duration,cfg),'tolerance_seconds':float(cfg['selection']['tolerance_seconds'])}
        refs[record.video_id]={**reference,'reference_sha256':digest(reference)}
        reference_sources[str(manifest_path.resolve())]=sha(manifest_path)
    runs=[r for r in core_generation_runs(cfg) if r.model_label=='qwen']
    expected={(s,p) for s in ('U','V','A_cue','AV','O') for p in ('P0','P1')}
    if {(r.selection.name,r.prior) for r in runs}!=expected or len(runs)!=10 or not all(r.timeline for r in runs):
        raise ValueError('exact ten Qwen core timeline cells required')
    rows=[];sources={str(config):sha(config),str(identity_path):sha(identity_path),str(annotations.resolve()):sha(annotations)}
    for run in runs:
        report_path=exp/'eval'/f'generation_{run.run_id}.json'
        generation_path=generations_path(exp,run);moment_path=moments_path(exp,run.selection)
        for path in (report_path,generation_path,moment_path):sources[str(path.resolve())]=sha(path)
        rows+=export_condition(json.loads(report_path.read_text()),strict_rows(generation_path),strict_rows(moment_path),refs,run)
    # A completed source changing during export must not produce mixed-condition provenance.
    for source,expected_hash in {**sources,**reference_sources}.items():
        if sha(source)!=expected_hash:raise ValueError('input changed during export: '+source)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    implementation={str(Path(__file__).resolve()):sha(__file__)}
    for name in ('src.evaluation.metrics','src.generation.outputs','src.experiments.data','src.datasets.adapters','src.datasets.reframed'):
        module=sys.modules[name];implementation[str(Path(module.__file__).resolve())]=sha(module.__file__)
    provenance={'implementation':implementation,'sources':sources,'reference_sources':reference_sources,'references':refs,'output_sha256':sha(out),
                'exporter_sha256':sha(__file__),'analysis_unit':'movie','expected_clips':30,'conditions':10,
                'scope':'temporal metrics only; valid outputs define predictions; all frozen clips retained',
                'reference_check':'recomputed every per-video metric from bound annotation adapter and raw outputs; historical reports have no original reference digest',
                'source_validation':'original source/input identity checked without modifying experiment guard'}
    with sidecar.open('x') as f:json.dump(provenance,f,indent=2,allow_nan=False)
    print(json.dumps({'rows':len(rows),'clips':30,'movies':len({r['cluster_id'] for r in rows}),'conditions':10}))

if __name__=='__main__':main()
