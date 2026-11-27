"""Rebuild local tables/plots from completion-verified cached metrics, not inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
RESULTS = Path('/scratch/yy5919/pyro-resume-results')
DERIVED = Path('/scratch/yy5919/pyro-kd-derived')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completion_gate(path):
    """No scoring inputs may be read before this metadata/hash-only gate passes."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get('local_experiments_complete') is not True or manifest.get('errors'):
        raise ValueError('local experiment completion manifest is not ready')
    artifacts = manifest.get('artifact_sha256')
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError('completion artifact inventory required')
    for name, digest in artifacts.items():
        if sha(name) != digest:
            raise ValueError('completion artifact changed: '+name)
    return manifest


def require_bound(path, manifest):
    """Every direct cached data input must be covered by the completion inventory."""
    path = Path(path).resolve()
    bound = {str(Path(k).resolve()):v for k,v in manifest['artifact_sha256'].items()}
    if str(path) not in bound or sha(path) != bound[str(path)]:
        raise ValueError('input absent from verified completion inventory: '+str(path))
    return str(path)


def build_commands(args, manifest):
    config=require_bound(args.config,manifest)
    root=args.results_root
    ad=root/'experiments/reframed_test_local_v1'
    from src.experiments.config import load_config
    from src.experiments.matrix import core_generation_runs, ablation5_runs
    from src.experiments.generation_stage import generations_path, moments_path
    from src.experiments.selection_stage import selections_path
    cfg=load_config(config)
    if (Path(cfg['paths']['output_root'])/cfg['experiment']['name']).resolve()!=ad.resolve():
        raise ValueError('config and results root differ')
    for run in [r for r in core_generation_runs(cfg) if r.model_label=='qwen']+ablation5_runs(cfg):
        for path in [generations_path(ad,run),moments_path(ad,run.selection),selections_path(ad,run.selection),ad/'eval'/f'generation_{run.run_id}.json',args.tts_root/run.run_id/'overlap/report.json']:
            require_bound(path,manifest)
    commands=[]
    def add(script,*argv):
        commands.append([sys.executable,str(REPO/'scripts'/script),*map(str,argv)])
    for matrix,name in [('core','reframed-formal-core'),('ablation5','reframed-formal-ablation5')]:
        out=args.output_root/name
        argv=['--config',config,'--matrix',matrix,'--tts-root',args.tts_root,'--output-dir',out,'--scope','Frozen30clip test subset; local models; API and human validation deferred']
        if matrix=='core':argv += ['--deferred-model','api']
        add('export_core_results.py',*argv)
        add('plot_core_coverage.py','--input',out/(matrix+'-results.json'),'--output',out/'coverage','--title','REFRAMED '+matrix+' Qwen attempted and valid coverage')
        if matrix=='ablation5':
            add('plot_core_coverage.py','--input',out/'ablation5-results.json','--output',out/'coverage-omni','--model','omni','--title','REFRAMED native Omni attempted and valid coverage')
    longplan=REPO/'configs/long_qa_comparisons.json'
    require_bound(longplan,manifest)
    conditions=json.loads(longplan.read_text())['conditions']
    families=[('video-mme','long-qa-frozen',conditions,10),('egoschema','egoschema-heldout',['U','V_local','V_memory'],3)]
    for dataset,directory,names,count in families:
        if len(names)!=count:raise ValueError('QA condition inventory mismatch')
        argv=['--dataset',dataset,'--output-dir',args.output_root/directory,'--scope','Frozen local QA; fixed denominator; failures incorrect; no human validation']
        for name in names:
            report=root/directory/'evaluation'/(name+('.json' if dataset=='video-mme' else '.evaluation.json'))
            argv += ['--report',name,require_bound(report,manifest)]
            if dataset=='video-mme':
                inp=args.derived_root/'moments'/('video-mme-long-'+name+'-v1.jsonl')
                argv += ['--qa-input',name,require_bound(inp,manifest)]
        add('export_qa_results.py',*argv)
    contrasts=[('reframed-core-temporal',root/'formal-core-temporal/contrasts.json',17),
               ('reframed-ablation5-temporal',root/'formal-ablation5-temporal/contrasts.json',3),
               ('long-qa-paired',root/'long-qa-frozen/evaluation/comparisons.json',13),
               ('egoschema-paired',root/'egoschema-heldout/evaluation/comparisons.json',3)]
    for name,path,count in contrasts:
        require_bound(path,manifest)
        obj=json.loads(path.read_text());rows=obj.get('results',obj.get('comparisons'))
        if not isinstance(rows,list) or len(rows)!=count:raise ValueError(name+': incomplete contrast family')
        add('plot_paired_results.py','--input',path,'--output-dir',args.output_root/name,'--title',name,'--scope',f'All{count} frozen contrasts; saved paired statistics; Holm family correction')
    add('export_selector_results.py','--input',require_bound(ad/'eval/selection_metrics.json',manifest),'--output-dir',args.output_root/'reframed-selectors')
    return commands


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completion-manifest',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--results-root',type=Path,default=RESULTS)
    p.add_argument('--derived-root',type=Path,default=DERIVED)
    p.add_argument('--tts-root',type=Path,default=DERIVED/'tts/formal-local-v1')
    p.add_argument('--config',type=Path,default=Path('/scratch/yy5919/pyro-revision15-formal/configs/experiments/reframed_test_local.toml'))
    a=p.parse_args(argv)
    if a.output_root.exists():raise ValueError('output root must not already exist')
    manifest=completion_gate(a.completion_manifest)
    commands=build_commands(a,manifest)
    a.output_root.mkdir(parents=True,exist_ok=False)
    record=dict(scope='Cached local metric table/plot reproduction only; no inference, scoring models, API calls or human labels',
                all_proposal_complete=False,completion_manifest=str(a.completion_manifest.resolve()),completion_manifest_sha256=sha(a.completion_manifest),
                source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                script_sha256={cmd[1]:sha(cmd[1]) for cmd in commands},commands=commands,completed_commands=0,complete=False)
    provenance=a.output_root/'reproduction.json'
    try:
        for cmd in commands:
            provenance.write_text(json.dumps(record,indent=2)+'\n')
            subprocess.run(cmd,check=True,cwd=REPO)
            record['completed_commands']+=1
        # Detect a writer changing inputs while tables were generated.
        completion_gate(a.completion_manifest)
        record['complete']=True
        record['outputs_sha256']={str(p.relative_to(a.output_root)):sha(p) for p in a.output_root.rglob('*') if p.is_file() and p!=provenance}
    finally:
        provenance.write_text(json.dumps(record,indent=2)+'\n')
    return 0

if __name__=='__main__':
    sys.exit(main())
