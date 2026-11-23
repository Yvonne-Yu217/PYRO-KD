"""Compare all three audio-context conditions using bound core references."""
import argparse,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.export_core_temporal_metrics import export_condition,strict_rows,sha
from scripts.analyze_paired import analyze
from src.evaluation.stats import holm_adjust

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','experiment-dir','core-provenance','plan','output']:p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();config=a.config.resolve();exp=a.experiment_dir.resolve();core=a.core_provenance.resolve();planpath=a.plan.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError('preserve prior analysis')
    os.chdir(config.parents[2])
    from src.experiments.config import load_config
    from src.experiments.matrix import ablation5_runs
    from src.experiments.generation_stage import moments_path,generations_path
    cfg=load_config(config);provenance=json.loads(core.read_text());refs=provenance['references']
    if len(refs)!=30 or provenance['expected_clips']!=30:raise ValueError('complete30clip references required')
    for path,h in {**provenance['sources'],**provenance['reference_sources']}.items():
        if sha(path)!=h:raise ValueError('core bound source changed: '+path)
    plan=json.loads(planpath.read_text())
    if plan['contrasts']!=[['timeline','no_context'],['native','no_context'],['native','timeline']]:raise ValueError('exact3 contrasts required')
    rows=[];sources={str(config):sha(config),str(core):sha(core),str(planpath):sha(planpath)};moment_ids=None
    for run in ablation5_runs(cfg):
        condition='timeline' if run.timeline else 'native' if run.model_label=='omni' else 'no_context'
        mp=moments_path(exp,run.selection);gp=generations_path(exp,run);rp=exp/'eval'/f'generation_{run.run_id}.json'
        moments=strict_rows(mp);ids={m['id'] for m in moments}
        if moment_ids is not None and ids!=moment_ids:raise ValueError('paired conditions use different moments')
        moment_ids=ids
        checked=export_condition(json.loads(rp.read_text()),strict_rows(gp),moments,refs,run)
        for row in checked:row['condition']=condition
        rows+=checked
        for path in [mp,gp,rp]:sources[str(path)]=sha(path)
    if len(rows)!=90 or {r['condition'] for r in rows}!={'no_context','timeline','native'}:raise ValueError('all3conditions required')
    comparisons=[analyze(rows,x,y,plan['metric'],plan['unit'],plan['resamples'],plan['seed']) for x,y in plan['contrasts']]
    if any(r['excluded'] or len(r['units'])!=10 for r in comparisons):raise ValueError('all10movies required')
    for result,adjusted in zip(comparisons,holm_adjust([r['wilcoxon']['pvalue'] for r in comparisons])):result['wilcoxon']['holm_pvalue']=adjusted
    for path,h in sources.items():
        if sha(path)!=h:raise ValueError('source changed during analysis')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(plan=plan,comparisons=comparisons,paired_rows=rows,sources=sources,script_sha256=sha(__file__)),indent=2,allow_nan=False)+'\n')
    print(json.dumps({'conditions':3,'clips':30,'movies':10,'comparisons':3}))
if __name__=='__main__':main()
