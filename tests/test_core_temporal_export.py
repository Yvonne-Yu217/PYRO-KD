import copy,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.export_core_temporal_metrics import export_condition,strict_rows
from scripts.analyze_core_temporal import run_family


def fixture():
    run=SimpleNamespace(selection=SimpleNamespace(name='U',is_oracle=False),prior='P0',model_label='qwen',timeline=True,run_id='run')
    refs={v:{'movie_id':'movie','reference_sha256':'hash-'+v,'starts':[1.],'requested_k':1,'tolerance_seconds':2.} for v in ['a','b']}
    moments=[{'id':'a1','video_id':'a','time':1.,'k_requested':1}]
    generations=[{'id':'a1','video_id':'a','run_id':'run','time':1.,'output':'A person walks.','word_limit':5}]
    videos=[]
    for v,n in [('a',1),('b',0)]:videos.append({'video_id':v,'n_moments':n,'n_expected':n,'n_ok':n,'n_error':0,'precision':float(n),'recall':float(n),'f1':float(n),'missing_ids':[],'unexpected_ids':[],'no_generation_rows':not n})
    report={'run_id':'run','selector':'U','prior':'P0','model':'qwen','timeline':True,'oracle':False,'n_videos':2,'expected_videos':2,'expected_moments':1,'missing_video_ids':['b'],'unexpected_video_ids':[],'missing_moment_ids':[],'macro_f1':{'mean':.5},'videos':videos}
    return report,generations,moments,refs,run


def test_zero_output_clip_is_retained_in_same_movie():
    rows=export_condition(*fixture())
    assert len(rows)==2 and rows[1]['temporal_f1']==0 and rows[1]['num_generated']==0
    assert rows[1]['num_preparation_shortfall']==1
    assert rows[0]['cluster_id']==rows[1]['cluster_id']=='movie'
    assert all('bertscore' not in r for r in rows)


def test_recomputation_rejects_changed_references_or_report():
    args=fixture();args[3]['a']['starts']=[100.]
    with pytest.raises(ValueError,match='recomputed'):export_condition(*args)
    args=fixture();args[0]['videos'][0]['f1']=.2
    with pytest.raises(ValueError,match='recomputed'):export_condition(*args)


def test_incomplete_or_wrong_run_attempts_refused():
    args=fixture();args[1].clear()
    with pytest.raises(ValueError,match='attempts'):export_condition(*args)
    args=fixture();args[1][0]['run_id']='different'
    with pytest.raises(ValueError,match='identity'):export_condition(*args)


def test_superseded_failure_uses_same_final_row_policy():
    args=fixture();failed={**args[1][0],'error':'backend failed','output':''};args[1].insert(0,failed)
    assert export_condition(*args)[0]['num_generation_errors']==0


def test_truncated_tail_refused_for_final_statistics(tmp_path):
    p=tmp_path/'x.jsonl';p.write_text('{"id":"a"}\n{"id":')
    with pytest.raises(ValueError,match='tail'):strict_rows(p)


def test_declared_family_has_all_pairs_and_does_not_use_clip_units():
    plan=json.loads((Path(__file__).resolve().parents[1]/'configs/core_temporal_contrasts.json').read_text())
    assert len({(r['a'],r['b']) for r in plan['contrasts']})==17
    assert sum(r['oracle_upper_bound'] for r in plan['contrasts'])==1
    plan['resamples']=20
    rows=[{'example_id':str(i),'cluster_id':str(i//3),'condition':f'{s}-{p}','temporal_f1':0.0} for s in ('U','V','A_cue','AV','O') for p in ('P0','P1') for i in range(30)]
    result=run_family(rows,plan)
    assert len(result['results'])==17
    assert all(len(r['units'])==10 and r['wilcoxon']['holm_pvalue']==1 for r in result['results'])
    with pytest.raises(ValueError,match='complete30'):run_family(rows[:-1],plan)
