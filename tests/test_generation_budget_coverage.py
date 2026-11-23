"""Budget failures count in compliance, without leaking into quality metrics."""
import csv
import json
from types import SimpleNamespace
from unittest.mock import patch

from src.experiments.generation_stage import (
    budget_compliance, evaluate_generation, generations_path, moments_path,
)
from src.experiments.matrix import core_generation_runs
from src.experiments.report import generation_table


def test_budget_known_flags_exclude_truncated_and_incomplete_decodes():
    rows = [
        {'output':'A tree.', 'over_budget':False},
        {'output':'Too many words.', 'over_budget':True, 'error':'word_budget: generated output exceeds the declared limit'},
        {'output':'Partial', 'over_budget':False, 'hit_token_limit':True},
        {'output':'Partial', 'over_budget':True, 'error':'hit_token_limit: max_new_tokens=128'},
        {'output':'', 'over_budget':False, 'error':'generate: failed'},
        {'output':'Legacy without a flag.'},
        {'output':'Bad flag.', 'over_budget':0},
    ]
    result = budget_compliance(rows, missing_count=2)
    assert result == {'n_known':2,'n_over_budget':1,'n_unknown_generated':5,
                      'n_missing':2,'known_fraction':2/9,'rate':.5}
    assert budget_compliance([],3)['rate'] is None
    assert budget_compliance([],3)['known_fraction'] == 0


def test_flagged_failure_counts_but_f1_and_text_remain_valid_only(tmp_path):
    cfg={'matrix':{'selectors':['U'],'priors':['P0'],'models':['m']},
         'models':{'m':{'backend':'echo'}},'selection':{'tolerance_seconds':.1},
         'evaluation':{'resamples':10,'text_metrics':[]}}
    run=core_generation_runs(cfg)[0]
    records=[SimpleNamespace(video_id='v',ad=[SimpleNamespace(start=1.,text='Tree reference'),
                                            SimpleNamespace(start=2.,text='Other reference')]),
             SimpleNamespace(video_id='missing_video',ad=[])]
    moments=[{'id':f'm{i}','video_id':'v'} for i in range(6)] + [{'id':'absent','video_id':'missing_video'}]
    path=moments_path(tmp_path,run.selection);path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in moments))
    base={'video_id':'v','time':1.,'word_limit':3}
    rows=[
        {**base,'id':'m0','output':'Old error', 'error':'word_budget: prior attempt','over_budget':True},
        {**base,'id':'m0','output':'A tree.','over_budget':False},
        {**base,'id':'m1','time':2.,'output':'A description exceeding the budget.',
         'over_budget':True,'error':'word_budget: generated output exceeds the declared limit'},
        {**base,'id':'m2','output':'Cut off', 'hit_token_limit':True},
        {**base,'id':'m3','output':'', 'error':'generate: failed'},
        {**base,'id':'m4','output':'', 'error':'legacy missing budget flag'},
    ]
    output=generations_path(tmp_path,run);output.parent.mkdir(parents=True)
    output.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    seen=[]
    def score(generated,references):
        seen.extend(zip(generated,references));return [1.] * len(generated)
    context=SimpleNamespace(duration=10.,audio_features={'speech':[]})
    with patch('src.experiments.generation_stage.load_context',return_value=(context,{})), \
         patch('src.experiments.generation_stage.ground_truth',side_effect=lambda r,*a:([x.start for x in r.ad],{})):
        report=evaluate_generation(cfg,run,records,tmp_path,scorers={'test':(score,None)})
    first,missing=report['videos']
    assert first['n_ok']==1 and first['n_error']==4
    assert first['precision']==1. and first['recall']==.5
    assert seen==[('A tree.','Tree reference')]
    assert first['over_budget_rate']==.5
    assert first['budget_compliance']['n_unknown_generated']==3
    assert first['missing_ids']==['m5']
    assert missing['over_budget_rate'] is None
    assert missing['budget_compliance']['n_missing']==1
    assert report['over_budget_rate']['mean']==.5
    assert report['budget_compliance']['n_known']==2
    assert report['budget_compliance']['n_over_budget']==1
    assert report['budget_compliance']['n_missing']==2
    assert report['budget_compliance']['known_fraction']==2/7
    # The CSV consumer supports both corrected and historical evaluator records.
    generation_table([report],tmp_path/'new')
    row=next(csv.DictReader((tmp_path/'new/generation_matrix.csv').open()))
    assert row['budget_known']=='2' and row['budget_missing']=='2'
    historical={k:v for k,v in report.items() if k!='budget_compliance'}
    generation_table([historical],tmp_path/'old')
    row=next(csv.DictReader((tmp_path/'old/generation_matrix.csv').open()))
    assert row['budget_known']==''
