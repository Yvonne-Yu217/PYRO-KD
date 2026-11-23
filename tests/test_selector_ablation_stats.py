import copy,json
from pathlib import Path
import pytest
from scripts.analyze_selector_ablations import checked_rows,family_analysis


def fixture():
    refs={v:{'movie_id':'movie','starts':[1.],'requested_k':1,'reference_sha256':'hash'} for v in ('a','b')}
    raw=[{'video_id':'a','run_id':'run','selector':'A','status':'ok','k':1,'times':[1.]},
         {'video_id':'b','run_id':'run','selector':'A','status':'ok','k':1,'times':[]}]
    videos=[{'video_id':'a','status':'ok','group':'movie','k':1,'num_selected':1,'tp':1,'fp':0,'fn':0,'precision':1.,'recall':1.,'f1':1.,'num_gt':1,'shortfall':0},
            {'video_id':'b','status':'ok','group':'movie','k':1,'num_selected':0,'tp':0,'fp':0,'fn':1,'precision':0.,'recall':0.,'f1':0.,'num_gt':1,'shortfall':1}]
    data={'tolerance_seconds':2.,'common_video_ids':['a','b'],'n_videos':2,
          'runs':{'A':{'complete':True,'missing_video_ids':[],'unexpected_video_ids':[],'run_id':'run','videos':videos}}}
    return data,{'A':raw},refs


def test_zero_selection_clip_kept_and_bound_to_movie():
    rows=checked_rows(*fixture())
    assert len(rows)==2 and rows[1]['temporal_f1']==0 and rows[1]['selected']==0
    assert rows[0]['cluster_id']==rows[1]['cluster_id']=='movie'


def test_reference_or_group_mismatch_fails():
    args=fixture();args[2]['a']['starts']=[50.]
    with pytest.raises(ValueError,match='temporal'):checked_rows(*args)
    args=fixture();args[0]['runs']['A']['videos'][0]['group']='other'
    with pytest.raises(ValueError,match='cluster'):checked_rows(*args)


def test_missing_clip_or_changed_selection_is_rejected():
    args=fixture();args[1]['A'].pop()
    with pytest.raises(ValueError,match='all frozen'):checked_rows(*args)
    args=fixture();args[1]['A'][0]['times']=[20.]
    with pytest.raises(ValueError,match='temporal'):checked_rows(*args)


def test_structural_family_keeps30clips_and10movies():
    plan=json.loads((Path(__file__).resolve().parents[1]/'configs/selector_ablation_contrasts.json').read_text())
    plan['resamples']=20
    names={c[k] for c in plan['contrasts'] for k in ('a','b')}
    rows=[{'example_id':str(i),'condition':name,'cluster_id':str(i//3),'temporal_f1':0.} for name in names for i in range(30)]
    results=family_analysis(rows,plan)
    assert len(results)==16 and all(r['wilcoxon']['holm_pvalue']==1 for r in results)
    assert all(len(r['units'])==10 and len(r['included_example_ids'])==30 for r in results)
    with pytest.raises(ValueError,match='all30'):family_analysis(rows[:-1],plan)
