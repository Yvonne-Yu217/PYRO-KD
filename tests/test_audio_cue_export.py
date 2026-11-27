import pytest
from scripts.export_audio_cue_diagnostics import summarize,association_rows,CORE


def video():
    return {'video_id':'clip','movie_id':'movie','duration':10.,'gt_times':[1.,5.],
            'events':[{'time':1.,'label':'Music','superclass':'music'}],
            'grid':list(range(11)),'audio_change':[0.,.2]+[0.]*9,
            'selections':{'U':[1.],'V':[5.],'A_cue':[],'AV':[5.],'O':[1.,5.]}}


def test_fixed_threshold_boundary_and_zero_selection_are_retained():
    result=summarize([video()]);low=result['low_change_reference_coverage']
    assert low['n_gt']==2 and low['n_low_change_gt']==1
    assert low['selectors']['A_cue']['pooled_recall']==0
    assert low['selectors']['U']['pooled_recall']==0
    assert low['selectors']['V']['pooled_recall']==1
    assert low['selectors']['O']['oracle_upper_bound']


def test_association_uses_detector_onsets_and_timeline_chance_not_labels():
    data=video();result=summarize([data]);music=result['association_by_superclass']['classes']['music']
    assert music['hit_rate']==.5
    assert music['chance_rate']==pytest.approx(31/101)
    assert music['lift']==pytest.approx(.5/(31/101))
    rows=association_rows(result['association_by_detected_label'],[data],'label')
    assert rows[0]['detected_onsets']==1 and rows[0]['clips_with_detected_onsets']==1
    assert 'detector_precision' not in result


def test_empty_proxy_subset_remains_undefined_not_perfect_recall():
    data=video();data['audio_change']=[1.]*11;data['events']=[]
    result=summarize([data]);low=result['low_change_reference_coverage']
    assert low['n_low_change_gt']==0 and all(r['pooled_recall'] is None for r in low['selectors'].values())
    assert result['association_by_detected_label']['classes']=={}


def test_duplicate_video_and_missing_selector_refused():
    data=video()
    with pytest.raises(ValueError,match='duplicate'):summarize([data,data])
    del data['selections']['U']
    with pytest.raises(ValueError,match='five'):summarize([data])
