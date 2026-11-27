import argparse
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image
from scripts.export_placed_moments import make_row
from scripts.run_ad_vlm import prepare


@pytest.fixture
def case(tmp_path):
    frames=[]
    for i in range(12):
        path=tmp_path/f'{i}.jpg'
        Image.new('RGB',(8,8)).save(path)
        frames.append({'index':i,'timestamp':i/2,'path':str(path)})
    return ({'candidate_index':2,'rank':0,'score':1,
             'placement':{'event_time':2.,'time':4.,'gap_start':4.,'gap_end':6.,'word_limit':6}},
            {'requested':2,'shortfall':1},{'frames':frames})


def test_event_window_and_playback_time_are_distinct(case):
    selected,selection,manifest=case
    event=make_row('v','AV_trigger',selected,selection,manifest,'event','hash')
    playback=make_row('v','AV_trigger',selected,selection,manifest,'placement','hash')
    assert event['time']==playback['time']==4
    assert event['event_time']==2 and event['placement_lag_seconds']==2
    assert event['frame_times']!=playback['frame_times']
    assert event['frame_center_time']==2 and playback['frame_center_time']==4
    assert event['k_requested']==2 and event['k_shortfall']==1
    assert event['word_limit']==playback['word_limit']==6
    assert event['transcript']==''
    args=argparse.Namespace(max_side=768,audio_context='none',prior='P0',span=2)
    _,prompt,_=prepare(event,args)
    assert 'around 0:02.0' in prompt


def test_invalid_capacity_rejected(case):
    selected,selection,manifest=case
    selected['placement']['word_limit']=100
    with pytest.raises(ValueError,match='capacity'):
        make_row('v','A_trigger',selected,selection,manifest,'event','hash')


@pytest.mark.parametrize('condition', ['U_snapped','O_snapped','V_local','V_memory',
                                      'A_trigger','A_suppress','AV_trigger','AV_suppress'])
def test_selector_formats_preserve_generation_contract(case, condition):
    from scripts.export_placed_moments import report_selection
    selected, selection, manifest = case
    constraints = {'max_advance': 2., 'max_lag': 2.}
    if condition.startswith('V_'):
        report = {'settings': constraints}
        video = {'modes': {condition[2:]: selection}}
    else:
        report = constraints
        video = {'conditions': {condition: selection}}
    video['annotation_text'] = 'reference description must never enter a prompt'
    resolved, bounds = report_selection(report, video, condition)
    row = make_row('v', condition, selected, resolved, manifest, 'event', 'hash')
    assert bounds == constraints
    assert row['selector'] == condition.split('_')[0]
    assert row['selection_condition'] == condition
    assert row['transcript'] == '' and 'annotation_text' not in row
    assert row['time'] == 4 and row['frame_center_time'] == 2
    assert row['k_shortfall'] == 1 and row['word_limit'] == 6


def test_missing_variant_does_not_fall_back():
    from scripts.export_placed_moments import report_selection
    with pytest.raises(KeyError):
        report_selection({'settings': {}}, {'modes': {'local': {}}}, 'V_memory')
