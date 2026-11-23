import pytest
from scripts.export_measured_gap_utilization import gap_metrics, summarize_video, aggregate


def narration(key,start,duration):
    return {'id':key,'start':start,'duration':duration}


def test_overlap_uses_union_not_sum():
    result=gap_metrics([narration('a',0,8),narration('b',2,8)],[],10)
    gap=result['gaps'][0]
    assert gap['utilization']==1
    assert gap['layered_intersection_ratio']==1.6
    assert gap['nominal_start_duration_ratio']==1.6
    assert result['excess_layers_in_gaps_seconds']==6


def test_boundary_crossing_retains_collisions_and_outside():
    result=summarize_video([narration('a',1,5),narration('b',8,4)],[(3,5)],[],10)
    gaps=result['tracks']['union']['gaps']
    assert [g['unique_narration_seconds'] for g in gaps]==[2,3]
    assert result['original_collision_metrics']['speech_overlap_seconds_timeline']['union']==2
    assert result['original_collision_metrics']['out_of_video_seconds_sum']==2
    assert result['tracks']['union']['pooled_utilization']==5/8


def test_unused_gaps_and_zero_output_videos_remain_in_denominator():
    first=summarize_video([narration('a',0,1)],[(2,4)],[],6)
    empty=summarize_video([],[],[],6)
    result=aggregate([first,empty])
    assert result['zero_render_videos']==1
    assert result['tracks']['union']['unused_gaps']==2
    assert result['tracks']['union']['total_gap_seconds']==10
    assert result['tracks']['union']['pooled_utilization']==.1
    assert result['tracks']['union']['macro_video_utilization']==.125


def test_all_speech_has_no_defined_gap_ratio():
    video=summarize_video([],[(0,10)],[],10)
    result=aggregate([video])
    assert result['tracks']['union']['pooled_utilization'] is None
    assert result['tracks']['union']['videos_with_gaps']==0
    assert video['tracks']['union']['gaps']==[]


def test_half_open_start_assignment_and_remaining_gap():
    result=gap_metrics([narration('speech',2,1),narration('gap',4,1)],[(2,4)],6)
    assert result['captions_without_start_gap']==1
    assert result['per_caption_remaining_gap_ratios']==[{'id':'gap','ratio':.5}]


def test_invalid_sources_rejected():
    with pytest.raises(ValueError):gap_metrics([],[(0,11)],10)
    with pytest.raises(ValueError):summarize_video([narration('a',0,1),narration('a',1,1)],[],[],10)
    with pytest.raises(ValueError):gap_metrics([],[],0)
