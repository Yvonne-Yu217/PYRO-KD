import pytest
from src.evaluation.event_alignment import alignment


def test_positive_negative_and_undefined_associations():
    r=alignment([0,1,2,3],[[1,0,0],[0,1,0],[1,0,0],[0,1,0]],
                [[1,0,0],[0,1,0],[1,0,0],[0,1,0]],[0,2],tolerance=0)
    assert r[0]['phi']==1 and r[1]['phi']==-1 and r[2]['phi'] is None
    assert r[0]['both']==2 and r[0]['near_ad_given_event']==1


def test_thresholds_and_tolerance_are_inclusive():
    r=alignment([0,3,6],[[.5],[.49],[.8]],[[.1],[.5],[.09]],[2],tolerance=2)[0]
    assert r['event_windows']==1 and r['near_ad_windows']==2 and r['both']==1


def test_absent_ad_starts_do_not_fabricate_correlation():
    assert alignment([0,1],[[1],[0]],[[1],[0]],[])[0]['phi'] is None


@pytest.mark.parametrize('times,probabilities', [([1,0],[[0],[1]]),([0,1],[[float('nan')],[1]]),([0,1],[[2],[1]])])
def test_invalid_grid_or_probabilities_fail(times,probabilities):
    with pytest.raises(ValueError):alignment(times,probabilities,[[0],[1]],[0])
