import pytest
from scripts.evaluate_placed_dev import score_condition


def test_shortfall_kept_separate_from_precision():
    result=score_condition([10],[10,30],4)
    assert result['precision']==1
    assert result['recall']==.5
    assert result['shortfall']==3
    assert result['matched_per_requested']==.25


def test_one_to_one_matching_and_boundary():
    result=score_condition([8,9,32],[10,30],3)
    assert result['tp']==2 and result['fp']==1 and result['fn']==0
    assert len(result['matches'])==2


def test_missing_selections_are_not_dropped():
    result=score_condition([],[10],2)
    assert result['f1']==0 and result['shortfall']==2


def test_over_budget_rejected():
    with pytest.raises(ValueError): score_condition([1,2],[1],1)
