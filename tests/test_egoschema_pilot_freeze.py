import pytest
from scripts.freeze_egoschema_pilot import freeze


def test_question_blind_disjoint_reproducible_partition():
    ids=[f'v{i}' for i in range(500)]
    pilot,heldout=freeze(ids)
    assert (pilot,heldout)==freeze(reversed(ids))
    assert len(pilot)==10 and len(heldout)==490
    assert not set(pilot)&set(heldout) and set(pilot)|set(heldout)==set(ids)


@pytest.mark.parametrize('ids,count',[(['a','a'],1),(['a','b'],2),(['a','b'],0)])
def test_invalid_membership_or_budget_rejected(ids,count):
    with pytest.raises(ValueError): freeze(ids,count)
