import pytest
from scripts.build_long_qa_inputs import normalize_options,evenly,QUESTION_COLUMNS

def test_option_order_is_preserved_and_validated():
    assert normalize_options(['A. first','B. second','C. third','D. fourth'])==['first','second','third','fourth']
    with pytest.raises(ValueError,match='order'):
        normalize_options(['B. first','A. second','C. third','D. fourth'])
    assert 'answer' not in QUESTION_COLUMNS and 'question' in QUESTION_COLUMNS

def test_timeline_budget_covers_both_ends():
    rows=list(range(101));chosen=evenly(rows,20)
    assert len(chosen)==len(set(chosen))==20 and chosen[0]==0 and chosen[-1]==100
    assert evenly(rows[:3],20)==rows[:3]
