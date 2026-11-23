from scripts.run_ad_vlm import finalize_generation
from src.generation.outputs import is_failure, split_rows, status_counts


def test_over_budget_preserves_decode_and_fails():
    result=finalize_generation('boat on water.', 'boat on water.<end>', 5, 1, .5, 128, 2)
    assert result['output']=='boat on water.' and result['raw_output']=='boat on water.<end>'
    assert result['word_count']==3 and result['word_limit_exceeded']
    assert is_failure(result)


def test_old_unflagged_over_budget_rows_are_not_scored_as_success():
    old=dict(id='a',output='man smiles and looks to side.',word_limit=5)
    assert is_failure(old)
    corrected=dict(id='a',output='A man smiles.',word_limit=5)
    final, history=split_rows([old,corrected])
    assert final==[corrected] and history==[old]
    assert status_counts([old])['n_error']==1


def test_at_budget_and_unbudgeted_qa_remain_valid():
    result=finalize_generation('A man smiles.', 'A man smiles.', 4, 1, .5, 128, 3)
    assert not is_failure(result) and result['word_count']==3
    assert not is_failure(dict(output='A'))
