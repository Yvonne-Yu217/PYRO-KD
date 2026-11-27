from scripts.run_ad_vlm import finalize_generation
from src.generation.budget_repair import repair_once


def result(text, budget=2):
    return finalize_generation(text,text,5,1,.2,128,budget)


def test_one_repair_preserves_original_and_cost():
    initial=result('boat on water.')
    calls=[]
    def generate(prompt):
        calls.append(prompt)
        return result('Boat floats.')
    repaired=repair_once(initial,'Describe.',2,generate)
    assert repaired['output']=='Boat floats.' and 'error' not in repaired
    assert repaired['generation_calls']==2 and repaired['total_new_tokens']==10
    assert repaired['generation_attempts'][0]['output']=='boat on water.'
    assert 'error' in initial and len(calls)==1


def test_failed_repair_does_not_loop_or_truncate():
    repaired=repair_once(result('boat on water.'),'Describe.',2,lambda prompt:result('A boat floats.'))
    assert repaired['word_limit_exceeded'] and repaired['output']=='A boat floats.'
    assert len(repaired['generation_attempts'])==2


def test_valid_and_truncated_outputs_are_not_repaired():
    def forbidden(prompt):raise AssertionError('unexpected repair')
    for initial in [result('Boat floats.'),dict(result('boat on water.'),hit_token_limit=True)]:
        repaired=repair_once(initial,'Describe.',2,forbidden)
        assert repaired['generation_calls']==1


def test_repair_exception_retains_original_failure():
    def failing(prompt):raise RuntimeError('simulated OOM')
    repaired=repair_once(result('boat on water.'),'Describe.',2,failing)
    assert repaired['output']=='boat on water.' and 'error' in repaired
    assert 'simulated OOM' in repaired['repair_error']
