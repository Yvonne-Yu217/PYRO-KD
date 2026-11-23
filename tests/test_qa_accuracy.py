import pytest
from src.evaluation.qa import score_qa


def prediction(ident='a',text='B',index=1,**extra):
    return {'id':ident,'video_id':ident,'output':text,'predicted_index':index,'run_fingerprint':'run',**extra}


def test_missing_errors_and_invalid_are_in_denominator():
    result=score_qa([prediction(),prediction('b',error='OOM'),prediction('c','B or C',None)],
                    {'a':1,'b':1,'c':1,'d':1},5)
    assert result['accuracy']==.25 and result['accuracy_among_valid']==1
    assert result['status_counts']=={'valid':1,'generation_error':1,'invalid_answer':1,'missing':1}
    assert not result['complete']


def test_retry_success_supersedes_error_without_extra_denominator():
    result=score_qa([prediction(error='OOM'),prediction()],{'a':1},5)
    assert result['accuracy']==1 and result['expected']==1 and result['complete']
    assert result['superseded_error_attempts']==1


@pytest.mark.parametrize('rows',[ [prediction(),prediction()], [prediction('unknown')],
    [prediction(index=2)], [prediction(),prediction('b',run_fingerprint='other')] ])
def test_inconsistent_predictions_rejected(rows):
    with pytest.raises(ValueError): score_qa(rows,{'a':1,'b':1},5)


def test_empty_has_explicit_missing_and_no_valid_conditional_score():
    result=score_qa([],{'a':1},5)
    assert result['accuracy']==0 and result['accuracy_among_valid'] is None
    assert result['status_counts']['missing']==1


def test_truncated_answer_is_error_even_if_letter_would_match():
    result=score_qa([prediction(hit_token_limit=True)],{'a':1},5)
    assert result['correct']==0 and result['status_counts']['generation_error']==1


def test_runner_invalid_answer_error_has_its_own_category():
    result=score_qa([prediction(text='B or C',index=None,error='invalid_answer: expected a single option letter')],{'a':1},5)
    assert result['status_counts']['invalid_answer']==1
    assert result['correct']==0


def test_mixed_selectors_rejected_even_with_one_fingerprint():
    with pytest.raises(ValueError,match='mixed selectors'):
        score_qa([prediction(selector='U'),prediction('b',selector='V')],{'a':1,'b':1},5)


def test_cli_does_not_read_labels_or_write_score_without_predictions(tmp_path,monkeypatch):
    import sys
    from scripts.evaluate_egoschema_qa import main
    source=tmp_path/'empty.jsonl';source.write_text('')
    output=tmp_path/'score.json'
    monkeypatch.setattr(sys,'argv',['evaluate_egoschema_qa','--input',str(source),
        '--answers',str(tmp_path/'labels-must-not-be-read.json'),'--split','pilot','--output',str(output)])
    with pytest.raises(ValueError,match='no predictions'): main()
    assert not output.exists()
