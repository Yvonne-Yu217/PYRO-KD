import pytest
from scripts.evaluate_video_mme_qa import score_long_qa


def fixture():
    membership={'expected_questions':3,'questions':[{'question_id':q,'video_id':v}
                                                  for q,v in [('q1','a'),('q2','a'),('q3','b')]]}
    answers=[dict(r,answer='B') for r in membership['questions']]
    videos={'a':{'domain':'sports'},'b':{'domain':'art'}}
    prediction={'id':'q1','video_id':'a','output':'B','predicted_index':1,'selector':'U','run_fingerprint':'same'}
    return membership,answers,videos,prediction


def test_missing_questions_count_in_domain_and_video_denominators():
    membership,answers,videos,prediction=fixture()
    result=score_long_qa([prediction],membership,answers,videos)
    assert result['accuracy']==1/3 and not result['complete']
    assert result['by_video']['a']['accuracy']==.5
    assert result['by_domain']['art']['accuracy']==0
    assert result['rows'][2]['video_id']=='b'


def test_invalid_answer_and_truncation_are_incorrect():
    membership,answers,videos,prediction=fixture()
    rows=[prediction,dict(prediction,id='q2',output='B or C',predicted_index=None),
          dict(prediction,id='q3',video_id='b',hit_token_limit=True)]
    result=score_long_qa(rows,membership,answers,videos)
    assert result['correct']==1 and result['expected']==3


@pytest.mark.parametrize('change',['missing_label','duplicate_label','wrong_video','wrong_label','condition'])
def test_inconsistent_sources_rejected(change):
    membership,answers,videos,prediction=fixture();predictions=[prediction]
    if change=='missing_label':answers.pop()
    if change=='duplicate_label':answers.append(answers[0])
    if change=='wrong_video':prediction['video_id']='b'
    if change=='wrong_label':answers[0]['answer']='B explanation'
    if change=='condition':predictions.append(dict(prediction,id='q2',audio_context='timeline'))
    with pytest.raises(ValueError):score_long_qa(predictions,membership,answers,videos)


def test_empty_prediction_cli_never_opens_labels_or_membership(tmp_path,monkeypatch):
    import sys
    from scripts.evaluate_video_mme_qa import main
    source=tmp_path/'empty.jsonl';source.write_text('')
    output=tmp_path/'score.json'
    monkeypatch.setattr(sys,'argv',['evaluate','--input',str(source),'--annotations',str(tmp_path/'unread'),
                                   '--membership',str(tmp_path/'unread-membership'),'--output',str(output)])
    with pytest.raises(ValueError,match='no predictions'):main()
    assert not output.exists()
