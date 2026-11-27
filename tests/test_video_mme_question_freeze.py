import pytest
from scripts.freeze_video_mme_questions import freeze_questions


def video(ident):
    return dict(video_id=ident, videoID='source'+ident, duration='long', domain='domain')


def test_freezes_all_question_membership_without_content_based_selection():
    subset={'videos':[video('a'), video('b')]}
    rows=[dict(video(v), question_id=v+str(i)) for v in ('b','a','unused') for i in range(3)]
    result=freeze_questions(rows,subset)
    assert result['expected_questions']==6
    assert result['video_question_counts']=={'a':3,'b':3}
    assert [r['question_id'] for r in result['questions']]==['a0','a1','a2','b0','b1','b2']


@pytest.mark.parametrize('mutation',['answer','question','duplicate','missing','wrong_video'])
def test_leakage_or_membership_errors_fail(mutation):
    row=dict(video('a'),question_id='q')
    rows=[row]
    if mutation in ('answer','question'):row[mutation]='must not be read'
    if mutation=='duplicate':rows.append(dict(row))
    if mutation=='missing':rows=[]
    if mutation=='wrong_video':row['videoID']='changed'
    with pytest.raises(ValueError):freeze_questions(rows,{'videos':[video('a')]})
