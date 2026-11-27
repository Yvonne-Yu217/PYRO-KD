import copy
import pytest
from scripts.export_qa_paired_metrics import export_reports


def report():
    rows=[dict(id='q1',video_id='video',answer_index=1,predicted_index=1,correct=True,status='valid'),
          dict(id='q2',video_id='video',answer_index=0,predicted_index=None,correct=False,status='missing')]
    return dict(membership_sha256='membership',subset_sha256='subset',annotations_sha256='annotations',
                run_fingerprint='run',expected=2,correct=1,accuracy=.5,rows=rows)


def test_missing_predictions_preserved_and_questions_share_video_cluster():
    a=report();b=copy.deepcopy(a)
    b['rows'][0].update(predicted_index=0,correct=False);b.update(correct=0,accuracy=0)
    rows=export_reports([('A',a),('B',b)],'video-mme')
    assert len(rows)==4 and {r['cluster_id'] for r in rows}=={'video'}
    assert [r['accuracy'] for r in rows]==[1.,0.,0.,0.]


@pytest.mark.parametrize('change',['source','membership','video','label','correctness','summary','duplicate'])
def test_misaligned_or_inconsistent_comparisons_rejected(change):
    a=report();b=copy.deepcopy(a)
    if change=='source':b['subset_sha256']='changed'
    if change=='membership':b['rows'][1]['id']='other'
    if change=='video':b['rows'][1]['video_id']='other'
    if change=='label':b['rows'][1]['answer_index']=2
    if change=='correctness':b['rows'][1]['correct']=True
    if change=='summary':b['accuracy']=1.
    if change=='duplicate':b['rows'][1]['id']='q1'
    with pytest.raises(ValueError):export_reports([('A',a),('B',b)],'video-mme')
