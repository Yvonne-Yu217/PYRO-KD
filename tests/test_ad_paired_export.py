import copy
import pytest
from scripts.export_ad_paired_metrics import export_reports
from scripts.analyze_paired import analyze


def report():
    return dict(dataset='reframed',scope='all_selected_videos',split='validation',
                selection_config_sha256='frozen-membership',metadata={'run_fingerprint':'synthetic-run'},
                coverage={'num_selected_videos':2},evaluation_config=dict(tolerance_seconds=2.,
                reference_track='merged_us',reference_merge_gap_seconds=.5),
                videos=[dict(video_id=ident,movie_id='same-movie',reference_sha256='reference-'+ident,
                             num_generated=1,num_ground_truth=2,num_generation_errors=0,
                             temporal=dict(tp=1,fp=0,fn=1,precision=1.,recall=.5,f1=2/3))
                        for ident in ['a','b']])


def test_missing_predictions_stay_zero_and_clips_cluster_by_movie():
    a,b=report(),report()
    b['videos'][0].update(num_generated=0,num_generation_errors=1,
        temporal=dict(tp=0,fp=0,fn=2,precision=0.,recall=0.,f1=0.))
    rows=export_reports([('a',a),('b',b)])
    assert len(rows)==4
    result=analyze(rows,'a','b','temporal_f1','cluster',resamples=20)
    assert len(result['units'])==1 and result['units'][0]['n_examples']==2
    assert result['units'][0]['a']==pytest.approx(2/3)
    assert result['units'][0]['b']==pytest.approx(1/3)
    assert result['bootstrap']['ci_low'] is None


@pytest.mark.parametrize('change', ['reference','movie','membership','protocol','counts','metric','coverage','fingerprint'])
def test_rejects_incomparable_or_inconsistent_reports(change):
    a,b=report(),copy.deepcopy(report())
    if change=='reference':b['videos'][0]['reference_sha256']='changed'
    if change=='movie':b['videos'][0]['movie_id']='changed'
    if change=='membership':b['selection_config_sha256']='changed'
    if change=='protocol':b['evaluation_config']['tolerance_seconds']=3.
    if change=='counts':b['videos'][0]['temporal']['tp']=2
    if change=='metric':b['videos'][0]['temporal']['f1']=.9
    if change=='coverage':b['videos'].pop()
    if change=='fingerprint':b['metadata'].clear()
    with pytest.raises(ValueError):export_reports([('a',a),('b',b)])
