import copy
import pytest
from scripts.preprocess_video_mme import validated_ids


def inputs():
    return ({'official_split':'test','videos':[{'video_id':'604','duration':'long'}]},
            {'complete':True,'audio_gate_passed':True,'subset_sha256':'hash',
             'videos':[{'video_id':'604','verified':True}]})


def test_exact_frozen_subset_with_no_question_fields():
    config,audit=inputs()
    ids,records=validated_ids(config,audit,'hash')
    assert ids==['604'] and records['604']['verified']


@pytest.mark.parametrize('change', ['missing','duplicate','stale','failed','short'])
def test_invalid_gate_cannot_start_preprocessing(change):
    config,audit=copy.deepcopy(inputs())
    if change=='missing': audit['videos']=[]
    if change=='duplicate': audit['videos']*=2
    if change=='stale': audit['subset_sha256']='other'
    if change=='failed': audit['complete']=False
    if change=='short': config['videos'][0]['duration']='short'
    with pytest.raises(ValueError):
        validated_ids(config,audit,'hash')


def test_verifier_decodes_pixels_and_rejects_truncated_jpeg(tmp_path):
    from PIL import Image
    from scripts.verify_video_mme_cache import check_frame
    path=tmp_path/'frame.jpg'
    Image.new('RGB',(32,32),(20,40,60)).save(path)
    frame={'index':0,'timestamp':0.,'path':str(path)}
    assert len(check_frame(frame)['sha256'])==64
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(OSError):
        check_frame(frame)
