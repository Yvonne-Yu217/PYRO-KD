import json
import numpy as np
import pytest
from src.utils.event_checkpoints import EventCheckpoints


def arrays(count=2):
    return np.full((count,3),.5,dtype=np.float32), np.ones((count,4),dtype=np.float32)


def test_roundtrip_resume_preserves_exact_raw_vectors(tmp_path):
    first=EventCheckpoints(tmp_path,'a'*64,3,4)
    first.save(0,*arrays())
    resumed=EventCheckpoints(tmp_path,'a'*64,3,4)
    for expected,actual in zip(arrays(),resumed.load(0,2)):
        np.testing.assert_array_equal(expected,actual)
    assert resumed.load(2,2) is None
    resumed.save(2,*arrays(1))
    assert resumed.load(2,1)[0].shape==(1,3)


def test_settings_or_audio_digest_change_cannot_reuse(tmp_path):
    first=EventCheckpoints(tmp_path,'a'*64,3,4); first.save(0,*arrays())
    assert EventCheckpoints(tmp_path,'b'*64,3,4).load(0,2) is None


def test_interrupted_run_resumes_missing_batches_and_same_cues(tmp_path):
    from src.utils.event_features import event_cues
    rng=np.random.default_rng(42)
    p=rng.random((7,3),dtype=np.float32)
    e=rng.random((7,4),dtype=np.float32)
    rms=np.ones(7,dtype=np.float32)
    cache=EventCheckpoints(tmp_path,'a'*64,3,4)
    cache.save(0,p[:2],e[:2])
    computed=[];parts=[]
    for start in range(0,7,2):
        count=min(2,7-start)
        batch=cache.load(start,count)
        if batch is None:
            computed.append(start)
            batch=(p[start:start+count],e[start:start+count])
            cache.save(start,*batch)
        parts.append(batch)
    assert computed==[2,4,6]
    restored=[np.concatenate([batch[i] for batch in parts]) for i in (0,1)]
    baseline=event_cues(p,e,rms,['Music','Laughter','Applause'])
    resumed=event_cues(*restored,rms,['Music','Laughter','Applause'])
    for key in baseline:np.testing.assert_array_equal(baseline[key],resumed[key])


def test_interrupted_payload_and_corruption_recomputed(tmp_path):
    cache=EventCheckpoints(tmp_path,'a'*64,3,4); cache.save(0,*arrays())
    payload,metadata=cache.paths(0,2)
    metadata.unlink()
    assert cache.load(0,2) is None
    cache.save(0,*arrays())
    payload.write_bytes(b'truncated')
    assert cache.load(0,2) is None
    cache.save(0,*arrays())
    record=json.loads(metadata.read_text());record['start']=1
    metadata.write_text(json.dumps(record))
    assert cache.load(0,2) is None


@pytest.mark.parametrize('kind',['nan','range','shape','zero','dtype'])
def test_invalid_model_outputs_never_checkpointed(tmp_path,kind):
    cache=EventCheckpoints(tmp_path,'a'*64,3,4)
    p,e=arrays()
    if kind=='nan':p[0,0]=np.nan
    if kind=='range':p[0,0]=1.1
    if kind=='shape':e=e[:1]
    if kind=='zero':e[0]=0
    if kind=='dtype':p=p.astype(np.float64)
    with pytest.raises(ValueError):cache.save(0,p,e)
    assert cache.load(0,2) is None
