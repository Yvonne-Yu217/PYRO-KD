import hashlib
import numpy as np
import pytest
import soundfile as sf
from src.generation.native_audio import span_for_center,load_native_audio


def test_span_clips_only_at_actual_audio_boundaries():
    assert span_for_center(2,640000,16000)==(0,272000)
    assert span_for_center(20,640000,16000)==(80000,560000)
    assert span_for_center(39,640000,16000)==(384000,640000)


@pytest.mark.parametrize('center',[float('nan'),-1,41])
def test_invalid_center_rejected(center):
    with pytest.raises(ValueError):span_for_center(center,640000,16000)


def test_exact_samples_and_changed_source_or_span_rejected(tmp_path):
    path=tmp_path/'source.wav';wave=np.arange(32000,dtype=np.float32)/64000
    sf.write(path,wave,16000,subtype='FLOAT')
    record=dict(path=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_samples=32000,sample_rate=16000,start_seconds=.75,end_seconds=1.25,frame_center_time=1.,context_seconds=.25,
                start_sample=12000,end_sample=20000,samples_sha256=hashlib.sha256(wave[12000:20000].tobytes()).hexdigest())
    np.testing.assert_array_equal(load_native_audio(record),wave[12000:20000])
    with pytest.raises(ValueError,match='boundaries'):load_native_audio(dict(record,start_sample=0))
    with pytest.raises(ValueError,match='slice hash'):load_native_audio(dict(record,samples_sha256='wrong'))
    with pytest.raises(ValueError,match='format'):load_native_audio(dict(record,sample_rate=8000))
    with pytest.raises(ValueError,match='seconds'):load_native_audio(dict(record,end_seconds=2))
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='source hash'):load_native_audio(record)
