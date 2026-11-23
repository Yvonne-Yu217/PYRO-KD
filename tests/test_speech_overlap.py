import pytest
from src.evaluation.speech_overlap import evaluate_intervals, intersection_seconds


def test_union_does_not_double_count_detectors_or_ad():
    result = evaluate_intervals([
        {'id':'a','start':1.,'duration':4.},
        {'id':'b','start':3.,'duration':3.}],
        [(2.,4.),(3.,5.)], [(3.,4.),(4.,5.5)], 10.)
    assert result['speech_overlap_seconds_sum'] == {'vad':5.,'whisper':4.5,'union':5.5}
    assert result['speech_overlap_seconds_timeline'] == {'vad':3.,'whisper':2.5,'union':3.5}
    assert result['narration_excess_layer_seconds'] == 2.
    assert result['fraction_descriptions_fitting'] == 0


def test_touching_endpoints_are_not_overlap_and_overrun_is_not_silence():
    result = evaluate_intervals([
        {'id':'fits','start':2.,'duration':2.},
        {'id':'overrun','start':9.,'duration':3.},
        {'id':'outside','start':12.,'duration':2.}], [(0.,2.),(4.,5.)], [], 10.)
    assert result['fraction_descriptions_fitting'] == pytest.approx(1/3)
    assert result['out_of_video_seconds_sum'] == 4
    assert [r['speech_free_in_video_seconds'] for r in result['rows']] == [2,1,0]


def test_empty_evaluation_is_undefined_not_perfect():
    assert evaluate_intervals([], [], [], 10)['fraction_descriptions_fitting'] is None


@pytest.mark.parametrize('intervals', [[(-1,2)],[(2,1)],[(0,float('nan'))]])
def test_invalid_intervals_fail(intervals):
    with pytest.raises(ValueError):
        intersection_seconds(intervals, [])


def test_duplicate_ids_and_out_of_bounds_speech_fail():
    row={'id':'a','start':0,'duration':1}
    with pytest.raises(ValueError,match='unique'):
        evaluate_intervals([row,row],[],[],10)
    with pytest.raises(ValueError,match='exceeds'):
        evaluate_intervals([],[(9,11)],[],10)


def test_actual_tempo_conversion_measures_samples(tmp_path):
    import math
    import shutil
    import struct
    import wave
    from src.evaluation.speech_overlap import accelerate_rendered_audio
    if not shutil.which('ffmpeg'):
        pytest.skip('ffmpeg is required for the actual PCM conversion test')
    source, target = tmp_path/'original.wav', tmp_path/'accelerated.wav'
    with wave.open(str(source), 'wb') as wav:
        wav.setparams((1,2,16000,0,'NONE','not compressed'))
        wav.writeframes(b''.join(struct.pack('<h',int(10000*math.sin(2*math.pi*440*i/16000))) for i in range(48000)))
    duration = accelerate_rendered_audio(source, target)
    assert duration == pytest.approx(2.5, abs=.04)
    with wave.open(str(target), 'rb') as wav:
        assert duration == wav.getnframes()/wav.getframerate()
    with pytest.raises(FileExistsError):
        accelerate_rendered_audio(source, target)


@pytest.mark.parametrize('change, message', [
    ({'tts_playback_speed':1.2}, 'unaccelerated'),
    ({'tts_text_sha256':'wrong'}, 'text binding'),
    ({'tts_audio_sha256':'wrong'}, 'waveform changed')])
def test_cli_rejects_wrong_render_binding_before_creating_output(tmp_path, monkeypatch, change, message):
    import hashlib
    import json
    import sys
    from scripts.evaluate_rendered_overlap import main
    audio=tmp_path/'original.wav'
    audio.write_bytes(b'test fixture: rejected before decoding')
    row={'id':'a','video_id':'v','time':1,'output':'A person enters.',
         'tts_playback_speed':1.0,'tts_audio':str(audio),
         'tts_audio_sha256':hashlib.sha256(audio.read_bytes()).hexdigest(),
         'tts_text_sha256':hashlib.sha256(b'A person enters.').hexdigest(),**change}
    source=tmp_path/'input.jsonl'
    source.write_text(json.dumps(row)+'\n')
    output=tmp_path/'result'
    monkeypatch.setattr(sys,'argv',['evaluate_rendered_overlap','--input',str(source),
        '--cache-dir',str(tmp_path/'cache'),'--output-dir',str(output)])
    with pytest.raises(ValueError,match=message):
        main()
    assert not output.exists()
