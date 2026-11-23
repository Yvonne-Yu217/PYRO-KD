"""Temporary synthetic input and mocked inference; tests runner wiring/resume only."""
import hashlib
import json
import sys
import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from scripts import run_ad_vlm
from src.generation import omni


def test_native_runner_records_waveform_and_resumes_without_loading(tmp_path, monkeypatch):
    audio = tmp_path / 'audio.wav'
    sf.write(audio, np.zeros(32000, dtype=np.float32), 16000, subtype='FLOAT')
    frame = tmp_path / 'frame.jpg'
    Image.new('RGB', (32, 32)).save(frame)
    waveform = np.zeros(32000, dtype=np.float32)
    record = dict(path=str(audio), source_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
                  source_samples=32000, sample_rate=16000, frame_center_time=1.,
                  context_seconds=15., start_sample=0, end_sample=32000,
                  start_seconds=0., end_seconds=2., samples_sha256=hashlib.sha256(waveform.tobytes()).hexdigest())
    row = dict(id='test', video_id='synthetic', frames=[str(frame)], time=1.,
               frame_center_time=1., word_limit=10, transcript='', native_audio=record)
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps(row)+'\n')
    output = tmp_path / 'result.jsonl'
    monkeypatch.setattr(sys, 'argv', ['run_ad_vlm.py', '--input', str(source), '--output', str(output),
                                     '--backend', 'omni', '--audio-context', 'native', '--batch-size', '1'])
    monkeypatch.setattr(omni, 'load_model', lambda *args: (object(), object(), 'FakeOmni'))
    calls = []
    def generate(processor, model, images, prompt, wave, *args):
        np.testing.assert_array_equal(wave, waveform)
        assert len(images) == 1 and 'audio covers 0.000000 to 2.000000' in prompt
        calls.append(True)
        return ['A test description.'], ['A test description.'], [4]
    monkeypatch.setattr(omni, 'generate_one', generate)
    with pytest.raises(SystemExit) as exit_info:
        run_ad_vlm.main()
    assert exit_info.value.code == 0 and len(calls) == 1
    result = json.loads(output.read_text())
    assert result['native_audio'] == record
    assert result['backend'] == 'omni' and result['audio_context'] == 'native'
    assert result['word_limit'] == 10 and result['output'] == 'A test description.'
    original = output.read_bytes()
    def no_load(*args):
        raise AssertionError('settled output must not load the model')
    monkeypatch.setattr(omni, 'load_model', no_load)
    run_ad_vlm.main()
    assert output.read_bytes() == original
    audio.write_bytes(b'changed audio')
    with pytest.raises(ValueError, match='source hash'):
        run_ad_vlm.main()
    assert output.read_bytes() == original
