"""Backend contract tests with a fake model, never evidence of real inference."""
import pytest
import torch
from src.generation.omni import encode, generate_one


class Inputs(dict):
    def to(self, *args, **kwargs):
        return self


class Processor:
    tokenizer = type('Tokenizer', (), {'pad_token_id': 0})()

    def apply_chat_template(self, conversation, **kwargs):
        self.conversation = conversation
        return 'template'

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        result = Inputs(input_ids=torch.tensor([[1, 2]]), pixel_values=torch.ones(1))
        if 'audio' in kwargs:
            result['input_features'] = torch.ones(1)
        return result

    def batch_decode(self, new, skip_special_tokens):
        assert new.tolist() == [[3, 4]]
        return ['A person walks.']


class Thinker:
    device = 'cpu'

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4]])


class Model:
    def __init__(self):
        self.thinker = Thinker()

    def generate(self, **kwargs):
        raise AssertionError('disabled-talker wrapper must never be called')


@pytest.mark.parametrize('waveform', [None, [0., .1, -.1]])
def test_waveform_is_explicit_and_output_is_text_only(waveform):
    processor, model = Processor(), Model()
    text, raw, count = generate_one(processor, model, ['frame'], 'prompt', waveform)
    assert text == raw == ['A person walks.'] and count == [2]
    assert model.thinker.kwargs['use_audio_in_video'] is False
    assert model.thinker.kwargs['max_new_tokens'] == 64
    assert model.thinker.kwargs['do_sample'] is False
    assert ('audio' in processor.kwargs) == (waveform is not None)
    assert ('input_features' in model.thinker.kwargs) == (waveform is not None)
    if waveform is not None:
        assert processor.kwargs['sampling_rate'] == 16000
        assert processor.kwargs['audio'] == [waveform]


def test_rejects_processor_that_drops_audio():
    class MissingAudio(Processor):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result.pop('input_features', None)
            return result
    with pytest.raises(ValueError, match='waveform'):
        encode(MissingAudio(), ['frame'], 'prompt', [0.])


def test_rejects_audio_output_tuple():
    class AudioThinker(Thinker):
        def generate(self, **kwargs):
            return super().generate(**kwargs), torch.zeros(16000)
    model = Model()
    model.thinker = AudioThinker()
    with pytest.raises(ValueError, match='text token sequence'):
        generate_one(Processor(), model, ['frame'], 'prompt')


def test_experiment_backend_uses_verified_text_path(monkeypatch):
    from src.generation import omni
    from src.generation.backends import QwenOmniBackend, GenRequest
    processor, model = object(), object()
    monkeypatch.setattr(omni, 'load_model', lambda *args: (processor, model, 'test'))
    calls = []
    def generate(p, m, images, prompt, **kwargs):
        assert p is processor and m is model
        calls.append(kwargs)
        return ['A plane flies.'], ['A plane flies.<end>'], [4]
    monkeypatch.setattr(omni, 'generate_one', generate)
    backend = QwenOmniBackend('pinned-model')
    result = backend.generate([GenRequest('one', 'describe', [], max_new_tokens=16)])[0]
    assert result.error is None and result.text == 'A plane flies.' and result.new_tokens == 4
    assert calls == [{'waveform': None, 'max_new_tokens': 16, 'temperature': 0.0}]
