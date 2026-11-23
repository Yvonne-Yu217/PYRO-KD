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


class Model:
    device = 'cpu'

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4]])


@pytest.mark.parametrize('waveform', [None, [0., .1, -.1]])
def test_waveform_is_explicit_and_output_is_text_only(waveform):
    processor, model = Processor(), Model()
    text, raw, count = generate_one(processor, model, ['frame'], 'prompt', waveform)
    assert text == raw == ['A person walks.'] and count == [2]
    assert model.kwargs['generation_mode'] == 'text'
    assert model.kwargs['thinker_max_new_tokens'] == 64
    assert model.kwargs['thinker_do_sample'] is False
    assert ('audio' in processor.kwargs) == (waveform is not None)
    assert ('input_features' in model.kwargs) == (waveform is not None)
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
    class AudioModel(Model):
        def generate(self, **kwargs):
            return super().generate(**kwargs), torch.zeros(16000)
    with pytest.raises(ValueError, match='text token sequence'):
        generate_one(Processor(), AudioModel(), ['frame'], 'prompt')
