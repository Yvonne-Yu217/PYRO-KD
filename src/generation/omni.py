"""Qwen2.5 Omni text-output backend; waveform inputs are explicitly optional."""
from src.generation.hf_vlm import build_conversation, clean


def load_model(model_id, attn_impl=None):
    import torch
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniForConditionalGeneration
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id, local_files_only=True)
    kwargs = dict(dtype=torch.bfloat16, device_map='auto', local_files_only=True)
    if attn_impl:
        kwargs['attn_implementation'] = attn_impl
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.disable_talker()
    return processor, model.eval(), type(model).__name__


def encode(processor, images, prompt, waveform=None):
    conversation = build_conversation(images, prompt)
    if waveform is not None:
        conversation[0]['content'].insert(-1, {'type': 'audio', 'audio': waveform})
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    kwargs = dict(text=[text], images=images, return_tensors='pt', padding=True)
    if waveform is not None:
        kwargs.update(audio=[waveform], sampling_rate=16000)
    encoded = processor(**kwargs)
    if 'pixel_values' not in encoded:
        raise ValueError('Omni processor omitted visual inputs')
    if waveform is not None and 'input_features' not in encoded:
        raise ValueError('Omni processor omitted waveform inputs')
    return encoded


def generate_one(processor, model, images, prompt, waveform=None,
                 max_new_tokens=64, temperature=0., top_p=.9):
    import torch
    # The installed wrapper accesses self.talker even after disable_talker().
    # Its text-only path delegates to this same multimodal thinker.
    thinker = model.thinker
    inputs = encode(processor, images, prompt, waveform).to(thinker.device, dtype=torch.bfloat16)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                  use_audio_in_video=False)
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p)
    with torch.inference_mode():
        output = thinker.generate(**inputs, **kwargs)
    if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
        raise ValueError('expected one text token sequence from Omni')
    new = output[:, inputs['input_ids'].shape[-1]:]
    raw = processor.batch_decode(new, skip_special_tokens=False)
    text = processor.batch_decode(new, skip_special_tokens=True)
    pad = getattr(processor.tokenizer, 'pad_token_id', None)
    count = int(new.shape[-1]) if pad is None else int((new != pad).sum())
    return [clean(text[0])], raw, [count]
