"""Frozen open VLMs via Hugging Face transformers (adapted from the team's
HPC smoke-test runner). One loader and one batched multi-image generate call
serve every model, so selectors and priors share the same generation path.
torch/transformers are imported lazily so prompt code stays testable on CPU.
"""

import re
from typing import Any, Sequence

# Short names -> Hugging Face repo ids. Any full repo id also works.
MODELS = {
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "muse-glimmer-30b": "meta-models/Muse-Glimmer-30B",
}

_QUANT_SKIP_MODULES = ["vision", "visual", "vision_tower", "vision_encoder", "lm_head"]
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
END_TOKENS = ("<|eot|>", "<|im_end|>", "<|endoftext|>", "<|end|>")


def resolve_model(name: str) -> str:
    return MODELS.get(name, name)


def clean(text: str) -> str:
    """Strip reasoning blocks and stray end-of-turn tokens."""
    text = THINK_RE.sub("", text)
    for token in END_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def load_model(model_id: str, attn_impl: str | None = None, quant: str = "none"):
    """Load (processor, model, class_name) in bf16 with device_map='auto'.

    quant='4bit'/'8bit' uses bitsandbytes for the language model only (the
    vision tower stays bf16) so Muse Glimmer 30B fits one 40 GB A100; report
    the quantization next to any results.
    """
    import torch
    import transformers
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"  # required for batched generation

    kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "device_map": "auto"}
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    if quant != "none":
        from transformers import BitsAndBytesConfig

        if quant == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=_QUANT_SKIP_MODULES,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True, llm_int8_skip_modules=_QUANT_SKIP_MODULES,
            )

    # Muse Glimmer registers under AutoModelForMultimodalLM; Qwen-VL under
    # AutoModelForImageTextToText. Try both so one loader serves every model.
    errors = []
    for cls_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, **kwargs)
            return processor, model.eval(), cls_name
        except (ValueError, KeyError) as exc:
            errors.append(f"{cls_name}: {exc}")
    raise RuntimeError(
        f"Could not load {model_id}. Upgrade transformers (pip install -U transformers).\n"
        + "\n".join(errors)
    )


def build_conversation(images: Sequence[Any], prompt: str, system: str | None = None) -> list[dict]:
    """Chronological frames first, then the text prompt, in one user turn."""
    content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    conversation: list[dict[str, Any]] = []
    if system:
        conversation.append({"role": "system", "content": [{"type": "text", "text": system}]})
    conversation.append({"role": "user", "content": content})
    return conversation


def generate_batch(
    processor,
    model,
    image_lists: Sequence[Sequence[Any]],
    prompts: Sequence[str],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 0.9,
    reasoning_strength: str | None = None,
    system: str | None = None,
) -> tuple[list[str], list[str], list[int]]:
    """Generate one description per (frames, prompt); returns (text, raw, new_tokens)."""
    import torch

    conversations = [
        build_conversation(images, prompt, system)
        for images, prompt in zip(image_lists, prompts)
    ]
    template_kwargs: dict[str, Any] = dict(
        tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, padding=True,
    )
    if reasoning_strength:  # Muse Glimmer only
        template_kwargs["reasoning_strength"] = reasoning_strength
    inputs = processor.apply_chat_template(conversations, **template_kwargs)
    inputs = inputs.to(model.device, dtype=torch.bfloat16)  # casts float tensors only

    gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.inference_mode():
        output = model.generate(**inputs, **gen_kwargs)

    new = output[:, inputs["input_ids"].shape[-1]:]
    raw = processor.batch_decode(new, skip_special_tokens=False)
    text = processor.batch_decode(new, skip_special_tokens=True)
    pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
    if pad_id is None:
        new_tokens = [new.shape[-1]] * new.shape[0]
    else:
        new_tokens = (new != pad_id).sum(dim=-1).tolist()
    return [clean(t) for t in text], raw, new_tokens
