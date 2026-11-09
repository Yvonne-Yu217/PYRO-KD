"""Print what a model processor returns for a multi-frame prompt (no model load, CPU only).

Used to debug generation failures such as a layer receiving uint8 pixels:

    python scripts/debug_processor.py --model muse-glimmer-30b --reasoning-strength low \
        --frames data/cache/smoke/smoke/frames/000004.jpg data/cache/smoke/smoke/frames/000006.jpg
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from src.generation.hf_vlm import build_conversation, resolve_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="muse-glimmer-30b")
    parser.add_argument("--reasoning-strength", default=None)
    parser.add_argument("--frames", nargs="+", required=True)
    args = parser.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(resolve_model(args.model))
    print("processor:", type(processor).__name__)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        print("image_processor:", type(image_processor).__name__)
        for key in ("do_rescale", "do_normalize", "rescale_factor", "image_mean", "image_std", "do_resize", "size"):
            if hasattr(image_processor, key):
                print(f"  {key} = {getattr(image_processor, key)}")

    images = [Image.open(path).convert("RGB") for path in args.frames]
    conversations = [build_conversation(images, "Describe.")]
    template_kwargs = dict(
        tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, padding=True,
    )
    if args.reasoning_strength:
        template_kwargs["reasoning_strength"] = args.reasoning_strength
    inputs = processor.apply_chat_template(conversations, **template_kwargs)
    print("apply_chat_template returned:")
    for key, value in inputs.items():
        print(f"  {key}: dtype={getattr(value, 'dtype', type(value).__name__)} shape={tuple(getattr(value, 'shape', ()))}")


if __name__ == "__main__":
    main()
