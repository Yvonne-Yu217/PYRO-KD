"""OpenCLIP text-text cosine between generated and reference AD (proposal §7C).

Pairs come from the temporal matcher (as for BERTScore), never from text
matching. The encoder is injectable; the default builds OpenCLIP's text tower
(same ViT-B-32 / laion2b weights as the visual features) lazily. Untested
against real weights here (open_clip is not installed on this laptop).
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

TextEncoder = Callable[[list[str]], list[list[float]]]


def openclip_text_encoder(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str | None = None) -> TextEncoder:
    import open_clip
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    def encode(texts: list[str]) -> list[list[float]]:
        with torch.no_grad():
            features = model.encode_text(tokenizer(texts).to(device)).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return features.cpu().tolist()

    return encode


def pair_similarities(generated: Sequence[str], references: Sequence[str], encoder: TextEncoder) -> list[float]:
    if len(generated) != len(references):
        raise ValueError("generated and references must be aligned")
    if not generated:
        return []
    vectors = encoder(list(generated) + list(references))
    n = len(generated)
    out = []
    for a, b in zip(vectors[:n], vectors[n:]):
        na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            raise ValueError("encoder returned a zero vector")
        out.append(sum(x * y for x, y in zip(a, b)) / (na * nb))
    return out
