"""Sentence-embedding cosine similarity for temporally matched AD pairs."""

from collections.abc import Iterable, Sequence
import math
from statistics import fmean, median
from typing import Any, Protocol

from .bertscore import GeneratedADItem
from .metrics import TemporalMatch
from .parsing import ADItem


DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


class SentenceEncoder(Protocol):
    """The subset of SentenceTransformer.encode used for evaluation."""

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Iterable[Iterable[float]]:
        """Return one sentence vector per input, in input order."""
        ...


def _unit_vector(embedding: Iterable[float]) -> list[float]:
    """Validate and normalize a sentence vector, including injected outputs."""
    values = [float(value) for value in embedding]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Embeddings must be nonempty vectors of finite numbers")
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embeddings must have a finite, nonzero norm")
    return [value / norm for value in values]


def compute_cosine_similarity(
    generated_items: Sequence[GeneratedADItem],
    ground_truth_items: Sequence[ADItem],
    matches: Sequence[TemporalMatch],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    encoder: SentenceEncoder | None = None,
) -> dict[str, Any]:
    """Score only supplied temporal matches using their original item indices.

    With no matches, return None aggregates without importing or loading a
    model. Otherwise, encode matched candidates followed by matched references
    in one batch with normalize_embeddings=True, then compare corresponding
    rows only. No matching is performed here and temporal scores are separate.

    The injected encoder must provide the encode interface above and use the
    model identified by model_name. Vectors are normalized again to tolerate
    floating-point drift or unnormalized injected outputs. Invalid matches,
    missing matched text, invalid vectors, or inconsistent shapes raise an
    error instead of producing fabricated scores. Output follows match order.
    """
    result: dict[str, Any] = {
        "num_scored_pairs": 0,
        "cosine_similarity_mean": None,
        "cosine_similarity_median": None,
        "pairs": [],
        "model_name": model_name,
    }
    if len(matches) == 0:
        return result

    candidates: list[str] = []
    references: list[str] = []
    used_predictions: set[int] = set()
    used_gt: set[int] = set()
    for i, match in enumerate(matches):
        prediction_index = match.prediction_index
        ground_truth_index = match.ground_truth_index
        if not 0 <= prediction_index < len(generated_items):
            raise ValueError(f"matches[{i}].prediction_index is out of range")
        if not 0 <= ground_truth_index < len(ground_truth_items):
            raise ValueError(f"matches[{i}].ground_truth_index is out of range")
        if prediction_index in used_predictions or ground_truth_index in used_gt:
            raise ValueError("matches must be one-to-one without reused entries")
        generated = generated_items[prediction_index]
        reference = ground_truth_items[ground_truth_index]
        if (
            generated.timestamp != match.prediction_time
            or reference.timestamp != match.ground_truth_time
        ):
            raise ValueError(f"matches[{i}] timestamps disagree with the indexed items")
        for label, text in (("generated", generated.text), ("ground-truth", reference.text)):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"matches[{i}] {label} text must be a nonempty string")
        used_predictions.add(prediction_index)
        used_gt.add(ground_truth_index)
        candidates.append(generated.text)
        references.append(reference.text)

    if encoder is None:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model_name)

    embeddings = list(encoder.encode(
        candidates + references,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ))
    count = len(matches)
    if len(embeddings) != 2 * count:
        raise ValueError("Encoder must return one vector per matched candidate/reference")
    vectors = [_unit_vector(embedding) for embedding in embeddings]
    if any(len(vector) != len(vectors[0]) for vector in vectors):
        raise ValueError("All embeddings must have the same dimension")
    similarities = [
        # Limit tiny floating-point overshoots to the cosine range.
        max(-1.0, min(1.0, math.fsum(
            a * b for a, b in zip(vectors[i], vectors[count + i], strict=True)
        )))
        for i in range(count)
    ]
    result.update(
        num_scored_pairs=count,
        cosine_similarity_mean=fmean(similarities),
        cosine_similarity_median=median(similarities),
        pairs=[
            {
                "prediction_index": match.prediction_index,
                "ground_truth_index": match.ground_truth_index,
                "cosine_similarity": similarities[i],
            }
            for i, match in enumerate(matches)
        ],
    )
    return result
