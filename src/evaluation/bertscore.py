"""BERTScore for text pairs selected by the temporal matcher."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import math
from typing import Any

from .metrics import TemporalMatch
from .parsing import ADItem


@dataclass(frozen=True)
class GeneratedADItem:
    """A selected timestamp in seconds and its generated description."""

    timestamp: float
    text: str


BERTScoreScorer = Callable[
    ...,
    tuple[tuple[Iterable[float], Iterable[float], Iterable[float]], str],
]


def compute_bertscore(
    generated_items: Sequence[GeneratedADItem],
    ground_truth_items: Sequence[ADItem],
    matches: Sequence[TemporalMatch],
    *,
    lang: str = "en",
    model_type: str | None = None,
    scorer: BERTScoreScorer | None = None,
) -> dict[str, Any]:
    """Score only supplied temporal matches; never perform text-based matching.

    Match indices refer to the original item sequences used for temporal
    matching, not sorted copies. Invalid indices, reused entries, mismatched
    timestamps, and missing matched text raise ValueError before scoring.

    By default, lazily call official bert_score.score with raw scores (no IDF
    weighting or baseline rescaling) and its language-specific default model.
    The configuration hash records the actual model/layers and library settings.
    An injected scorer must accept the same keywords and return
    ((precision, recall, f1), configuration_hash), as with return_hash=True.
    Its model_type metadata is the requested model; defaults are delegated to it.

    Empty matches require no dependency/model loading and return None means.
    Only text is scored; temporal recall never scales these scores.
    """
    result: dict[str, Any] = {
        "num_scored_pairs": 0,
        "precision_mean": None,
        "recall_mean": None,
        "f1_mean": None,
        "pairs": [],
        "lang": lang,
        "model_type": model_type,
        "configuration_hash": None,
        "idf": False,
        "rescale_with_baseline": False,
        "use_fast_tokenizer": False,
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

    if scorer is None:
        from bert_score import score
        from bert_score.utils import lang2model

        scorer = score
        if model_type is None:
            model_type = lang2model[lang.lower()]

    scores, configuration_hash = scorer(
        candidates,
        references,
        lang=lang,
        model_type=model_type,
        return_hash=True,
        idf=False,
        rescale_with_baseline=False,
        use_fast_tokenizer=False,
    )
    precision, recall, f1 = ([float(value) for value in column] for column in scores)
    count = len(matches)
    for label, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        if len(values) != count:
            raise ValueError(f"BERTScore {label} must contain one value per matched pair")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"BERTScore {label} contains non-finite values")

    result.update(
        num_scored_pairs=count,
        precision_mean=math.fsum(precision) / count,
        recall_mean=math.fsum(recall) / count,
        f1_mean=math.fsum(f1) / count,
        pairs=[
            {
                "prediction_index": match.prediction_index,
                "ground_truth_index": match.ground_truth_index,
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
            }
            for i, match in enumerate(matches)
        ],
        model_type=model_type,
        configuration_hash=configuration_hash,
    )
    return result
