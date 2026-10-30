"""Exact one-to-one evaluation of selected timestamps in seconds."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class TemporalMatch:
    """One matched pair, with indices referring to the original inputs."""

    prediction_index: int
    ground_truth_index: int
    prediction_time: float
    ground_truth_time: float
    error_seconds: float


def _nonnegative_finite(value: float, name: str) -> float:
    """Validate numeric seconds, rejecting booleans and non-finite values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def match_timestamps(
    predictions: Sequence[float],
    ground_truth: Sequence[float],
    tolerance_seconds: float = 2.0,
) -> list[TemporalMatch]:
    """Maximize match count, then minimize summed absolute error exactly.

    Dynamic programming considers matching or skipping each sorted timestamp.
    It suffices to consider noncrossing matches: uncrossing two pairs on the
    real line preserves eligibility under a common tolerance and cannot
    increase absolute error. Thus the ordered DP finds a global optimum.

    Time and memory are O(len(predictions) * len(ground_truth)). Equal-score
    ties prefer matching, then skipping a prediction, then skipping GT.
    Sorting ties use original indices; results are in original prediction
    order. Duplicate timestamps remain separate events and are not deduplicated.
    Times and tolerance must be finite, nonnegative real numbers. Eligibility
    uses the inclusive comparison abs(prediction - GT) <= tolerance_seconds.
    """
    tolerance = _nonnegative_finite(tolerance_seconds, "tolerance_seconds")
    predicted = sorted(
        (_nonnegative_finite(time, f"predictions[{i}]"), i)
        for i, time in enumerate(predictions)
    )
    reference = sorted(
        (_nonnegative_finite(time, f"ground_truth[{i}]"), i)
        for i, time in enumerate(ground_truth)
    )
    m, n = len(predicted), len(reference)
    if not m or not n:
        return []
    # Scores are (cardinality, negative total error), compared lexicographically.
    scores = [[(0, 0.0) for _ in range(n + 1)] for _ in range(m + 1)]
    actions = [["" for _ in range(n)] for _ in range(m)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            error = abs(predicted[i][0] - reference[j][0])
            candidates = []
            if error <= tolerance:
                count, negative_error = scores[i + 1][j + 1]
                candidates.append(((count + 1, negative_error - error), "match"))
            candidates.append((scores[i + 1][j], "skip_prediction"))
            candidates.append((scores[i][j + 1], "skip_gt"))
            # max returns the first candidate in a tie, making tie breaks stable.
            scores[i][j], actions[i][j] = max(candidates, key=lambda item: item[0])

    matches = []
    i = j = 0
    while i < m and j < n:
        action = actions[i][j]
        if action == "match":
            prediction_time, prediction_index = predicted[i]
            ground_truth_time, ground_truth_index = reference[j]
            matches.append(
                TemporalMatch(
                    prediction_index=prediction_index,
                    ground_truth_index=ground_truth_index,
                    prediction_time=prediction_time,
                    ground_truth_time=ground_truth_time,
                    error_seconds=abs(prediction_time - ground_truth_time),
                )
            )
            i += 1
            j += 1
        elif action == "skip_prediction":
            i += 1
        else:
            j += 1
    return sorted(matches, key=lambda match: match.prediction_index)


def temporal_precision_recall_f1(
    predictions: Sequence[float],
    ground_truth: Sequence[float],
    tolerance_seconds: float = 2.0,
) -> dict[str, Any]:
    """Return temporal counts, scores, mean absolute error, and match records.

    An empty pair of inputs has undefined (None) precision, recall, and F1.
    If only one input is empty, all three scores are zero. Mean error is None
    whenever no pairs match. Match records are dictionaries for JSON output.
    """
    matches = match_timestamps(predictions, ground_truth, tolerance_seconds)
    tp = len(matches)
    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    if len(predictions) == 0 and len(ground_truth) == 0:
        precision = recall = f1 = None
    else:
        precision = tp / len(predictions) if len(predictions) else 0.0
        recall = tp / len(ground_truth) if len(ground_truth) else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {
        "tolerance_seconds": float(tolerance_seconds),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_temporal_error_seconds": (
            math.fsum(match.error_seconds for match in matches) / tp if tp else None
        ),
        "matches": [asdict(match) for match in matches],
    }


def temporal_tolerance_sweep(
    predictions: Sequence[float],
    ground_truth: Sequence[float],
    tolerances: Sequence[float] = (0.0, 1.0, 2.0, 3.0, 5.0),
) -> list[dict[str, Any]]:
    """Evaluate each tolerance independently, preserving order and duplicates.

    The primary operating point is +/-2 seconds; this sweep is supporting
    analysis. Each result includes its own tolerance and optimal matches.
    """
    return [
        temporal_precision_recall_f1(predictions, ground_truth, tolerance)
        for tolerance in tolerances
    ]
