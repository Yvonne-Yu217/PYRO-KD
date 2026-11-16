"""Bootstrap confidence intervals over videos (standard library only).

The unit of resampling is the video: excerpts and moments within a video are
not independent, so every statistic here is computed on per-video values.
"""

import math
import random
from typing import Sequence

DEFAULT_RESAMPLES = 10_000


def _clean(values: Sequence[float | None]) -> list[float]:
    cleaned = [float(v) for v in values if v is not None]
    if any(not math.isfinite(v) for v in cleaned):
        raise ValueError("values must be finite")
    return cleaned


def _interval(samples: list[float], alpha: float) -> tuple[float, float]:
    samples.sort()
    low = samples[int((alpha / 2) * (len(samples) - 1))]
    high = samples[int((1 - alpha / 2) * (len(samples) - 1))]
    return low, high


def bootstrap_mean_ci(
    values: Sequence[float | None],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int | None]:
    """Mean and percentile bootstrap (1 - alpha) interval; None entries are dropped."""
    data = _clean(values)
    if not data:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None}
    mean = math.fsum(data) / len(data)
    if len(data) == 1 or resamples < 1:
        return {"n": len(data), "mean": mean, "ci_low": mean, "ci_high": mean}
    rng = random.Random(seed)
    n = len(data)
    means = [math.fsum(data[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    low, high = _interval(means, alpha)
    return {"n": n, "mean": mean, "ci_low": low, "ci_high": high}


def paired_bootstrap_diff(
    a: Sequence[float | None],
    b: Sequence[float | None],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float | int | None]:
    """Mean of a - b over videos with a paired bootstrap interval.

    a and b are aligned per video; pairs where either side is None are dropped.
    The same resampled video indices are used for both, which is what makes the
    interval paired.
    """
    if len(a) != len(b):
        raise ValueError("a and b must be aligned per video")
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if x is not None and y is not None]
    if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in pairs):
        raise ValueError("values must be finite")
    if not pairs:
        return {"n": 0, "mean_diff": None, "ci_low": None, "ci_high": None}
    diffs = [x - y for x, y in pairs]
    mean = math.fsum(diffs) / len(diffs)
    if len(diffs) == 1 or resamples < 1:
        return {"n": len(diffs), "mean_diff": mean, "ci_low": mean, "ci_high": mean}
    rng = random.Random(seed)
    n = len(diffs)
    means = [math.fsum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)]
    low, high = _interval(means, alpha)
    return {"n": n, "mean_diff": mean, "ci_low": low, "ci_high": high}
