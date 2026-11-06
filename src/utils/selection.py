"""Shared selection budget and the uniform (U) baseline selector."""

import math
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
MIN_SEPARATION_SECONDS = 2.0

# Median MTG-AD validation AD segments per minute (scripts/prepare_mtg_ad.py
# recomputes it and records it in configs/mtg_ad_splits.json).
DEFAULT_RATE_PER_MINUTE = 3.5


def k_for_duration(duration_seconds: float, rate_per_minute: float = DEFAULT_RATE_PER_MINUTE) -> int:
    """K = max(1, round(rate * minutes)), identical for every selector."""
    if not (math.isfinite(duration_seconds) and duration_seconds > 0):
        raise ValueError("duration_seconds must be positive and finite")
    return max(1, round(rate_per_minute * duration_seconds / 60))


def uniform_timestamps(duration_seconds: float, k: int) -> list[float]:
    """U selector: t_i = i * D / (K + 1) for i = 1..K (no endpoints)."""
    if k < 1:
        raise ValueError("k must be at least 1")
    step = duration_seconds / (k + 1)
    return [round(i * step, 3) for i in range(1, k + 1)]


def select_top_k(
    candidates: Sequence[T],
    k: int,
    min_separation: float = MIN_SEPARATION_SECONDS,
    time_of: Callable[[T], float] = lambda c: c[0],
    score_of: Callable[[T], float] = lambda c: c[1],
) -> list[T]:
    """Greedy top-K by score with a minimum time separation, returned in time order.

    Candidates are taken from highest score down (ties: earlier time first) and
    skipped when within min_separation of one already chosen. If fewer than k
    candidates survive, fewer are returned: the caller reports the shortfall
    (k - len(result)) instead of padding, so every selector is treated alike.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    chosen: list[T] = []
    for candidate in sorted(candidates, key=lambda c: (-score_of(c), time_of(c))):
        if len(chosen) == k:
            break
        if all(abs(time_of(candidate) - time_of(c)) >= min_separation for c in chosen):
            chosen.append(candidate)
    return sorted(chosen, key=time_of)
