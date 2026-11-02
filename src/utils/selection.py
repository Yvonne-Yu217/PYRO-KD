"""Shared selection budget and the uniform (U) baseline selector."""

import math

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
