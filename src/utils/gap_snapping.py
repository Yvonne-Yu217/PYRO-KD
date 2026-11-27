"""Separate event salience from feasible AD placement (Proposal Revision 1.5).

The caller supplies VAD-derived legal gaps, required whole-word capacity and
explicit advance/lag bounds. This module neither changes salience nor invents
a gap when audio is absent. Bounds must be frozen using development data.
"""
from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class SnappedMoment:
    event_time: float
    time: float
    gap_start: float
    gap_end: float
    word_limit: int

    @property
    def lag_seconds(self) -> float:
        return self.time - self.event_time


def snap_to_gap(
    event_time: float,
    gaps: Sequence[tuple[float, float]],
    required_words: int,
    *,
    max_advance: float,
    max_lag: float,
    words_per_second: float = 3.0,
) -> SnappedMoment | None:
    """Nearest feasible time, with ties favoring placement after the event.

    Gaps are half-open [start, end), already cleared of speech by VAD. To fit
    W words, a start must leave floor((end-start)*rate) >= W. Overlapping gaps
    are merged; no rounding may grant extra capacity. Returns None if no
    placement fits, including an empty/missing-gap track.
    """
    if not math.isfinite(event_time) or event_time < 0:
        raise ValueError('event_time must be finite and nonnegative')
    if isinstance(required_words, bool) or not isinstance(required_words, int) or required_words < 1:
        raise ValueError('required_words must be a positive integer')
    if any(not math.isfinite(v) or v < 0 for v in (max_advance, max_lag)):
        raise ValueError('advance and lag must be finite and nonnegative')
    if not math.isfinite(words_per_second) or words_per_second <= 0:
        raise ValueError('words_per_second must be finite and positive')
    normalized = []
    for start, end in sorted(gaps):
        if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end <= start:
            raise ValueError('gaps must have finite nonnegative ordered bounds')
        if normalized and start <= normalized[-1][1]:
            normalized[-1] = (normalized[-1][0], max(end, normalized[-1][1]))
        else:
            normalized.append((start, end))
    options = []
    for start, end in normalized:
        low = max(start, event_time - max_advance)
        high = min(end - required_words / words_per_second, event_time + max_lag)
        if high < low:
            continue
        time = min(max(event_time, low), high)
        # A computed upper bound can round upward at a whole-word boundary.
        if math.floor((end - time) * words_per_second) < required_words:
            time = math.nextafter(time, -math.inf)
        if time < low or time >= end:
            continue
        budget = math.floor((end - time) * words_per_second)
        if budget < required_words:
            continue
        options.append(SnappedMoment(event_time, time, start, end, budget))
    return min(options, key=lambda m: (abs(m.lag_seconds), m.lag_seconds < 0, m.time), default=None)
