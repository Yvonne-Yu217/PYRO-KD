"""Audio-only selector (A): non-speech gaps ranked by length and acoustics.

Candidates are the VAD gaps from audio_features.py. Each gap scores

    score = w_duration * dur_n - w_energy * rms_n - w_onset * onset_n

where each term is divided by its maximum over the video's gaps. Long, quiet
gaps win by default; a negative w_onset flips onset into a bonus (the sign is
a dev-set question, since a sudden sound may be one to protect or one that
marks a visual event).

Candidates form a chronological earliest-feasible packing of half-open usable
gaps [start, end): start at the first available time, then at least 2 s later,
carrying separation across gap boundaries. Long gaps provide multiple moments.
This packing has maximum cardinality: moving an earlier feasible point later
cannot create room for more subsequent points. All candidates are compatible,
so ranking them by the existing gap score (ties: earlier time) returns K whenever
the usable non-speech intervals can hold K. No time rounding or speech padding
is used. Cached gaps are also clipped against detected speech before selection.
"""

from dataclasses import dataclass
import math
from typing import Sequence

from src.utils.audio_features import Gap, gaps_of, merge_segments, speech_of
from src.utils.selection import MIN_SEPARATION_SECONDS, select_top_k


@dataclass(frozen=True)
class AudioCandidate:
    time: float
    gap_start: float
    gap_end: float
    score: float


class AudioSelector:
    def __init__(self, duration_weight: float = 1.0, energy_weight: float = 0.5, onset_weight: float = 0.0):
        self.duration_weight = duration_weight
        self.energy_weight = energy_weight
        self.onset_weight = onset_weight

    def candidates(
        self, gaps: Sequence[Gap], min_separation: float = MIN_SEPARATION_SECONDS,
    ) -> list[AudioCandidate]:
        """Maximum-capacity compatible moments; each inherits its gap's score."""
        if not math.isfinite(min_separation) or min_separation <= 0:
            raise ValueError("min_separation must be positive and finite")
        if any(not math.isfinite(g.start) or not math.isfinite(g.end) or g.start < 0 for g in gaps):
            raise ValueError("gap bounds must be finite nonnegative seconds")
        gaps = sorted((g for g in gaps if g.end > g.start), key=lambda g: (g.start, g.end))
        if not gaps:
            return []
        max_duration = max(g.duration for g in gaps) or 1.0
        max_rms = max(g.rms for g in gaps) or 1.0
        max_onset = max(g.onset for g in gaps) or 1.0
        candidates = []
        next_time = 0.0
        for g in gaps:
            score = (
                self.duration_weight * g.duration / max_duration
                - self.energy_weight * g.rms / max_rms
                - self.onset_weight * g.onset / max_onset
            )
            time = max(g.start, next_time)
            while time < g.end:
                candidates.append(AudioCandidate(time, g.start, g.end, score))
                next_time = time + min_separation
                # Preserve the exact >= separation contract despite float rounding.
                if next_time - time < min_separation:
                    next_time = math.nextafter(next_time, math.inf)
                time = next_time
        return candidates

    def select(
        self,
        features: dict,
        k: int,
        min_separation: float = MIN_SEPARATION_SECONDS,
    ) -> list[AudioCandidate]:
        """Return min(K, non-speech capacity) moments in chronological order."""
        # Defensively subtract detected speech, even if a cached gap crosses it.
        speech = merge_segments(speech_of(features))
        usable = []
        for gap in gaps_of(features):
            cursor = gap.start
            for start, end in speech:
                if end <= cursor:
                    continue
                if start >= gap.end:
                    break
                if start > cursor:
                    usable.append(Gap(cursor, start, gap.rms, gap.onset))
                cursor = max(cursor, end)
            if cursor < gap.end:
                usable.append(Gap(cursor, gap.end, gap.rms, gap.onset))
        return select_top_k(
            self.candidates(usable, min_separation), k, min_separation,
            time_of=lambda c: c.time, score_of=lambda c: c.score,
        )
