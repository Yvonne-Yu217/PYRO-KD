"""Audio-only selector (A): non-speech gaps ranked by length and acoustics.

Candidates are the VAD gaps from audio_features.py. Each gap scores

    score = w_duration * dur_n - w_energy * rms_n - w_onset * onset_n

where each term is divided by its maximum over the video's gaps. Long, quiet
gaps win by default; a negative w_onset flips onset into a bonus (the sign is
a dev-set question, since a sudden sound may be one to protect or one that
marks a visual event). The moment is placed at the gap start, where
professional AD usually begins. Top-K uses a minimum separation and never
pads: shortfalls are reported, not filled.
"""

from dataclasses import dataclass
from typing import Sequence

from src.utils.audio_features import Gap, gaps_of
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

    def candidates(self, gaps: Sequence[Gap]) -> list[AudioCandidate]:
        if not gaps:
            return []
        max_duration = max(g.duration for g in gaps) or 1.0
        max_rms = max(g.rms for g in gaps) or 1.0
        max_onset = max(g.onset for g in gaps) or 1.0
        return [
            AudioCandidate(
                time=g.start, gap_start=g.start, gap_end=g.end,
                score=(
                    self.duration_weight * g.duration / max_duration
                    - self.energy_weight * g.rms / max_rms
                    - self.onset_weight * g.onset / max_onset
                ),
            )
            for g in gaps
        ]

    def select(
        self,
        features: dict,
        k: int,
        min_separation: float = MIN_SEPARATION_SECONDS,
    ) -> list[AudioCandidate]:
        """Up to k candidates in time order; len(result) < k means a shortfall."""
        return select_top_k(
            self.candidates(gaps_of(features)), k, min_separation,
            time_of=lambda c: c.time, score_of=lambda c: c.score,
        )
