"""Rank feasible placements with memory updated only by selected candidates.

The caller constructs salience candidates without labels and supplies explicit
placement bounds. This greedy nearest-gap method does not optimize alternative
placements jointly and never reduces the requested K to disguise a shortfall.
"""
from dataclasses import dataclass
import math
from typing import Sequence

from src.utils.gap_snapping import SnappedMoment, snap_to_gap


@dataclass(frozen=True)
class Candidate:
    event_time: float
    base_score: float
    local_novelty: float = 0.0


@dataclass(frozen=True)
class Selected:
    candidate_index: int
    rank: int
    placement: SnappedMoment
    score: float
    novelty: float


@dataclass
class Selection:
    requested: int
    selected: list[Selected]
    infeasible_indices: list[int]
    separation_rejected_indices: list[int]
    budget_unselected_indices: list[int]

    @property
    def shortfall(self):
        return self.requested - len(self.selected)


def select_placed(candidates: Sequence[Candidate], gaps, k: int, *, required_words: int,
                  max_advance: float, max_lag: float, novelty_weight: float,
                  embeddings=None, min_separation: float = 2.0,
                  segment_ids=None, segment_quotas=None) -> Selection:
    """Snap first, rank second; embeddings enable running cosine novelty.

    Base scores may be signed (e.g. event suppression). With embeddings, the
    first pick uses local novelty and later picks use (1-max cosine)/2 against
    all selected events. Without embeddings local novelty remains fixed. Memory
    follows selection rank, not chronological playback. Separation applies to
    placement times. Word limits are gap capacity, not reserved AD durations;
    rendered AD-to-AD overlap still needs a downstream check.
    """
    import numpy as np

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError('k must be a positive integer')
    if not math.isfinite(novelty_weight) or novelty_weight < 0:
        raise ValueError('novelty_weight must be finite and nonnegative')
    if not math.isfinite(min_separation) or min_separation < 0:
        raise ValueError('min_separation must be finite and nonnegative')
    if (segment_ids is None) != (segment_quotas is None):
        raise ValueError('segment IDs and quotas must be supplied together')
    if segment_ids is not None:
        if (len(segment_ids) != len(candidates) or not segment_quotas
                or any(isinstance(q,bool) or not isinstance(q,int) or q<0 for q in segment_quotas)
                or sum(segment_quotas)>k
                or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<len(segment_quotas) for i in segment_ids)):
            raise ValueError('invalid segment IDs/quotas for requested budget')
        segment_counts = [0]*len(segment_quotas)
    # Validate bounds/gaps even when candidate list is empty.
    snap_to_gap(0, gaps, required_words, max_advance=max_advance, max_lag=max_lag)
    if any(not all(math.isfinite(v) for v in (c.event_time,c.base_score,c.local_novelty))
           or c.event_time < 0 or c.local_novelty < 0 for c in candidates):
        raise ValueError('candidate values must be finite with nonnegative time/novelty')
    unit = None
    if embeddings is not None:
        vectors = np.asarray(embeddings, dtype=np.float64)
        if (vectors.ndim != 2 or vectors.shape[0] != len(candidates)
                or vectors.shape[1] == 0 or not np.isfinite(vectors).all()):
            raise ValueError('embeddings must be finite and aligned with candidates')
        norms = np.linalg.norm(vectors,axis=1)
        if np.any(norms <= 0):
            raise ValueError('embeddings must be nonzero')
        unit = vectors / norms[:,None]
    placements = [snap_to_gap(c.event_time,gaps,required_words,max_advance=max_advance,max_lag=max_lag)
                  for c in candidates]
    infeasible = [i for i,p in enumerate(placements) if p is None]
    remaining = {i for i,p in enumerate(placements) if p is not None}
    chosen, separated = [], []
    similarity = np.full(len(candidates),-1.0)
    while remaining and len(chosen) < k:
        eligible = (remaining if segment_ids is None else
                    {i for i in remaining if segment_counts[segment_ids[i]] < segment_quotas[segment_ids[i]]})
        if not eligible:
            break
        def novelty(i):
            return (float((1-np.clip(similarity[i],-1,1))/2)
                    if chosen and unit is not None else candidates[i].local_novelty)
        def score(i):
            return candidates[i].base_score + novelty_weight*novelty(i)
        i = min(eligible,key=lambda j:(-score(j),placements[j].time,candidates[j].event_time,j))
        chosen.append(Selected(i,len(chosen),placements[i],score(i),novelty(i)))
        remaining.remove(i)
        if segment_ids is not None:
            segment_counts[segment_ids[i]] += 1
        if unit is not None:
            similarity = np.maximum(similarity,unit @ unit[i])
        rejected = {j for j in remaining if abs(placements[j].time-placements[i].time)<min_separation}
        separated.extend(sorted(rejected))
        remaining -= rejected
    return Selection(k,sorted(chosen,key=lambda s:(s.placement.time,s.candidate_index)),
                     infeasible,sorted(separated),sorted(remaining))
