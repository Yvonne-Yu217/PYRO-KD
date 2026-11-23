"""Long-video selection (proposal §5.5; ablation 7: flat vs hierarchical).

1. Chunking: cut the video at boundaries ranked long silence > music onset >
   shot cut, keeping each segment between min and max length (a hard cut at
   max length when no boundary qualifies).
2. Budget: K is split across segments. Each segment first gets a floor of
   floor_fraction x its duration-proportional share (so quiet segments are
   not starved); the rest goes in proportion to event mass (the summed
   salience of cue-driven, non-fill candidates in the segment, i.e. event
   density x length). Allocation is capped by each segment's capacity under
   t_min, with the excess passed on, and always sums to min(K, capacity).
3. Selection: segments are processed in order with the two-stage selector,
   carrying the running context memory (selected times for t_min, and their
   CLIP embeddings for memory novelty / MMR) from one segment to the next.
"""

from __future__ import annotations

import math
from typing import Any

from src.selection.cues import CueBundle
from src.selection.two_stage import (
    Candidate, SelectionResult, SelectionState, generate_candidates, greedy_select, merge_config,
)

DEFAULT_HIERARCHY: dict[str, Any] = {
    "min_segment_seconds": 60.0,
    "max_segment_seconds": 300.0,
    "long_silence_seconds": 8.0,
    "priority": {"long_silence": 3, "music": 2, "shot": 1},
    "floor_fraction": 0.5,
}


def boundaries(bundle: CueBundle, cfg: dict) -> list[tuple[float, int, str]]:
    """(time, priority, kind) cut points: middles of long silences, music onsets, shots."""
    out: list[tuple[float, int, str]] = []
    pr = cfg["priority"]
    for s, e in bundle.gaps or []:
        if e - s >= cfg["long_silence_seconds"]:
            out.append(((s + e) / 2.0, pr["long_silence"], "long_silence"))
    for event in bundle.events:
        if event.get("superclass") == "music":
            out.append((float(event["time"]), pr["music"], "music"))
    out.extend((t, pr["shot"], "shot") for t in bundle.shots)
    return sorted(out)


def segment(duration: float, cuts: list[tuple[float, int, str]], cfg: dict) -> list[tuple[float, float]]:
    """Greedy left-to-right chunking within [min, max] segment length."""
    lo_len, hi_len = cfg["min_segment_seconds"], cfg["max_segment_seconds"]
    if not (0 < lo_len <= hi_len):
        raise ValueError("need 0 < min_segment_seconds <= max_segment_seconds")
    segments: list[tuple[float, float]] = []
    start = 0.0
    target = (lo_len + hi_len) / 2.0
    while duration - start > hi_len:
        options = [(t, p) for t, p, _ in cuts if start + lo_len <= t <= start + hi_len]
        if options:
            cut = max(options, key=lambda o: (o[1], -abs(o[0] - start - target)))[0]
        else:
            cut = start + hi_len
        segments.append((start, cut))
        start = cut
    if segments and duration - start < lo_len:
        segments[-1] = (segments[-1][0], duration)
    else:
        segments.append((start, duration))
    return segments


def capacity(times: list[float], t_min: float) -> int:
    """Most t_min-separated picks among sorted candidate times (earliest-first packing)."""
    count, last = 0, -math.inf
    for t in sorted(times):
        if t - last >= t_min:
            count, last = count + 1, t
    return count


def allocate(
    k: int, lengths: list[float], masses: list[float], floor_fraction: float = 0.5,
    capacities: list[int] | None = None,
) -> list[int]:
    """Split K: floor share by length, the rest by event mass (largest remainder),
    capped by capacity with the excess redistributed. Sums to min(K, sum(capacity))."""
    if not 0 <= floor_fraction <= 1:
        raise ValueError("floor_fraction must be in [0, 1]")
    n = len(lengths)
    if n == 0:
        return []
    caps = list(capacities) if capacities is not None else [k] * n
    total_len = sum(lengths) or 1.0
    alloc = [min(caps[i], math.floor(floor_fraction * k * lengths[i] / total_len)) for i in range(n)]
    target = min(k, sum(caps))
    weights = [m if m > 0 else 0.0 for m in masses]
    if sum(weights) == 0:
        weights = list(lengths)
    while sum(alloc) < target:
        open_ = [i for i in range(n) if alloc[i] < caps[i]]
        remaining = target - sum(alloc)
        w_total = sum(weights[i] for i in open_) or float(len(open_))
        quotas = {i: remaining * ((weights[i] / w_total) if sum(weights[j] for j in open_) else 1 / len(open_)) for i in open_}
        add = {i: min(caps[i] - alloc[i], math.floor(q)) for i, q in quotas.items()}
        for i, v in add.items():
            alloc[i] += v
        left = target - sum(alloc)
        if left <= 0:
            break
        order = sorted(
            (i for i in open_ if alloc[i] < caps[i]),
            key=lambda i: (-(quotas[i] - math.floor(quotas[i])), -weights[i], i),
        )
        if not order:
            break
        for i in order[:left]:
            alloc[i] += 1
    return alloc


def select_hierarchical(
    bundle: CueBundle, k: int, config: dict | None = None, hierarchy: dict | None = None,
) -> SelectionResult:
    cfg = merge_config(config)
    h = {**DEFAULT_HIERARCHY, **(hierarchy or {})}
    if "priority" in (hierarchy or {}):
        h["priority"] = {**DEFAULT_HIERARCHY["priority"], **hierarchy["priority"]}
    candidates, diagnostics = generate_candidates(bundle, cfg)
    if diagnostics.get("status") == "no_vad":
        return SelectionResult([], [], "missing_vad", diagnostics)
    segments = segment(bundle.duration, boundaries(bundle, h), h)

    def inside(c: Candidate, seg: tuple[float, float], last: bool) -> bool:
        return seg[0] <= c.time < seg[1] or (last and c.time == seg[1])

    per_segment = [
        [c for c in candidates if inside(c, seg, i == len(segments) - 1)] for i, seg in enumerate(segments)
    ]
    masses = [sum(max(0.0, c.salience) for c in cs if c.source != "fill") for cs in per_segment]
    caps = [capacity([c.time for c in cs], cfg["min_separation"]) for cs in per_segment]
    budget = allocate(k, [e - s for s, e in segments], masses, h["floor_fraction"], caps)

    state = SelectionState()
    chosen: list[Candidate] = []
    for seg, cs, k_i in zip(segments, per_segment, budget):
        if k_i <= 0:
            continue
        picked = greedy_select(cs, k_i, bundle, cfg, state, bounds=seg)
        for c in picked:
            state.times.append(c.time)
            vector = bundle.embedding_near(c.time)
            if vector is not None:
                state.memory.append(vector)
        chosen.extend(picked)
    chosen.sort(key=lambda c: c.time)
    diagnostics.update(
        n_candidates=len(candidates), shortfall=k - len(chosen),
        segments=[{"start": s, "end": e, "k": b, "event_mass": round(m, 4), "capacity": cap}
                  for (s, e), b, m, cap in zip(segments, budget, masses, caps)],
    )
    return SelectionResult([c.time for c in chosen], [c.to_dict() for c in chosen], "ok", diagnostics)
