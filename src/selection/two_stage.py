"""Two-stage cue-driven selector (proposal §5.1).

    Score = a*VN(t) + b*NSE(t) + g*SC(t) + e*AVD(t) + d*GF(p) - l*OP(p)

1. Candidate generation: peaks of each enabled salience stream (VN, NSE, SC)
   give event times t, tagged with their source. With fill="grid", every
   t_min-spaced grid point is also a low-salience candidate so that a fixed K
   can still be met where cues are sparse (recorded as source "fill").
2. Gap snapping (placement="snap"): each candidate gets a placement p inside a
   usable VAD gap long enough for min_words. If t is in such a gap, p = t;
   otherwise the nearest feasible point at most max_lag after t (or max_lead
   before it) is used; candidates with no feasible point are dropped and
   counted. placement="event" keeps p = t (ablation 4) and lets OP penalize
   speech collisions.
3. Selection: greedy top-K under the hard t_min separation, optionally with
   MMR (score - lambda * max similarity to the selected set), where similarity
   is CLIP cosine when embeddings exist and a temporal kernel otherwise.

VN can be "local" (change to the previous frame, as in V) or "memory": a
blend (1 - b) * local + b * memory novelty, where memory novelty is
1 - max cosine to the moments already selected (the running context memory of
§3.3/§5.5; ablation 8), normalized over the remaining candidates, and 1 for
everything while the memory is empty. b = 1 is the proposal's pure definition;
it lets an undescribed static frame outrank a shot cut, so the default keeps
half of the local change signal (untuned, like every weight here). Ties in score go to the candidate farthest from
everything selected (and from the clip ends), then earlier time, so equal-score
candidates spread over the video instead of packing at its start.

Salience terms are read at the event time t, feasibility terms at the
placement p. NSE and SC are weighted averages of their components with signed
weights allowed: weights["nse"] < 0 is the global *suppress* hypothesis
(H2b), and signed per-superclass weights in nse_components test the
class-dependent answer (RQ2). AVD = VN * (1 - audio change) is the
audio-visual disagreement feature.

All default weights are untuned placeholders; tune them on the dev split only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

from src.generation.prompts import WORDS_PER_SECOND, word_limit_for_gap
from src.selection.cues import CueBundle
from src.utils.audio_features import available_gap
from src.utils.selection import MIN_SEPARATION_SECONDS

DEFAULT_CONFIG: dict[str, Any] = {
    "weights": {"vn": 1.0, "nse": 0.5, "sc": 0.5, "av_disagreement": 0.0, "gf": 0.5, "op": 1.0},
    "nse_components": {
        "music": 1.0, "impulsive": 1.0, "vocal_nonspeech": 1.0, "ambient": 0.5, "other": 0.5,
        "clap_change": 1.0, "loudness": 0.5, "onset": 0.5,
    },
    "sc_components": {"deictic": 1.0, "semantic_shift": 1.0, "speaker_change": 1.0, "long_silence": 0.5},
    "candidate_streams": ["vn", "nse", "sc"],
    "peak_threshold": 0.2,
    "fill": "grid",              # grid | none
    "placement": "snap",         # snap | event
    "max_lag_seconds": 3.0,
    "max_lead_seconds": 0.5,
    "min_words": 3,
    "gf_target_words": 12,
    "tts_words_per_second": 2.5,  # base TTS rate; rendered at tts_speed (1.2x -> 3 words/s)
    "tts_speed": 1.2,
    "novelty_mode": "local",      # local | memory
    "memory_blend": 0.5,          # memory mode: VN = (1 - b) * local + b * memory novelty; b = 1 is pure memory
    "diversity": "tmin",          # tmin | mmr
    "mmr_lambda": 0.3,
    "min_separation": MIN_SEPARATION_SECONDS,
    "salience_window_seconds": 0.5,
    "require_vad": True,
}


def merge_config(overrides: dict | None) -> dict:
    cfg = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
           for k, v in DEFAULT_CONFIG.items()}
    for key, value in (overrides or {}).items():
        if key not in cfg:
            raise KeyError(f"unknown two-stage option {key!r}")
        if isinstance(value, dict) and isinstance(cfg[key], dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    if cfg["placement"] not in ("snap", "event"):
        raise ValueError("placement must be 'snap' or 'event'")
    if cfg["novelty_mode"] not in ("local", "memory"):
        raise ValueError("novelty_mode must be 'local' or 'memory'")
    if cfg["diversity"] not in ("tmin", "mmr"):
        raise ValueError("diversity must be 'tmin' or 'mmr'")
    if cfg["fill"] not in ("grid", "none"):
        raise ValueError("fill must be 'grid' or 'none'")
    return cfg


@dataclass
class Candidate:
    event_time: float
    source: str
    time: float = 0.0  # placement p
    vn: float = 0.0
    nse: float = 0.0
    sc: float = 0.0
    avd: float = 0.0
    audio_change: float = 0.0  # audio change at the bin where vn was read (what avd was computed from)
    gf: float = 0.0
    op: float = 0.0
    salience: float = 0.0  # a*VN + b*NSE + g*SC + e*AVD (with local VN)
    score: float = 0.0
    components: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lag"] = round(self.time - self.event_time, 3)
        return d


@dataclass
class SelectionResult:
    times: list[float]
    selected: list[dict]
    status: str = "ok"
    diagnostics: dict = field(default_factory=dict)


def _weighted(components: dict[str, list[float]], weights: dict[str, float], i: int) -> tuple[float, dict]:
    """sum_c w_c * x_c(t) / max_c |w_c|, clipped to [-1, 1], over available components.

    Dividing by the largest weight (not the weight sum) keeps one event's
    contribution independent of how many other streams happen to be cached.
    """
    total, parts = 0.0, {}
    scale = max((abs(w) for w in weights.values()), default=0.0)
    for name, w in weights.items():
        track = components.get(name)
        if track is None or w == 0:
            continue
        value = track[i]
        parts[name] = value
        total += w * value
    if scale == 0:
        return 0.0, parts
    return max(-1.0, min(1.0, total / scale)), parts


def _peaks(track: list[float], threshold: float, min_distance_bins: int) -> list[int]:
    """Local maxima >= threshold with non-maximum suppression."""
    raw = [
        i for i, v in enumerate(track)
        if v >= threshold and (i == 0 or v >= track[i - 1]) and (i == len(track) - 1 or v > track[i + 1])
    ]
    kept: list[int] = []
    for i in sorted(raw, key=lambda j: (-track[j], j)):
        if all(abs(i - j) >= min_distance_bins for j in kept):
            kept.append(i)
    return sorted(kept)


def feasible_intervals(bundle: CueBundle, min_words: int) -> list[tuple[float, float]]:
    """[s, e - min_words/3] inside each usable gap: a placement there leaves room for
    at least min_words under the shared 3-words-per-second cap."""
    need = min_words / WORDS_PER_SECOND
    return [(s, e - need) for s, e in (bundle.gaps or []) if e - need >= s]


def snap(t: float, intervals: list[tuple[float, float]], max_lag: float, max_lead: float) -> float | None:
    """Nearest feasible placement for event time t within the lag/lead allowance."""
    best = None
    for s, e in intervals:
        if s <= t <= e:
            return t
        if s > t and s - t <= max_lag:
            option = s
        elif e < t and t - e <= max_lead:
            option = e
        else:
            continue
        if best is None or abs(option - t) < abs(best - t) or (abs(option - t) == abs(best - t) and option > best):
            best = option
    return best


def overlap_fraction(p: float, speech: list[tuple[float, float]] | None, seconds: float) -> float:
    """Fraction of [p, p + seconds] that collides with speech."""
    if not speech or seconds <= 0:
        return 0.0
    end = p + seconds
    collided = sum(max(0.0, min(end, e) - max(p, s)) for s, e in speech)
    return min(1.0, collided / seconds)


def generate_candidates(bundle: CueBundle, cfg: dict) -> tuple[list[Candidate], dict]:
    """Stage 1 + 2: candidates with placements and per-term values."""
    weights = cfg["weights"]
    nse_w, sc_w = cfg["nse_components"], cfg["sc_components"]
    n = len(bundle.grid)
    nse_track = [_weighted(bundle.nse, nse_w, i)[0] for i in range(n)] if bundle.nse else None
    nse_unsigned = [max((bundle.nse[c][i] for c in nse_w if c in bundle.nse and nse_w[c] != 0), default=0.0)
                    for i in range(n)] if bundle.nse else None
    sc_track = [_weighted(bundle.sc, sc_w, i)[0] for i in range(n)] if bundle.sc else None
    audio_change = bundle.audio_change()
    min_bins = max(1, int(round(cfg["min_separation"] / bundle.hop)))

    events: dict[float, str] = {}
    streams = {"vn": bundle.vn, "nse": nse_unsigned, "sc": sc_track}
    for name in cfg["candidate_streams"]:
        track = streams.get(name)
        if track is None or weights.get(name, 0) == 0:
            continue
        for i in _peaks(track, cfg["peak_threshold"], min_bins):
            events.setdefault(bundle.grid[i], name)
    if cfg["fill"] == "grid":
        step = max(1, min_bins)
        for i in range(0, n, step):
            events.setdefault(bundle.grid[i], "fill")

    diagnostics = {"by_source": {}, "dropped_infeasible": 0, "streams_missing": dict(bundle.missing)}
    have_vad = bundle.gaps is not None
    if cfg["placement"] == "snap" and not have_vad and cfg["require_vad"]:
        diagnostics["status"] = "no_vad"
        return [], diagnostics
    intervals = feasible_intervals(bundle, cfg["min_words"]) if have_vad else []
    render_seconds = cfg["min_words"] / (cfg["tts_words_per_second"] * cfg["tts_speed"])
    window = cfg["salience_window_seconds"]

    candidates: dict[float, Candidate] = {}
    for t, source in sorted(events.items()):
        if cfg["placement"] == "snap" and have_vad:
            p = snap(t, intervals, cfg["max_lag_seconds"], cfg["max_lead_seconds"])
            if p is None:
                diagnostics["dropped_infeasible"] += 1
                continue
        else:
            p = t
        p = round(p, 3)
        c = Candidate(event_time=t, source=source, time=p)
        i_lo, i_hi = bundle.index(t - window), bundle.index(t + window)
        best_i = i_lo
        if bundle.vn is not None:
            best_i = max(range(i_lo, i_hi + 1), key=lambda j: bundle.vn[j])
            c.vn = bundle.vn[best_i]
        if nse_track is not None:
            j = max(range(i_lo, i_hi + 1), key=lambda j: abs(nse_track[j]))
            c.nse, parts = _weighted(bundle.nse, nse_w, j)
            c.components["nse"] = parts
        if sc_track is not None:
            j = max(range(i_lo, i_hi + 1), key=lambda j: abs(sc_track[j]))
            c.sc, parts = _weighted(bundle.sc, sc_w, j)
            c.components["sc"] = parts
        c.audio_change = audio_change[best_i]
        c.avd = c.vn * (1.0 - c.audio_change)
        if have_vad:
            gap = available_gap(p, bundle.speech or [], bundle.duration)
            c.gf = min(1.0, word_limit_for_gap(gap) / cfg["gf_target_words"])
            c.op = overlap_fraction(p, bundle.speech, render_seconds)
        c.salience = (
            weights.get("vn", 0) * c.vn + weights.get("nse", 0) * c.nse
            + weights.get("sc", 0) * c.sc + weights.get("av_disagreement", 0) * c.avd
        )
        c.score = c.salience + weights.get("gf", 0) * c.gf - weights.get("op", 0) * c.op
        previous = candidates.get(p)
        if previous is None or c.score > previous.score:
            candidates[p] = c
        diagnostics["by_source"][source] = diagnostics["by_source"].get(source, 0) + 1
    return sorted(candidates.values(), key=lambda c: c.time), diagnostics


@dataclass
class SelectionState:
    """Already-selected moments (times and embeddings) carried across segments."""

    times: list[float] = field(default_factory=list)
    memory: list = field(default_factory=list)


def greedy_select(
    candidates: list[Candidate],
    k: int,
    bundle: CueBundle,
    cfg: dict,
    state: SelectionState | None = None,
    bounds: tuple[float, float] | None = None,
) -> list[Candidate]:
    """Stage 3: greedy top-k with hard t_min, optional MMR and running-memory novelty."""
    import numpy as np

    if k < 1 or not candidates:
        return []
    state = state or SelectionState()
    t_min = cfg["min_separation"]
    lo, hi = bounds or (0.0, bundle.duration)
    weights = cfg["weights"]
    a = weights.get("vn", 0.0)
    e_w = weights.get("av_disagreement", 0.0)
    use_memory = cfg["novelty_mode"] == "memory" and bundle.embeddings is not None
    use_mmr = cfg["diversity"] == "mmr"
    blend = float(cfg["memory_blend"])
    if not 0.0 <= blend <= 1.0:
        raise ValueError("memory_blend must be in [0, 1]")

    emb = None
    if bundle.embeddings is not None and (use_memory or use_mmr):
        emb = np.asarray([bundle.embedding_near(c.time) for c in candidates], dtype=np.float32)
    max_sim = np.full(len(candidates), -np.inf if emb is not None else 0.0)
    if emb is not None and state.memory:
        mem = np.asarray(state.memory, dtype=np.float32)
        max_sim = (emb @ mem.T).max(axis=1)
    tau = max(t_min, (hi - lo) / (k + 1))
    import bisect

    times = [c.time for c in candidates]  # candidates are sorted by placement time
    blocked = [False] * len(candidates)
    nearest = [math.inf] * len(candidates)  # distance to the nearest taken time

    def take(t: float) -> None:
        a_i = bisect.bisect_left(times, t - t_min + 1e-9)
        b_i = bisect.bisect_right(times, t + t_min - 1e-9)
        for j in range(a_i, b_i):
            blocked[j] = True
        for j in range(len(times)):
            d = abs(times[j] - t)
            if d < nearest[j]:
                nearest[j] = d

    for t in state.times:
        take(t)
    temporal_sim = [0.0] * len(candidates)
    if use_mmr and emb is None:
        for t in state.times:
            for j, x in enumerate(times):
                temporal_sim[j] = max(temporal_sim[j], math.exp(-abs(x - t) / tau))

    chosen: list[Candidate] = []
    for _ in range(k):
        eligible = [i for i in range(len(candidates)) if not blocked[i]]
        if not eligible:
            break
        any_taken = bool(state.times or chosen)
        memory_vn = None
        if use_memory:
            if state.memory or chosen:
                raw = {i: max(0.0, 1.0 - float(max_sim[i])) for i in eligible}
                peak = max(raw.values()) or 1.0
                memory_vn = {i: raw[i] / peak for i in eligible}
            else:
                memory_vn = {i: 1.0 for i in eligible}

        def key(i):
            c = candidates[i]
            score = c.score
            if memory_vn is not None:
                vn = (1.0 - blend) * c.vn + blend * memory_vn[i]
                avd = vn * (1.0 - c.audio_change)
                score += a * (vn - c.vn) + e_w * (avd - c.avd)
            if use_mmr and any_taken:
                similarity = float(max_sim[i]) if emb is not None and np.isfinite(max_sim[i]) else temporal_sim[i]
                score -= cfg["mmr_lambda"] * similarity
            spread = min(nearest[i], c.time - lo, hi - c.time)
            return (round(score, 9), round(spread, 6), -c.time)

        best = max(eligible, key=key)
        pick = candidates[best]
        chosen.append(pick)
        take(pick.time)
        if emb is not None:
            max_sim = np.maximum(max_sim, emb @ emb[best])
        elif use_mmr:
            for j, x in enumerate(times):
                temporal_sim[j] = max(temporal_sim[j], math.exp(-abs(x - pick.time) / tau))
    return sorted(chosen, key=lambda c: c.time)


def select(bundle: CueBundle, k: int, config: dict | None = None) -> SelectionResult:
    """Flat two-stage selection of up to K moments over the whole video."""
    cfg = merge_config(config)
    candidates, diagnostics = generate_candidates(bundle, cfg)
    if diagnostics.get("status") == "no_vad":
        return SelectionResult([], [], "missing_vad", diagnostics)
    chosen = greedy_select(candidates, k, bundle, cfg)
    diagnostics.update(n_candidates=len(candidates), shortfall=k - len(chosen))
    return SelectionResult([c.time for c in chosen], [c.to_dict() for c in chosen], "ok", diagnostics)
