"""Non-speech event salience (NSE, proposal §3.1) from cached tracks.

Inputs are tracks already computed by a tagger, CLAP, or librosa; everything
here is deterministic numpy so it is unit-testable without models:

- class onsets: sharp rises in a class posterior (music starts, applause begins);
- embedding change: CLAP embedding distance to a running window of the past,
  which catches transitions with no AudioSet label (genre or tempo shifts);
- loudness jumps: RMS level (dB) rise against the preceding window;
- onset strength: librosa's spectral-flux onset envelope, pooled per hop.

All tracks are on their own time axes; resample_to_grid puts them on the
selector's grid. Salience says *what* might be worth describing; it never
says where a description can go (that is VAD feasibility).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from src.audio.taxonomy import EXCLUDED, superclass_of

DEFAULT_NSE_CONFIG = {
    "onset_rise": 0.2,          # posterior rise over the preceding window
    "onset_min_prob": 0.3,      # posterior must also reach this level
    "onset_window_seconds": 3.0,
    "onset_min_separation": 2.0,  # per label
    "clap_window_seconds": 5.0,
    "loudness_window_seconds": 3.0,
    "event_spread_seconds": 1.0,  # half-width of the triangular kernel on the grid
}


@dataclass(frozen=True)
class AudioEvent:
    """One detected non-speech event (an onset), in seconds."""

    time: float
    label: str
    superclass: str
    strength: float
    source: str = "tagger"

    def to_dict(self) -> dict:
        return asdict(self)


def _np():
    import numpy as np

    return np


def _hop(times: Sequence[float]) -> float:
    if len(times) < 2:
        return 1.0
    return float(times[1] - times[0])


def class_onsets(
    times: Sequence[float],
    posteriors,
    labels: Sequence[str],
    config: dict | None = None,
    overrides: dict[str, str] | None = None,
) -> list[AudioEvent]:
    """Onsets: frames where a non-speech class rises sharply above its recent past.

    rise(t) = p(t) - mean(p over the preceding window). An onset needs
    rise >= onset_rise and p >= onset_min_prob, must be a local maximum of the
    rise, and is suppressed within onset_min_separation of a stronger onset of
    the same label. Speech and silence labels are skipped.
    """
    np = _np()
    cfg = {**DEFAULT_NSE_CONFIG, **(config or {})}
    p = np.asarray(posteriors, dtype=np.float64)
    if p.size == 0:
        return []
    if p.ndim != 2 or p.shape[0] != len(times) or p.shape[1] != len(labels):
        raise ValueError(f"posteriors must be (len(times), len(labels)); got {p.shape}")
    hop = _hop(times)
    window = max(1, int(round(cfg["onset_window_seconds"] / hop)))
    events: list[AudioEvent] = []
    for c, label in enumerate(labels):
        superclass = superclass_of(label, overrides)
        if superclass in EXCLUDED:
            continue
        column = p[:, c]
        rise = np.zeros_like(column)
        for t in range(len(column)):
            past = column[max(0, t - window):t]
            baseline = past.mean() if len(past) else 0.0
            rise[t] = column[t] - baseline
        candidates = [
            t for t in range(len(column))
            if rise[t] >= cfg["onset_rise"] and column[t] >= cfg["onset_min_prob"]
            and (t == 0 or rise[t] >= rise[t - 1])
            and (t == len(column) - 1 or rise[t] > rise[t + 1])
        ]
        kept: list[int] = []
        for t in sorted(candidates, key=lambda i: (-rise[i], i)):
            if all(abs(times[t] - times[j]) >= cfg["onset_min_separation"] for j in kept):
                kept.append(t)
        events.extend(
            AudioEvent(float(times[t]), label, superclass, float(rise[t])) for t in kept
        )
    return sorted(events, key=lambda e: (e.time, e.label))


def embedding_change(times: Sequence[float], embeddings, window_seconds: float = 5.0) -> list[float]:
    """1 - cosine(e_t, mean of the unit embeddings in the preceding window); first frame 0."""
    np = _np()
    e = np.asarray(embeddings, dtype=np.float64)
    if e.size == 0:
        return []
    if e.ndim != 2 or len(e) != len(times):
        raise ValueError("embeddings must be (len(times), dims)")
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    unit = e / np.where(norms == 0.0, 1.0, norms)
    window = max(1, int(round(window_seconds / _hop(times))))
    change = [0.0]
    for t in range(1, len(unit)):
        reference = unit[max(0, t - window):t].mean(axis=0)
        norm = np.linalg.norm(reference)
        if norm == 0.0:
            change.append(0.0)
            continue
        change.append(float(max(0.0, 1.0 - float(unit[t] @ reference) / norm)))
    return change


def loudness_db(waveform, sample_rate: int, hop_seconds: float = 0.5) -> tuple[list[float], list[float]]:
    """(times, RMS level in dBFS) per non-overlapping hop; silence floors at -100 dB."""
    np = _np()
    x = np.asarray(waveform, dtype=np.float64)
    hop = max(1, int(round(hop_seconds * sample_rate)))
    times, levels = [], []
    for i, start in enumerate(range(0, len(x), hop)):
        chunk = x[start:start + hop]
        rms = math.sqrt(float(np.mean(chunk ** 2))) if len(chunk) else 0.0
        times.append(round(i * hop / sample_rate, 6))
        levels.append(20.0 * math.log10(rms) if rms > 1e-5 else -100.0)
    return times, levels


def loudness_jumps(times: Sequence[float], levels_db: Sequence[float], window_seconds: float = 3.0) -> list[float]:
    """Positive dB rise of each hop over the mean of the preceding window (0 when quieter)."""
    if not levels_db:
        return []
    window = max(1, int(round(window_seconds / _hop(times))))
    out = [0.0]
    for t in range(1, len(levels_db)):
        past = levels_db[max(0, t - window):t]
        out.append(max(0.0, float(levels_db[t]) - sum(past) / len(past)))
    return out


def onset_strength_track(waveform, sample_rate: int, hop_seconds: float = 0.5) -> tuple[list[float], list[float]]:
    """librosa onset strength, max-pooled into hop_seconds bins: (times, values)."""
    import librosa

    np = _np()
    hop_length = 512
    envelope = librosa.onset.onset_strength(y=np.asarray(waveform, dtype=np.float32), sr=sample_rate, hop_length=hop_length)
    frame_times = librosa.frames_to_time(np.arange(len(envelope)), sr=sample_rate, hop_length=hop_length)
    duration = len(waveform) / sample_rate
    grid = [round(i * hop_seconds, 6) for i in range(int(math.ceil(duration / hop_seconds)) or 1)]
    return grid, resample_to_grid(frame_times.tolist(), envelope.tolist(), grid, reduce="max")


def resample_to_grid(
    times: Sequence[float], values: Sequence[float], grid: Sequence[float], reduce: str = "max",
) -> list[float]:
    """Pool values into grid bins [g_i - h/2, g_i + h/2); empty bins take the nearest value.

    reduce is "max" (keeps short peaks) or "mean". Returns zeros when there are no values.
    """
    if reduce not in ("max", "mean"):
        raise ValueError("reduce must be 'max' or 'mean'")
    if not grid:
        return []
    if not len(values):
        return [0.0] * len(grid)
    if len(times) != len(values):
        raise ValueError("times and values differ in length")
    pairs = sorted(zip((float(t) for t in times), (float(v) for v in values)))
    half = _hop(grid) / 2.0
    out = []
    import bisect

    keys = [t for t, _ in pairs]
    for g in grid:
        lo = bisect.bisect_left(keys, g - half)
        hi = bisect.bisect_left(keys, g + half)
        chunk = [v for _, v in pairs[lo:hi]]
        if chunk:
            out.append(max(chunk) if reduce == "max" else sum(chunk) / len(chunk))
        else:
            i = bisect.bisect_left(keys, g)
            near = [j for j in (i - 1, i) if 0 <= j < len(pairs)]
            out.append(pairs[min(near, key=lambda j: abs(keys[j] - g))][1])
    return out


def events_to_track(
    events: Sequence[AudioEvent | dict],
    grid: Sequence[float],
    spread_seconds: float = DEFAULT_NSE_CONFIG["event_spread_seconds"],
    superclass: str | None = None,
) -> list[float]:
    """Triangular bumps of height = strength at each event (max where they overlap).

    With superclass given, only that superclass's events contribute.
    """
    import bisect

    track = [0.0] * len(grid)
    if spread_seconds <= 0:
        raise ValueError("spread_seconds must be positive")
    for event in events:
        e = event if isinstance(event, dict) else event.to_dict()
        if superclass is not None and e.get("superclass") != superclass:
            continue
        t = float(e["time"])
        lo = bisect.bisect_right(grid, t - spread_seconds)
        hi = bisect.bisect_left(grid, t + spread_seconds)
        for i in range(lo, hi):
            distance = abs(grid[i] - t)
            if distance < spread_seconds:
                track[i] = max(track[i], float(e["strength"]) * (1.0 - distance / spread_seconds))
    return track


def normalize(track: Sequence[float], percentile: float = 99.0) -> list[float]:
    """Scale to [0, 1] by a robust maximum (the given percentile), clipping above it."""
    if not track:
        return []
    ordered = sorted(float(v) for v in track)
    index = min(len(ordered) - 1, max(0, int(math.ceil(percentile / 100.0 * len(ordered))) - 1))
    scale = ordered[index] or ordered[-1]
    if scale <= 0:
        return [0.0] * len(track)
    return [min(1.0, max(0.0, float(v) / scale)) for v in track]
