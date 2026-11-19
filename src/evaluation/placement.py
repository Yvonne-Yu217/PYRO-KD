"""Placement metrics beyond temporal P/R/F1 (proposal §7A) and placement diagnostics.

Speech overlap: each AD is rendered at tts_speed (1.2x) from a base TTS rate
of tts_words_per_second, so a w-word description lasts w / (rate * speed)
seconds from its start time; overlap is the seconds of that interval that
fall inside detected speech. For selector-only runs the word count is the
moment's word budget (the worst case: the model uses every word it is
allowed); for generated AD it is the actual word count.

Silence utilization: rendered duration / available gap at the start time
(time to the next speech onset, the same Delta-t as the word cap). Undefined
(None) when the start is inside speech (gap 0).

Diagnostics (from the U-vs-A validation work): span of the picks, share of
picks in the densest tenth of the video, share in the single busiest VAD gap,
median spacing, picks inside speech, and the share of GT AD starts outside
detected speech. These expose clustering (all picks in one gap) and the
speech-free-video start-packing failure.
"""

from __future__ import annotations

from typing import Sequence

from src.generation.prompts import word_limit_for_gap
from src.utils.audio_features import available_gap

TTS_WORDS_PER_SECOND = 2.5
TTS_SPEED = 1.2


def rendered_seconds(n_words: int, words_per_second: float = TTS_WORDS_PER_SECOND, speed: float = TTS_SPEED) -> float:
    if n_words < 0 or words_per_second <= 0 or speed <= 0:
        raise ValueError("need n_words >= 0 and positive rate and speed")
    return n_words / (words_per_second * speed)


def interval_overlap(start: float, end: float, speech: Sequence[tuple[float, float]]) -> float:
    return sum(max(0.0, min(end, e) - max(start, s)) for s, e in speech)


def speech_overlap(
    placements: Sequence[tuple[float, int]],
    speech: Sequence[tuple[float, float]],
    words_per_second: float = TTS_WORDS_PER_SECOND,
    speed: float = TTS_SPEED,
) -> dict:
    """Overlap of rendered ADs (time, n_words) with speech: total and per AD."""
    per = []
    for t, n in placements:
        d = rendered_seconds(int(n), words_per_second, speed)
        per.append(round(interval_overlap(float(t), float(t) + d, speech), 4))
    total_rendered = sum(rendered_seconds(int(n), words_per_second, speed) for _, n in placements)
    return {
        "overlap_seconds": round(sum(per), 4),
        "per_ad_overlap_seconds": per,
        "ads_with_overlap": sum(v > 0 for v in per),
        "overlap_fraction_of_rendered": (sum(per) / total_rendered) if total_rendered else None,
    }


def silence_utilization(
    placements: Sequence[tuple[float, int]],
    speech: Sequence[tuple[float, float]],
    duration: float,
    words_per_second: float = TTS_WORDS_PER_SECOND,
    speed: float = TTS_SPEED,
) -> dict:
    """Per-AD rendered duration / available gap, and the mean over ADs with a gap."""
    ratios: list[float | None] = []
    for t, n in placements:
        gap = available_gap(float(t), speech, duration)
        ratios.append(round(rendered_seconds(int(n), words_per_second, speed) / gap, 4) if gap > 0 else None)
    kept = [r for r in ratios if r is not None]
    return {"per_ad": ratios, "mean": (sum(kept) / len(kept)) if kept else None, "n_in_speech": ratios.count(None)}


def budget_placements(times: Sequence[float], speech: Sequence[tuple[float, float]], duration: float) -> list[tuple[float, int]]:
    """(time, word budget) per selected time under the shared cap (selector-only runs)."""
    return [(t, word_limit_for_gap(available_gap(t, speech, duration))) for t in times]


def densest_window_share(times: Sequence[float], window: float) -> float | None:
    """Largest fraction of times inside any closed window of the given width."""
    ordered = sorted(times)
    if not ordered:
        return None
    best, j = 0, 0
    for i, t in enumerate(ordered):
        while ordered[j] < t - window:
            j += 1
        best = max(best, i - j + 1)
    return best / len(ordered)


def placement_diagnostics(
    times: Sequence[float], duration: float, speech: Sequence[tuple[float, float]] | None,
    gaps: Sequence[tuple[float, float]] | None,
) -> dict:
    ordered = sorted(times)
    spacing = sorted(b - a for a, b in zip(ordered, ordered[1:]))
    out = {
        "num_picks": len(ordered),
        "span_fraction": ((ordered[-1] - ordered[0]) / duration) if len(ordered) > 1 and duration > 0 else None,
        "densest_decile_share": densest_window_share(ordered, duration / 10) if duration > 0 else None,
        "median_spacing_seconds": spacing[len(spacing) // 2] if spacing else None,
        "busiest_gap_share": None,
        "picks_in_speech": None,
    }
    if speech is not None and ordered:
        out["picks_in_speech"] = sum(available_gap(t, speech, duration) == 0.0 for t in ordered)
    if gaps is not None and ordered:
        per_gap = [sum(1 for t in ordered if s <= t <= e) for s, e in gaps]
        out["busiest_gap_share"] = max(per_gap, default=0) / len(ordered)
    return out


def gt_in_gap_fraction(gt_times: Sequence[float], speech: Sequence[tuple[float, float]] | None, duration: float) -> float | None:
    """Share of GT AD starts outside detected speech (how often humans used gaps)."""
    if speech is None or not gt_times:
        return None
    return sum(available_gap(t, speech, duration) > 0 for t in gt_times) / len(gt_times)


def timeline_svg(rows: Sequence[dict], width: int = 960, title: str = "") -> str:
    """Dependency-free SVG: one lane per (video, selector) with speech shaded, GT ticks
    and selected times. rows: {"label", "duration", "speech", "gt", "times"}."""
    lane, pad, left = 22, 6, 220
    height = 30 + len(rows) * (lane + pad)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="11">',
             f'<text x="4" y="16" font-weight="bold">{_esc(title)}</text>']
    usable = width - left - 10
    for r, row in enumerate(rows):
        y = 26 + r * (lane + pad)
        dur = max(float(row["duration"]), 1e-9)

        def x(t):
            return left + usable * min(max(float(t), 0.0), dur) / dur

        parts.append(f'<text x="4" y="{y + 15}">{_esc(str(row["label"]))[:34]}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{usable}" height="{lane}" fill="#f2f2f2"/>')
        for s, e in row.get("speech") or []:
            parts.append(f'<rect x="{x(s):.1f}" y="{y}" width="{max(0.5, x(e) - x(s)):.1f}" height="{lane}" fill="#c9d7f0"/>')
        for t in row.get("gt") or []:
            parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{y}" y2="{y + lane / 2}" stroke="#2a7d2a" stroke-width="1.5"/>')
        for t in row.get("times") or []:
            parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{y + lane / 2}" y2="{y + lane}" stroke="#c0392b" stroke-width="1.5"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
