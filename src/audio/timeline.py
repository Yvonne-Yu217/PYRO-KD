"""Compact audio timeline for prompts (proposal §5.4), e.g.

    [0:42 music starts] [0:55 applause] [1:03 speech: "we made it"]

The format is a held-constant control (§5.6): every configuration that shows
a timeline uses this function with the same settings. For long videos, the
global timeline keeps the strongest events within max_items, so it also serves
as the QA-stage summary.
"""

from __future__ import annotations

from typing import Sequence

TIMELINE_VERSION = "v1"
DEFAULT_TIMELINE_CONFIG = {
    "max_items": 12,
    "quote_words": 8,
    "include_speech": True,
    "include_cues": False,  # deictic / semantic-shift cues are selector internals, not shown
}


def clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _event_phrase(event: dict) -> str:
    label = str(event.get("label", "sound")).split(",")[0].strip().lower()
    if event.get("superclass") == "music" or label == "music":
        return "music starts"
    return label


def timeline_items(
    events: Sequence[dict],
    speech_segments: Sequence[dict],
    start: float = 0.0,
    end: float = float("inf"),
    config: dict | None = None,
) -> list[tuple[float, str, float]]:
    """(time, text, priority) items inside [start, end], strongest first, then trimmed."""
    cfg = {**DEFAULT_TIMELINE_CONFIG, **(config or {})}
    items: list[tuple[float, str, float]] = []
    for e in events:
        t = float(e["time"])
        if start <= t <= end:
            items.append((t, _event_phrase(e), float(e.get("strength", 1.0))))
    if cfg["include_speech"]:
        for s in speech_segments:
            t = float(s["start"])
            text = " ".join(str(s.get("text", "")).split())
            if not text or not (start <= t <= end):
                continue
            words = text.split()
            quote = " ".join(words[: cfg["quote_words"]]) + ("..." if len(words) > cfg["quote_words"] else "")
            quote = quote.replace('"', "'")
            items.append((t, f'speech: "{quote}"', 0.5))
    items.sort(key=lambda item: (-item[2], item[0]))
    kept = items[: cfg["max_items"]]
    return sorted(kept, key=lambda item: item[0])


def format_timeline(
    events: Sequence[dict],
    speech_segments: Sequence[dict],
    start: float = 0.0,
    end: float = float("inf"),
    config: dict | None = None,
) -> str:
    """The timeline string, or "" when nothing falls in the window."""
    return " ".join(f"[{clock(t)} {text}]" for t, text, _ in timeline_items(events, speech_segments, start, end, config))
