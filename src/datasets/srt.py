"""Minimal SRT parsing for REFRAMED AD and subtitle tracks."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SubtitleItem:
    """One SRT cue with clip-relative start/end times in seconds."""

    start: float
    end: float
    text: str


_TIME = re.compile(
    r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{1,3})"
)
_TIMING_LINE = re.compile(
    r"^\s*(?P<start>\S+)\s*-->\s*(?P<end>\S+)(?:\s+.*)?$"
)


def srt_time_to_seconds(value: str) -> float:
    """Convert 'HH:MM:SS,mmm' (or with '.') to seconds."""
    match = _TIME.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid SRT timestamp {value!r}")
    minutes = int(match["m"])
    seconds = int(match["s"])
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"invalid SRT timestamp {value!r}")
    fraction = int(match["ms"].ljust(3, "0")) / 1000
    return int(match["h"]) * 3600 + minutes * 60 + seconds + fraction


def parse_srt(text: str | None) -> list[SubtitleItem]:
    """Parse SRT text into cues, preserving order and multi-line text.

    Cue index lines are optional. Text lines are joined with a newline.
    Malformed timing lines and cues with end < start raise ValueError.
    """
    if not text:
        return []
    normalized = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    items = []
    for block_number, block in enumerate(re.split(r"\n\s*\n", normalized.strip()), 1):
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        if "-->" not in lines[0]:
            lines = lines[1:]
        if not lines:
            continue
        timing = _TIMING_LINE.fullmatch(lines[0])
        if timing is None:
            raise ValueError(f"SRT block {block_number}: expected a timing line")
        start = srt_time_to_seconds(timing["start"])
        end = srt_time_to_seconds(timing["end"])
        if end < start:
            raise ValueError(f"SRT block {block_number}: end precedes start")
        items.append(SubtitleItem(start, end, "\n".join(lines[1:]).strip()))
    return items
