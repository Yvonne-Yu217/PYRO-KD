"""Parse professional MTG-AD start times without inventing end times."""

from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class ADItem:
    """A professional AD start time in seconds and its description."""

    timestamp: float
    text: str


_ENTRY = re.compile(
    r"^(?P<timestamp>\S+?)(?:\s*-\s*STANDARD\b[ \t:]*|[ \t]+)(?P<text>.*)$"
)
_DECIMAL_OR_MINUTES = re.compile(
    r"(?:(?P<hours>\d+):)?(?P<minutes>\d+):(?P<seconds>\d{2})"
    r"(?:[.,](?P<fraction>\d+))?"
)
_COLON_MILLISECONDS = re.compile(
    r"(?P<minutes>\d+):(?P<seconds>\d{2}):(?P<milliseconds>\d{3})"
)


def _timestamp_to_seconds(timestamp: str) -> float:
    """Parse explicit clock formats, rejecting ambiguous three-part values."""
    legacy = _COLON_MILLISECONDS.fullmatch(timestamp)
    if legacy:
        # MTG-AD's unambiguous M:SS:mmm form, e.g. 0:17:460 = 17.460 s.
        minutes = int(legacy["minutes"])
        seconds = int(legacy["seconds"])
        fraction = int(legacy["milliseconds"]) / 1000
        hours = 0
    else:
        match = _DECIMAL_OR_MINUTES.fullmatch(timestamp)
        if match is None:
            raise ValueError(f"unsupported timestamp {timestamp!r}")
        if match["hours"] is not None and match["fraction"] is None:
            raise ValueError(
                f"ambiguous timestamp {timestamp!r}: use H:MM:SS.000 for"
                " hours/minutes/seconds or M:SS:mmm for colon milliseconds"
            )
        hours = int(match["hours"] or 0)
        minutes = int(match["minutes"])
        if match["hours"] is not None and (
            len(match["minutes"]) != 2 or minutes >= 60
        ):
            raise ValueError(f"invalid clock minutes in timestamp {timestamp!r}")
        seconds = int(match["seconds"])
        fraction = float("0." + match["fraction"]) if match["fraction"] else 0.0
    if seconds >= 60:
        raise ValueError(f"invalid seconds in timestamp {timestamp!r}")
    result = hours * 3600 + minutes * 60 + seconds + fraction
    if not math.isfinite(result):
        raise ValueError(f"non-finite timestamp {timestamp!r}")
    return float(result)


def parse_mtg_ad(text: str) -> list[ADItem]:
    """Parse '<timestamp>-STANDARD text' or '<timestamp> text' lines.

    Supports H:MM:SS.fraction, M:SS.fraction, M:SS, and M:SS:mmm, with
    either a dot or comma for decimal fractions. Fractional precision is
    preserved rather than truncated to milliseconds. Three-part timestamps
    without a decimal and with two final digits are ambiguous: MTG-AD reads
    them as M:SS:fraction, but they also resemble H:MM:SS. Reject them here
    instead of guessing. Use .000 for whole-second clock timestamps.

    Unicode U+2028/U+2029 separators between the type tag and description are
    normalized before splitting lines. Input order and duplicate entries are
    preserved. Every nonblank line must be a valid entry with nonempty text;
    malformed entries raise ValueError with the source line number.
    """
    items = []
    normalized = text.replace("\u2028", " ").replace("\u2029", " ")
    for line_number, raw_line in enumerate(normalized.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            match = _ENTRY.fullmatch(line)
            if match is None:
                raise ValueError("expected '<timestamp>-STANDARD description'")
            timestamp = _timestamp_to_seconds(match["timestamp"])
            description = match["text"].strip()
            if not description:
                raise ValueError("AD description must not be empty")
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid MTG-AD entry on line {line_number}: {exc}") from exc
        items.append(ADItem(timestamp, description))
    return items
