"""AD prompts. P0 is the plain instruction; P1 (guidelines + 3 professional
exemplars) is added in Week 3. Everything except the prior itself is shared,
so prompt differences cannot leak into the selector comparison."""

import math
from typing import Sequence

from src.datasets.srt import SubtitleItem

WORDS_PER_SECOND = 3  # proposal: word limit = floor(gap seconds * 3)
DEFAULT_WORD_LIMIT = 15
MAX_WORDS = 30  # concision cap so long silences do not license long descriptions
CONTEXT_SECONDS = 15.0

P0 = (
    "These {num_frames} frames are from a video, in order, spanning about "
    "{span:.0f} seconds around {time}.\n"
    "{context}"
    "Write one audio description for a blind or low-vision viewer of what is "
    "happening in these frames, in at most {word_limit} words. "
    "Output only the description."
)

PROMPTS = {"P0": P0}


def format_clock(seconds: float) -> str:
    minutes, secs = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes)}:{secs:04.1f}"


def transcript_context(
    transcript: Sequence[SubtitleItem],
    time: float,
    window: float = CONTEXT_SECONDS,
) -> str:
    """Speech overlapping [time - window, time + window], in order."""
    lines = [
        item.text for item in transcript
        if item.end >= time - window and item.start <= time + window
    ]
    return " ".join(lines)


def word_limit_for_gap(gap_seconds: float | None, max_words: int = MAX_WORDS) -> int:
    """min(max_words, floor(gap * 3)); nonpositive gaps have a zero budget.

    A positive gap shorter than 1/3 s also fits no whole word. Unknown gaps
    use the explicit 15-word fallback, capped by max_words, not a fit estimate.
    """
    if max_words < 0:
        raise ValueError("max_words must be nonnegative")
    if gap_seconds is None:
        return min(max_words, DEFAULT_WORD_LIMIT)
    if not math.isfinite(gap_seconds):
        raise ValueError("gap_seconds must be finite or None")
    return min(max_words, math.floor(max(0.0, gap_seconds) * WORDS_PER_SECOND))


def build_prompt(
    prior: str,
    time: float,
    num_frames: int,
    span: float,
    word_limit: int,
    transcript_text: str = "",
) -> str:
    if prior not in PROMPTS:
        raise ValueError(f"unknown prior {prior!r}; available: {sorted(PROMPTS)}")
    if word_limit <= 0:
        raise ValueError("moment is not safely speakable: word_limit must be positive")
    context = (
        f'Nearby speech in the video: "{transcript_text}"\n'
        "Do not repeat what is said.\n"
        if transcript_text else ""
    )
    return PROMPTS[prior].format(
        num_frames=num_frames, span=span, time=format_clock(time),
        context=context, word_limit=word_limit,
    )
