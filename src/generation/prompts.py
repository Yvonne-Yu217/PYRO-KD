"""AD prompts. P0 is plain; P1 adds guidelines and 3 sourced professional
exemplars. Everything except the prior itself is shared,
so prompt differences cannot leak into the selector comparison."""

import math
from typing import Sequence

from src.datasets.srt import SubtitleItem

WORDS_PER_SECOND = 3  # proposal: word limit = floor(gap seconds * 3)
DEFAULT_WORD_LIMIT = 15
MAX_WORDS = 30  # historical cap; only used when explicitly requested
WORD_BUDGET_RULE = "floor-gap-times-3-v1"
CONTEXT_SECONDS = 15.0

P0 = (
    "These {num_frames} frames are from a video, in order, spanning about "
    "{span:.0f} seconds around {time}.\n"
    "{context}"
    "Write one audio description for a blind or low-vision viewer of what is "
    "happening in these frames, in at most {word_limit} words. "
    "Output only the description."
)

P1_GUIDELINES = (
    "Audio-description guidelines:\n"
    "Describe observable actions, people, objects and spatial relationships. "
    "Use concise, objective, present-tense language. Do not invent identities, "
    "motives, emotions or events unsupported by the frames. Prioritize visual "
    "information needed to understand the action. Avoid repeating speech or "
    "information already clear from the supplied audio context. Respect the "
    "current word limit; do not pad short descriptions.\n"
    "Professional descriptions below illustrate style only. Their people, "
    "objects and events are unrelated to the current frames.\n"
)
PROMPTS = {"P0": P0, "P1": P0}


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


def word_limit_for_gap(gap_seconds: float | None, max_words: int | None = None) -> int:
    """Revision 1.5: floor(gap * 3); nonpositive gaps have a zero budget.

    A positive gap shorter than 1/3 s also fits no whole word. Unknown gaps
    use the explicit 15-word fallback, not a fit estimate. An optional cap is
    retained for explicit historical comparisons; core runs leave it unset.
    """
    if max_words is not None and max_words < 0:
        raise ValueError("max_words must be nonnegative")
    if gap_seconds is None:
        return DEFAULT_WORD_LIMIT if max_words is None else min(max_words, DEFAULT_WORD_LIMIT)
    if not math.isfinite(gap_seconds):
        raise ValueError("gap_seconds must be finite or None")
    budget = math.floor(max(0.0, gap_seconds) * WORDS_PER_SECOND)
    return budget if max_words is None else min(max_words, budget)


def build_prompt(
    prior: str,
    time: float,
    num_frames: int,
    span: float,
    word_limit: int,
    transcript_text: str = "",
    exemplars: Sequence[str] = (),
    audio_timeline_text: str | None = None,
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
    if audio_timeline_text is not None:
        if transcript_text:
            raise ValueError("timeline and legacy transcript context must not be mixed")
        context = audio_timeline_text + "\nDo not repeat information already conveyed by audio.\n"
    if prior == "P1" and (
        len(exemplars) != 3 or any(not isinstance(x, str) or not x.strip() for x in exemplars)
    ):
        raise ValueError("P1 requires exactly three nonempty professional exemplars")
    if prior == "P0" and exemplars:
        raise ValueError("P0 must not receive exemplars")
    plain = PROMPTS[prior].format(
        num_frames=num_frames, span=span, time=format_clock(time),
        context=context, word_limit=word_limit,
    )
    if prior == "P0":
        return plain
    examples = "\n".join(f"Example {i}: {text}" for i, text in enumerate(exemplars, 1))
    return P1_GUIDELINES + examples + "\n\nCurrent video:\n" + plain
