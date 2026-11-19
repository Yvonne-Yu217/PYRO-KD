"""AD and QA prompts. P0 is the plain instruction; P1 prepends the guideline
summary and 3 professional exemplars (src/generation/priors.py). Everything
except the prior itself is shared, so prompt differences cannot leak into the
selector comparison. An optional audio timeline (src/audio/timeline.py) adds
one line of context; with no timeline the P0 text is byte-identical to the
original, so earlier runs stay comparable."""

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

PROMPTS = {"P0": P0, "P1": "{prior_block}" + P0}

QA_PROMPT = (
    "These {num_frames} frames are sampled from a video, in order, at {times}.\n"
    "{context}"
    "Question: {question}\n"
    "{options}\n"
    "Answer with the letter of the correct option only."
)


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
    audio_timeline: str = "",
    exemplars: dict | None = None,
) -> str:
    """The AD prompt. P1 needs real exemplars (loaded from configs/p1_exemplars.json
    when not passed); a missing file raises priors.MissingAsset."""
    if prior not in PROMPTS:
        raise ValueError(f"unknown prior {prior!r}; available: {sorted(PROMPTS)}")
    if word_limit <= 0:
        raise ValueError("moment is not safely speakable: word_limit must be positive")
    context = (
        f'Nearby speech in the video: "{transcript_text}"\n'
        "Do not repeat what is said.\n"
        if transcript_text else ""
    )
    if audio_timeline:
        context += f"Audio timeline around this moment: {audio_timeline}\n"
    prior_block = ""
    if prior == "P1":
        from src.generation.priors import load_exemplars, prior_block as make_block

        prior_block = make_block(exemplars if exemplars is not None else load_exemplars())
    return PROMPTS[prior].format(
        num_frames=num_frames, span=span, time=format_clock(time),
        context=context, word_limit=word_limit, prior_block=prior_block,
    )


def build_qa_prompt(
    question: str, options: Sequence[str], frame_times: Sequence[float], audio_timeline: str = "",
    transcript_text: str = "",
) -> str:
    """Zero-shot multiple-choice QA over keyframes, with an optional audio timeline."""
    from src.audio.timeline import clock

    if not options:
        raise ValueError("a multiple-choice question needs options")
    letters = "ABCDEFGHIJ"
    context = ""
    if transcript_text:
        context += f'Speech in the video: "{transcript_text}"\n'
    if audio_timeline:
        context += f"Audio timeline of the video: {audio_timeline}\n"
    return QA_PROMPT.format(
        num_frames=len(frame_times), times=", ".join(clock(t) for t in frame_times), context=context,
        question=question.strip(), options="\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options)),
    )


def count_words(text: str) -> int:
    return len(text.split())


def enforce_word_limit(text: str, word_limit: int, policy: str = "truncate") -> dict:
    """Apply the strict word cap (Risk 1). policy "truncate" keeps the first
    word_limit words for rendering; "flag" keeps the text. Both record the raw
    count and whether the model exceeded its budget."""
    if policy not in ("truncate", "flag"):
        raise ValueError("policy must be 'truncate' or 'flag'")
    words = text.split()
    over = len(words) > word_limit
    rendered = " ".join(words[:word_limit]) if over and policy == "truncate" else text.strip()
    return {"text": rendered, "raw_words": len(words), "words": count_words(rendered),
            "over_budget": over, "truncated": over and policy == "truncate"}
