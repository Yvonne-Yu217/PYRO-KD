"""Speech cues (SC, proposal §3.2): deictic phrases, semantic shift, speaker
change, and long speech-free stretches.

Transcript segments are dicts {"start", "end", "text"} with optional
"words": [{"start", "end", "word"}] (word timestamps from VAD-gated Whisper).
MTG-AD's released ASR has segment times only; then a phrase time is
interpolated inside its segment by character position, and the event says so
(timing="interpolated") so analyses can separate the two.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import re
from typing import Callable, Sequence

# Visually referential phrases: the speaker points at something the dialogue
# does not fully explain. Lower-case regexes over normalized text.
DEICTIC_PATTERNS: tuple[str, ...] = (
    r"\blook at (this|that|these|those|him|her|it|them)\b",
    r"\b(take|have) a look\b",
    r"\blook\b(?= (here|there|over|how|what|at))",
    r"\bover (there|here)\b",
    r"\bright (here|there)\b",
    r"\bwatch (this|that|out|how|me)\b",
    r"\bcheck (this|that|it) out\b",
    r"\b(as )?you can see\b",
    r"\bsee (this|that|here|how)\b",
    r"\blike this\b",
    r"\bthis one\b",
    r"\bthat one\b",
    r"\bhere we (go|have|are)\b",
    r"\bthis is what\b",
)
_DEICTIC = [re.compile(p) for p in DEICTIC_PATTERNS]

DEFAULT_SC_CONFIG = {
    "semantic_window": 3,          # segments on each side
    "semantic_threshold": 0.35,    # 1 - cosine between windows
    "long_silence_seconds": 8.0,   # gaps at least this long count as long speech-free stretches
}


@dataclass(frozen=True)
class SpeechCue:
    time: float
    kind: str  # deictic | semantic_shift | speaker_change | long_silence
    strength: float
    text: str = ""
    timing: str = "word"  # word | interpolated | segment | vad

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def _segment_ok(segment: dict) -> bool:
    try:
        return (
            isinstance(segment.get("text"), str)
            and math.isfinite(float(segment["start"])) and math.isfinite(float(segment["end"]))
            and float(segment["end"]) >= float(segment["start"]) >= 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def deictic_cues(segments: Sequence[dict]) -> list[SpeechCue]:
    """One cue per deictic phrase occurrence, at the phrase's first word."""
    cues: list[SpeechCue] = []
    for segment in segments:
        if not _segment_ok(segment):
            continue
        words = segment.get("words") or []
        start, end = float(segment["start"]), float(segment["end"])
        if words:
            tokens = [(_normalize(str(w.get("word", ""))), w) for w in words]
            tokens = [(t, w) for t, w in tokens if t]
            text = " ".join(t for t, _ in tokens)
            offsets, cursor = [], 0
            for t, _ in tokens:
                offsets.append(cursor)
                cursor += len(t) + 1
            for pattern in _DEICTIC:
                for match in pattern.finditer(text):
                    index = max(i for i, o in enumerate(offsets) if o <= match.start())
                    cues.append(SpeechCue(float(tokens[index][1]["start"]), "deictic", 1.0, match.group(0), "word"))
        else:
            text = _normalize(segment["text"])
            for pattern in _DEICTIC:
                for match in pattern.finditer(text):
                    fraction = match.start() / max(1, len(text))
                    cues.append(SpeechCue(
                        round(start + fraction * (end - start), 3), "deictic", 1.0, match.group(0), "interpolated",
                    ))
    return sorted(cues, key=lambda c: c.time)


Encoder = Callable[[list[str]], list[list[float]]]


def lexical_encoder(dims: int = 256) -> Encoder:
    """Hashed bag-of-words vectors. A dependency-free fallback, clearly weaker
    than a sentence encoder; record which encoder produced a cue file."""

    def encode(texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            v = [0.0] * dims
            for token in _normalize(text).split():
                v[int(hashlib.md5(token.encode()).hexdigest(), 16) % dims] += 1.0
            vectors.append(v)
        return vectors

    return encode


def sentence_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None) -> Encoder:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    def encode(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).tolist()

    return encode


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def semantic_shift_cues(
    segments: Sequence[dict], encoder: Encoder, window: int = 3, threshold: float = 0.35,
) -> list[SpeechCue]:
    """Topic shifts: 1 - cos(mean of the previous `window` segment vectors, mean of
    the next `window`), at each segment start; kept where it is a local maximum
    and at least `threshold`."""
    usable = [s for s in segments if _segment_ok(s) and s["text"].strip()]
    usable.sort(key=lambda s: float(s["start"]))
    if len(usable) < 2:
        return []
    vectors = encoder([s["text"] for s in usable])
    if len(vectors) != len(usable):
        raise ValueError("encoder returned the wrong number of vectors")

    def mean(rows):
        return [sum(col) / len(rows) for col in zip(*rows)]

    shifts = [0.0]
    for i in range(1, len(usable)):
        before = vectors[max(0, i - window):i]
        after = vectors[i:i + window]
        shifts.append(max(0.0, 1.0 - _cosine(mean(before), mean(after))))
    cues = []
    for i in range(1, len(usable)):
        left = shifts[i - 1] if i > 1 else -1.0
        right = shifts[i + 1] if i + 1 < len(shifts) else -1.0
        if shifts[i] >= threshold and shifts[i] >= left and shifts[i] > right:
            cues.append(SpeechCue(float(usable[i]["start"]), "semantic_shift", round(shifts[i], 4), "", "segment"))
    return cues


def speaker_change_cues(turns: Sequence[dict]) -> list[SpeechCue]:
    """Times where the diarized speaker differs from the previous turn's."""
    ordered = sorted((t for t in turns if "speaker" in t and "start" in t), key=lambda t: float(t["start"]))
    return [
        SpeechCue(float(cur["start"]), "speaker_change", 1.0, "", "segment")
        for prev, cur in zip(ordered, ordered[1:]) if cur["speaker"] != prev["speaker"]
    ]


def long_silence_cues(gaps: Sequence[tuple[float, float]], min_seconds: float = 8.0) -> list[SpeechCue]:
    """One cue at the start of each VAD gap of at least min_seconds; strength grows
    with length (min_seconds -> 0.5, 2 x min_seconds or more -> 1)."""
    cues = []
    for start, end in gaps:
        length = end - start
        if length >= min_seconds:
            strength = min(1.0, 0.5 * length / min_seconds)
            cues.append(SpeechCue(float(start), "long_silence", round(strength, 4), "", "vad"))
    return cues


def long_silence_track(
    gaps: Sequence[tuple[float, float]], grid: Sequence[float], min_seconds: float = 8.0,
) -> list[float]:
    """1 at grid points inside a VAD gap of at least min_seconds, else 0."""
    long_gaps = [(s, e) for s, e in gaps if e - s >= min_seconds]
    return [1.0 if any(s <= g < e for s, e in long_gaps) else 0.0 for g in grid]
