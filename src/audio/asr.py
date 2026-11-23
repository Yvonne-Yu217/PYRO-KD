"""VAD-gated Whisper ASR with word timestamps (proposal §2).

Whisper only sees VAD-positive regions: speech segments are padded, merged
across short pauses, and split at max_region_seconds; each region is
transcribed on its own and the word times are shifted back onto the video
clock. Gating reduces hallucinated text on music and noise and saves compute.

The transcriber is injected (a callable on a mono float32 chunk), so region
building, offsetting and validation are tested without a model. The default
backend is faster-whisper, imported only when it is built.

Ablation 3 (Whisper-only audio events) needs Whisper to hear the non-speech
audio as well, so it runs ungated (vad_gated=False) and parses Whisper's
bracketed tags with whisper_tag_events. That is the only ungated use.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Sequence

from src.audio.salience import AudioEvent
from src.audio.taxonomy import superclass_of

DEFAULT_ASR_CONFIG = {
    "backend": "faster-whisper",
    "model": "large-v3",
    "language": None,           # None = detect
    "compute_type": "float16",  # int8 on CPU
    "beam_size": 5,
    "vad_gated": True,
    "region_pad_seconds": 0.2,
    "region_merge_gap_seconds": 1.0,
    "max_region_seconds": 30.0,
}

# chunk (float32 mono, 16 kHz) -> [{"start", "end", "text", "words": [{"start", "end", "word", "probability"}]}]
Transcriber = Callable[..., list[dict]]


def vad_regions(
    speech: Sequence[tuple[float, float]],
    duration: float,
    pad: float = 0.2,
    merge_gap: float = 1.0,
    max_region: float = 30.0,
) -> list[tuple[float, float]]:
    """Padded, merged speech regions inside [0, duration], none longer than max_region."""
    if max_region <= 0:
        raise ValueError("max_region must be positive")
    padded = sorted((max(0.0, s - pad), min(duration, e + pad)) for s, e in speech if e > s)
    merged: list[list[float]] = []
    for s, e in padded:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    regions = []
    for s, e in merged:
        pieces = max(1, math.ceil((e - s) / max_region))
        step = (e - s) / pieces
        regions.extend((round(s + i * step, 3), round(s + (i + 1) * step, 3)) for i in range(pieces))
    return regions


def transcribe_regions(
    waveform, sample_rate: int, regions: Sequence[tuple[float, float]], transcriber: Transcriber,
) -> list[dict]:
    """Run the transcriber on each region; return segments on the video clock.

    Word and segment times are offset by the region start and clipped to the
    region, so nothing can land outside the speech the VAD found.
    """
    segments: list[dict] = []
    for start, end in regions:
        chunk = waveform[int(start * sample_rate):int(end * sample_rate)]
        if len(chunk) == 0:
            continue
        for seg in transcriber(chunk):
            text = str(seg.get("text", "")).strip()
            if not text:
                continue

            def clock(t):
                return round(min(end, max(start, start + float(t))), 3)

            words = [
                {"start": clock(w["start"]), "end": clock(w["end"]), "word": str(w.get("word", "")).strip(),
                 **({"probability": float(w["probability"])} if w.get("probability") is not None else {})}
                for w in seg.get("words") or []
                if w.get("start") is not None and w.get("end") is not None
            ]
            segments.append({
                "start": clock(seg.get("start", 0.0)),
                "end": clock(seg.get("end", end - start)),
                "text": text,
                "words": words,
            })
    segments.sort(key=lambda s: s["start"])
    return segments


def faster_whisper_transcriber(
    model: str = "large-v3", device: str = "auto", compute_type: str = "float16",
    language: str | None = None, beam_size: int = 5,
) -> Transcriber:
    """faster-whisper with word timestamps at temperature 0; loads the model once."""
    from faster_whisper import WhisperModel

    whisper = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(chunk) -> list[dict]:
        segments, _info = whisper.transcribe(
            chunk, language=language, beam_size=beam_size, word_timestamps=True,
            temperature=0.0, vad_filter=False, condition_on_previous_text=False,
        )
        return [
            {
                "start": s.start, "end": s.end, "text": s.text,
                "words": [
                    {"start": w.start, "end": w.end, "word": w.word, "probability": w.probability}
                    for w in (s.words or [])
                ],
            }
            for s in segments
        ]

    return transcribe


def validate_segments(segments: Sequence[dict], duration: float) -> str | None:
    """First defect in ASR output (times out of range or non-monotone words), or None."""
    for i, seg in enumerate(segments):
        try:
            start, end = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            return f"segment {i} lacks numeric start/end"
        if not (0 <= start <= end <= duration + 1e-6):
            return f"segment {i} is outside [0, {duration}]"
        previous = start
        for j, word in enumerate(seg.get("words") or []):
            ws, we = float(word["start"]), float(word["end"])
            if not (start - 1e-6 <= ws <= we <= end + 1e-6) or ws < previous - 1e-6:
                return f"segment {i} word {j} is out of order or outside its segment"
            previous = ws
    return None


# Whisper sometimes emits non-speech tags such as "[MUSIC]" or "(applause)".
_TAG = re.compile(r"[\[\(]\s*([A-Za-z][A-Za-z \-]{1,30}?)\s*[\]\)]|(♪+)")
_TAG_TO_LABEL = {
    "music": "Music", "music playing": "Music", "upbeat music": "Music", "soft music": "Music",
    "applause": "Applause", "clapping": "Clapping", "laughter": "Laughter", "laughs": "Laughter",
    "laughing": "Laughter", "cheering": "Cheering", "door slams": "Slam", "door closes": "Door",
    "knocking": "Knock", "horn honks": "Vehicle horn, car horn, honking", "gunshot": "Gunshot, gunfire",
    "explosion": "Explosion", "dog barking": "Bark", "silence": "Silence", "inaudible": "Noise",
    "noise": "Noise", "wind": "Wind", "rain": "Rain",
}


def whisper_tag_events(segments: Sequence[dict]) -> list[AudioEvent]:
    """Non-speech events from Whisper's bracketed tags (ablation 3's baseline).

    Each tag becomes an event at its segment start with strength 1. Tags that
    do not map to an AudioSet name keep their lower-cased text as the label
    (superclass "other"). Silence tags are dropped.
    """
    events = []
    for seg in segments:
        text = str(seg.get("text", ""))
        for match in _TAG.finditer(text):
            raw = (match.group(1) or "music").strip().lower()
            label = _TAG_TO_LABEL.get(raw, raw)
            superclass = superclass_of(label)
            if superclass in ("speech", "silence"):
                continue
            events.append(AudioEvent(float(seg["start"]), label, superclass, 1.0, source="whisper_tag"))
    return sorted(events, key=lambda e: (e.time, e.label))
