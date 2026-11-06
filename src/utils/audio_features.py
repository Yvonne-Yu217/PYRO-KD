"""Speech/non-speech structure and per-gap acoustic features for one video.

Pipeline: 16 kHz mono audio -> Silero VAD speech segments -> inverted into
non-speech gaps (short ones dropped, edges next to speech pulled in by a safety
margin) -> librosa RMS energy and onset strength summarized inside each gap.
Everything is cached as audio_features.json next to the video's manifest, so
selectors and the word cap never re-run the models.

VAD marks absence of speech, not silence: music, alarms, and laughter can sit
inside a gap, which is why energy and onset features are kept per gap.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

FEATURES_NAME = "audio_features.json"
SAMPLE_RATE = 16_000
HOP_LENGTH = 512

DEFAULT_CONFIG = {
    "vad_threshold": 0.5,
    "min_speech_seconds": 0.25,
    "min_silence_seconds": 0.1,
    "min_gap_seconds": 1.0,
    "edge_margin_seconds": 0.25,
}


@dataclass(frozen=True)
class Gap:
    """A usable non-speech interval (seconds) with acoustic summaries."""

    start: float
    end: float
    rms: float
    onset: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def merge_segments(segments: Sequence[tuple[float, float]], max_gap: float = 0.0) -> list[tuple[float, float]]:
    """Sort and merge segments that overlap or are separated by <= max_gap."""
    merged: list[tuple[float, float]] = []
    for start, end in sorted(segments):
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def invert_to_gaps(
    speech: Sequence[tuple[float, float]],
    duration: float,
    min_gap: float = DEFAULT_CONFIG["min_gap_seconds"],
    margin: float = DEFAULT_CONFIG["edge_margin_seconds"],
) -> list[tuple[float, float]]:
    """Complement of speech within [0, duration], as usable gaps.

    Edges that touch speech move inward by margin; edges at the clip start or
    end do not. Gaps shorter than min_gap after trimming are discarded.
    """
    speech = merge_segments([(max(s, 0.0), min(e, duration)) for s, e in speech if e > 0 and s < duration])
    gaps = []
    cursor = 0.0
    for index, (start, end) in enumerate(speech + [(duration, duration)]):
        gap_start = cursor + (margin if cursor > 0.0 else 0.0)
        gap_end = start - (margin if index < len(speech) else 0.0)
        if gap_end - gap_start >= min_gap:
            gaps.append((gap_start, gap_end))
        cursor = max(cursor, end)
    return gaps


def available_gap(t: float, speech: Sequence[tuple[float, float]], duration: float) -> float:
    """Seconds from t until the next speech onset (0 if t is inside speech).

    This is the Delta-t behind the word cap, applied identically to every
    selector so gap fit is comparable across them.
    """
    for start, end in sorted(speech):
        if start <= t < end:
            return 0.0
        if start > t:
            return start - t
    return max(duration - t, 0.0)


def mean_in_window(times: Sequence[float], values: Sequence[float], start: float, end: float) -> float:
    """Mean of values whose frame time lies in [start, end]; 0.0 if there are none."""
    selected = [v for t, v in zip(times, values) if start <= t <= end]
    return sum(selected) / len(selected) if selected else 0.0


def detect_speech(waveform, sample_rate: int = SAMPLE_RATE, config: dict | None = None) -> list[tuple[float, float]]:
    """Silero VAD speech intervals in seconds for a mono float32 waveform."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    stamps = get_speech_timestamps(
        torch.from_numpy(waveform), load_silero_vad(), sampling_rate=sample_rate,
        threshold=cfg["vad_threshold"],
        min_speech_duration_ms=int(cfg["min_speech_seconds"] * 1000),
        min_silence_duration_ms=int(cfg["min_silence_seconds"] * 1000),
        return_seconds=True,
    )
    return [(float(s["start"]), float(s["end"])) for s in stamps]


def gap_features(waveform, sample_rate: int, gaps: Sequence[tuple[float, float]]) -> list[Gap]:
    """Mean RMS energy and mean onset strength inside each gap.

    Both curves use librosa's centered frames, so frame i is at
    i * hop / sr seconds (frames_to_time), matching the video clock.
    """
    import librosa
    import numpy as np

    rms = librosa.feature.rms(y=waveform, hop_length=HOP_LENGTH)[0]
    onset = librosa.onset.onset_strength(y=waveform, sr=sample_rate, hop_length=HOP_LENGTH)
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=HOP_LENGTH).tolist()
    onset_times = librosa.frames_to_time(np.arange(len(onset)), sr=sample_rate, hop_length=HOP_LENGTH).tolist()
    return [
        Gap(
            start=start, end=end,
            rms=mean_in_window(rms_times, rms.tolist(), start, end),
            onset=mean_in_window(onset_times, onset.tolist(), start, end),
        )
        for start, end in gaps
    ]


def compute_audio_features(audio_path: str | Path, config: dict | None = None) -> dict:
    """Run VAD and gap analysis on one cached 16 kHz WAV; returns the JSON record."""
    from src.utils.media import load_audio

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    waveform, sample_rate = load_audio(audio_path, SAMPLE_RATE)
    duration = len(waveform) / sample_rate
    speech = detect_speech(waveform, sample_rate, cfg)
    gaps = gap_features(
        waveform, sample_rate,
        invert_to_gaps(speech, duration, cfg["min_gap_seconds"], cfg["edge_margin_seconds"]),
    )
    return {
        "audio_path": str(audio_path),
        "duration": round(duration, 3),
        "config": cfg,
        "speech": [[round(s, 3), round(e, 3)] for s, e in speech],
        "speech_seconds": round(sum(e - s for s, e in speech), 3),
        "gaps": [asdict(g) for g in gaps],
    }


def write_features(cache_dir: str | Path, features: dict) -> Path:
    path = Path(cache_dir) / FEATURES_NAME
    path.write_text(json.dumps(features, indent=2))
    return path


def load_features(cache_dir: str | Path) -> dict | None:
    """Read cached features, or None if this video has none (e.g. no audio stream)."""
    path = Path(cache_dir) / FEATURES_NAME
    return json.loads(path.read_text()) if path.exists() else None


def speech_of(features: dict) -> list[tuple[float, float]]:
    return [(s, e) for s, e in features["speech"]]


def gaps_of(features: dict) -> list[Gap]:
    return [Gap(**g) for g in features["gaps"]]
