"""Speech/non-speech structure and per-gap acoustic features for one video.

Pipeline: 16 kHz mono audio -> Silero VAD speech segments -> inverted into
non-speech gaps (short ones dropped, edges next to speech pulled in by a safety
margin) -> librosa RMS energy and onset strength summarized inside each gap.
Everything is cached as audio_features.json next to the video's manifest, so
selectors and the word cap never re-run the models.

VAD marks absence of speech, not silence: music, alarms, and laughter can sit
inside a gap, which is why energy and onset features are kept per gap.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Sequence

FEATURES_NAME = "audio_features.json"
# Bump when the feature computation changes in a way the settings below miss.
FEATURES_VERSION = 2
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


def wav_digest(audio_path: str | Path) -> str:
    """SHA-256 of the WAV's bytes (~10 MB per 5-minute video; cheap next to VAD).

    Content, not size/mtime, so copying or restoring a cache keeps features
    valid, while any re-extraction that changes the audio invalidates them.
    Raises OSError if the WAV is missing or unreadable.
    """
    digest = hashlib.sha256()
    with Path(audio_path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _settings_hash(audio_digest: str, config: dict | None) -> str:
    from src.utils.media import config_hash

    return config_hash({
        "features_version": FEATURES_VERSION,
        "config": {**DEFAULT_CONFIG, **(config or {})},
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "audio_sha256": audio_digest,
    })


def features_config_hash(
    audio_path: str | Path, config: dict | None = None, manifest_hash: str | None = None,
) -> str:
    """Hash of everything audio_features.json depends on.

    Covers the VAD/gap config, sample rate, hop length, feature version, and
    the WAV's contents, so a changed config or changed audio invalidates
    cached features. Raises OSError if the WAV is missing.

    manifest_hash is accepted for callers written against the first version of
    this function and is ignored: the WAV's bytes already identify the audio,
    and the manifest hash also changes with frame-only settings (fps, max_side).
    """
    del manifest_hash
    return _settings_hash(wav_digest(audio_path), config)


def _private_file(mode: str, **mkstemp_args):
    """Create a unique temporary file and return (path, open file object).

    The raw mkstemp descriptor is owned by the returned file object; if wrapping
    it fails, the descriptor is closed and the file removed, so no path leaks
    an open descriptor or a stray file.
    """
    descriptor, name = tempfile.mkstemp(**mkstemp_args)
    path = Path(name)
    try:
        handle = os.fdopen(descriptor, mode, **({"encoding": "utf-8"} if "b" not in mode else {}))
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return path, handle


def _snapshot_wav(audio_path: str | Path) -> tuple[Path, str]:
    """Copy the WAV to a private temporary file while hashing the copied bytes.

    Decoding that private copy means the digest always describes exactly the
    bytes analysed, whatever happens to the cached WAV meanwhile (including an
    A->B->A replacement, which before/after hashes of the path cannot detect).
    The caller deletes the returned file.
    """
    digest = hashlib.sha256()
    # Open the source first: a missing or unreadable WAV then fails before any
    # temporary file or descriptor exists, so nothing can leak across a batch.
    with Path(audio_path).open("rb") as source:
        snapshot, target = _private_file(prefix="audio_snapshot_", suffix=".wav", mode="wb")
        try:
            with target:
                for block in iter(lambda: source.read(1 << 20), b""):
                    digest.update(block)
                    target.write(block)
        except BaseException:
            snapshot.unlink(missing_ok=True)
            raise
    return snapshot, digest.hexdigest()


def compute_audio_features(
    audio_path: str | Path, config: dict | None = None, manifest_hash: str | None = None,
) -> dict:
    """Run VAD and gap analysis on one cached 16 kHz WAV; returns the JSON record.

    The WAV is snapshotted to a private file and the snapshot is both hashed
    and decoded, so config_hash always matches the analysed audio even if the
    cached WAV is replaced concurrently. manifest_hash is ignored (see
    features_config_hash).
    """
    from src.utils.media import load_audio

    del manifest_hash
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    snapshot, audio_digest = _snapshot_wav(audio_path)
    try:
        waveform, sample_rate = load_audio(snapshot, SAMPLE_RATE)
    finally:
        snapshot.unlink(missing_ok=True)
    digest = _settings_hash(audio_digest, cfg)
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
        "config_hash": digest,
        "speech": [[round(s, 3), round(e, 3)] for s, e in speech],
        "speech_seconds": round(sum(e - s for s, e in speech), 3),
        "gaps": [asdict(g) for g in gaps],
    }


def write_features(cache_dir: str | Path, features: dict) -> Path:
    """Write atomically through a staging file unique to this call.

    Concurrent writers each stage their own file and publish it with one
    rename, so the published file is always one writer's complete record and
    a writer can never modify a file another writer already published.
    """
    path = Path(cache_dir) / FEATURES_NAME
    staging, handle = _private_file(
        prefix=f".{FEATURES_NAME}.", suffix=".tmp", dir=path.parent, mode="w",
    )
    try:
        with handle:
            handle.write(json.dumps(features, indent=2))
        staging.replace(path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return path


def features_are_current(features: dict | None, expected_hash: str) -> bool:
    """True only for features stamped with the expected config hash."""
    return bool(features) and features.get("config_hash") == expected_hash


def load_features(cache_dir: str | Path) -> dict | None:
    """Read cached features, or None if absent or unreadable.

    A truncated or corrupt file reads as missing rather than raising, so one
    broken cache cannot abort a batch. This does not check staleness or
    structure; use load_current_features for anything that builds moments or
    word limits.
    """
    path = Path(cache_dir) / FEATURES_NAME
    try:
        features = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return features if isinstance(features, dict) else None


def _is_finite_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # a JSON integer too large for a float
        return False


_GAP_FIELDS = frozenset(f.name for f in fields(Gap))


def malformed_features(features) -> str | None:
    """The first structural defect that would break a consumer, or None.

    speech must be a list of [start, end] pairs and gaps a list of records
    with exactly Gap's fields, all finite numbers with 0 <= start <= end.
    Same rules as cache_checks.malformed_audio_features (#18), so the moment
    builder and the selector evaluation reject the same files.
    """
    if not isinstance(features, dict):
        return "features are not an object"
    speech, gaps = features.get("speech"), features.get("gaps")
    if not isinstance(speech, list):
        return "speech is not a list"
    if not isinstance(gaps, list):
        return "gaps is not a list"
    for index, pair in enumerate(speech):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return f"speech[{index}] is not a [start, end] pair"
        start, end = pair
        if not (_is_finite_number(start) and _is_finite_number(end)) or not 0 <= start <= end:
            return f"speech[{index}] needs finite numbers with 0 <= start <= end"
    for index, gap in enumerate(gaps):
        if not isinstance(gap, dict) or set(gap) != _GAP_FIELDS:
            return f"gaps[{index}] must be a record with exactly {sorted(_GAP_FIELDS)}"
        if not all(_is_finite_number(gap[name]) for name in _GAP_FIELDS):
            return f"gaps[{index}] has a non-numeric or non-finite value"
        if not 0 <= gap["start"] <= gap["end"]:
            return f"gaps[{index}] needs 0 <= start <= end"
    return None


class InvalidAudioPath(ValueError):
    """The manifest's audio_path is neither null/empty nor a string."""


def resolve_audio_path(manifest: dict, cache_dir: str | Path) -> Path | None:
    """The video's WAV, or None for a video without an audio stream.

    The cache dir owns its WAV: preprocessing always writes it there as
    audio_16k.wav, so that file is used whenever it exists. A recorded path is
    never resolved against the current working directory, where a relative
    path could name another video's WAV. Only an absolute recorded path (a
    custom location) is used when the local file is absent; otherwise the
    local path is returned and callers report it as missing.

    Raises InvalidAudioPath for a non-string audio_path (list, number, object,
    boolean), which callers report per video instead of aborting a batch.
    """
    audio_path = manifest.get("audio_path")
    if audio_path is None or audio_path == "":
        return None
    if not isinstance(audio_path, str):
        raise InvalidAudioPath(f"audio_path must be a string or null, got {type(audio_path).__name__}")
    recorded = Path(audio_path)
    local = Path(cache_dir) / (recorded.name or "audio_16k.wav")
    if local.exists():
        return local
    if recorded.is_absolute() and recorded.exists():
        return recorded
    return local


def owned_media_manifest(manifest: dict, cache_dir: str | Path) -> dict:
    """The manifest with audio_path replaced by the WAV this cache dir owns.

    Pass this to media.cache_is_complete so the completeness check looks at the
    same WAV that is fingerprinted and analysed, independent of the working
    directory. Frames and source fields are untouched. Raises InvalidAudioPath.
    """
    audio_path = resolve_audio_path(manifest, cache_dir)
    return {**manifest, "audio_path": str(audio_path) if audio_path is not None else None}


def load_current_features(
    cache_dir: str | Path, manifest: dict, config: dict | None = None,
) -> tuple[dict | None, str | None]:
    """(features, None) only if readable, well-formed and current for this WAV.

    Otherwise (None, reason) with reason one of "invalid_audio_path",
    "no_audio_stream", "no_audio_features", "unreadable_audio_features",
    "malformed_audio_features", "missing_wav" or "stale_audio_features", so
    callers can report why instead of using outdated or broken gaps. Structure
    is checked before the hash: a correctly stamped file can still carry a
    payload that would crash a consumer.
    """
    try:
        audio_path = resolve_audio_path(manifest, cache_dir)
    except InvalidAudioPath:
        return None, "invalid_audio_path"
    if audio_path is None:
        return None, "no_audio_stream"
    if not (Path(cache_dir) / FEATURES_NAME).exists():
        return None, "no_audio_features"
    features = load_features(cache_dir)
    if features is None:
        return None, "unreadable_audio_features"
    if malformed_features(features) is not None:
        return None, "malformed_audio_features"
    try:
        expected = features_config_hash(audio_path, config)
    except OSError:
        return None, "missing_wav"
    if not features_are_current(features, expected):
        return None, "stale_audio_features"
    return features, None


def speech_of(features: dict) -> list[tuple[float, float]]:
    return [(s, e) for s, e in features["speech"]]


def gaps_of(features: dict) -> list[Gap]:
    return [Gap(**g) for g in features["gaps"]]
