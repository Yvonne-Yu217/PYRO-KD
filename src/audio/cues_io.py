"""audio_cues.json: the cached output of the three-stream audio stack.

One file per video, beside audio_features.json (VAD) in the video's cache dir.
Schema (version 1):

    {
      "version": 1, "config": {...}, "config_hash": "...", "duration": float,
      "streams": {"asr": "ok" | "missing: <why>", "tagger": ..., "clap": ...,
                  "acoustic": ..., "semantic": ..., "diarization": ...},
      "asr": {"source": "whisper-vad-gated" | "whisper-ungated" | "dataset", "segments": [...]} | null,
      "tagger": {"model": str, "times": [...], "labels": [...], "posteriors": [[...]]} | null,
      "clap": {"model": str, "times": [...], "change": [...]} | null,
      "acoustic": {"times": [...], "loudness_db": [...], "loudness_jump": [...], "onset_strength": [...]} | null,
      "events": [AudioEvent...],         # NSE onsets (tagger and, for ablation 3, whisper tags)
      "speech_cues": [SpeechCue...]
    }

A missing stream is recorded with its reason, never silently treated as an
absence of events. The config_hash covers the config, the WAV's SHA-256, the
VAD features' own hash (long-silence cues and ASR gating depend on it), the
backend names and the digest of any dataset transcript used, so changing any
of them invalidates the file. Writes are atomic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

from src.audio import asr as asr_mod
from src.audio import salience, speech_cues, tagging
from src.audio.taxonomy import TAXONOMY_VERSION

CUES_NAME = "audio_cues.json"
CUES_VERSION = 1
DEFAULT_CUES_CONFIG: dict[str, Any] = {
    "asr": dict(asr_mod.DEFAULT_ASR_CONFIG),
    "tagger": dict(tagging.DEFAULT_TAGGER_CONFIG),
    "clap": dict(tagging.DEFAULT_CLAP_CONFIG),
    "nse": dict(salience.DEFAULT_NSE_CONFIG),
    "sc": dict(speech_cues.DEFAULT_SC_CONFIG),
    "acoustic_hop_seconds": 0.5,
    "taxonomy": TAXONOMY_VERSION,
    "superclass_overrides": {},
}

NO_CUES = "no_audio_cues"
UNREADABLE_CUES = "unreadable_audio_cues"
MALFORMED_CUES = "malformed_audio_cues"
STALE_CUES = "stale_audio_cues"


@dataclass
class CueBackends:
    """Injected models; None means that stream is skipped and reported missing."""

    transcriber: Callable | None = None
    transcriber_name: str | None = None
    tagger: Callable | None = None
    tagger_labels: list[str] | None = None
    tagger_name: str | None = None
    clap: Callable | None = None
    clap_name: str | None = None
    text_encoder: Callable | None = None
    text_encoder_name: str | None = None
    diarization_turns: list[dict] | None = None
    diarization_name: str | None = None
    dataset_transcript: list[dict] | None = None  # e.g. MTG-AD's released ASR
    acoustic: bool = True
    notes: dict = field(default_factory=dict)

    def identity(self) -> dict:
        transcript_digest = None
        if self.dataset_transcript is not None:
            transcript_digest = hashlib.sha256(
                json.dumps(self.dataset_transcript, sort_keys=True).encode()
            ).hexdigest()[:16]
        return {
            "transcriber": self.transcriber_name if self.transcriber else None,
            "tagger": self.tagger_name if self.tagger else None,
            "clap": self.clap_name if self.clap else None,
            "text_encoder": self.text_encoder_name if self.text_encoder else None,
            "diarization": self.diarization_name if self.diarization_turns is not None else None,
            "dataset_transcript": transcript_digest,
            "acoustic": self.acoustic,
        }


def merged_config(config: dict | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CUES_CONFIG))
    for key, value in (config or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    return cfg


def cues_hash(audio_sha256: str, vad_hash: str | None, config: dict, backends: CueBackends) -> str:
    from src.utils.media import config_hash

    return config_hash({
        "version": CUES_VERSION, "config": config, "audio_sha256": audio_sha256,
        "vad_hash": vad_hash, "backends": backends.identity(),
    })


def compute_audio_cues(
    audio_path: str | Path,
    vad_features: dict,
    backends: CueBackends,
    config: dict | None = None,
) -> dict:
    """Run every available stream on one cached WAV; returns the audio_cues record."""
    import numpy as np

    from src.utils.audio_features import _snapshot_wav, gaps_of, speech_of
    from src.utils.media import load_audio

    cfg = merged_config(config)
    snapshot, digest = _snapshot_wav(audio_path)
    try:
        waveform, sr = load_audio(snapshot, 16_000)
    finally:
        snapshot.unlink(missing_ok=True)
    duration = len(waveform) / sr
    streams: dict[str, str] = {}
    overrides = cfg.get("superclass_overrides") or None

    # Stream 2: speech content.
    asr_record = None
    if backends.transcriber is not None:
        gated = bool(cfg["asr"]["vad_gated"])
        regions = (
            asr_mod.vad_regions(
                speech_of(vad_features), duration, cfg["asr"]["region_pad_seconds"],
                cfg["asr"]["region_merge_gap_seconds"], cfg["asr"]["max_region_seconds"],
            ) if gated else [(0.0, duration)]
        )
        segments = asr_mod.transcribe_regions(waveform, sr, regions, backends.transcriber)
        problem = asr_mod.validate_segments(segments, duration)
        if problem:
            raise ValueError(f"ASR output invalid: {problem}")
        asr_record = {"source": "whisper-vad-gated" if gated else "whisper-ungated", "segments": segments}
        streams["asr"] = "ok"
    elif backends.dataset_transcript is not None:
        asr_record = {"source": "dataset", "segments": list(backends.dataset_transcript)}
        streams["asr"] = "ok (dataset transcript, segment-level timing)"
    else:
        streams["asr"] = "missing: no transcriber or dataset transcript"

    events: list[dict] = []
    # Stream 3: non-speech events.
    tagger_record = None
    if backends.tagger is not None:
        times, post = tagging.run_windowed(
            waveform, sr, backends.tagger, cfg["tagger"]["window_seconds"], cfg["tagger"]["hop_seconds"],
            cfg["tagger"]["batch_size"],
        )
        labels, kept = tagging.keep_labels(
            backends.tagger_labels or [], post, cfg["tagger"]["keep_top_labels"], cfg["tagger"]["keep_min_posterior"],
        )
        tagger_record = {
            "model": backends.tagger_name, "times": times, "labels": labels,
            "posteriors": np.round(kept, 4).tolist(),
        }
        events += [e.to_dict() for e in salience.class_onsets(times, kept, labels, cfg["nse"], overrides)]
        streams["tagger"] = "ok"
    else:
        streams["tagger"] = "missing: no tagger backend"
    if asr_record is not None and asr_record["source"] == "whisper-ungated":
        events += [e.to_dict() for e in asr_mod.whisper_tag_events(asr_record["segments"])]

    clap_record = None
    if backends.clap is not None:
        times, emb = tagging.run_windowed(
            waveform, sr, backends.clap, cfg["clap"]["window_seconds"], cfg["clap"]["hop_seconds"],
            cfg["clap"]["batch_size"],
        )
        clap_record = {
            "model": backends.clap_name, "times": times,
            "change": [round(v, 5) for v in salience.embedding_change(times, emb, cfg["nse"]["clap_window_seconds"])],
        }
        streams["clap"] = "ok"
    else:
        streams["clap"] = "missing: no CLAP backend"

    acoustic_record = None
    if backends.acoustic:
        hop = cfg["acoustic_hop_seconds"]
        times, levels = salience.loudness_db(waveform, sr, hop)
        try:
            _, onset = salience.onset_strength_track(waveform, sr, hop)
            onset = onset[: len(times)] + [0.0] * max(0, len(times) - len(onset))
            onset_status = "ok"
        except ImportError:
            onset, onset_status = [0.0] * len(times), "loudness only (librosa missing)"
        acoustic_record = {
            "times": times,
            "loudness_db": [round(v, 3) for v in levels],
            "loudness_jump": [round(v, 3) for v in salience.loudness_jumps(times, levels, cfg["nse"]["loudness_window_seconds"])],
            "onset_strength": [round(float(v), 4) for v in onset],
        }
        streams["acoustic"] = onset_status
    else:
        streams["acoustic"] = "missing: disabled"

    # Speech cues.
    cues: list[dict] = []
    segments = asr_record["segments"] if asr_record else []
    cues += [c.to_dict() for c in speech_cues.deictic_cues(segments)]
    if backends.text_encoder is not None and segments:
        cues += [c.to_dict() for c in speech_cues.semantic_shift_cues(
            segments, backends.text_encoder, cfg["sc"]["semantic_window"], cfg["sc"]["semantic_threshold"],
        )]
        streams["semantic"] = "ok"
    else:
        streams["semantic"] = "missing: no text encoder" if backends.text_encoder is None else "missing: no transcript"
    if backends.diarization_turns is not None:
        cues += [c.to_dict() for c in speech_cues.speaker_change_cues(backends.diarization_turns)]
        streams["diarization"] = "ok"
    else:
        streams["diarization"] = "missing: no diarization (optional in the proposal)"
    gaps = [(g.start, g.end) for g in gaps_of(vad_features)]
    cues += [c.to_dict() for c in speech_cues.long_silence_cues(gaps, cfg["sc"]["long_silence_seconds"])]
    cues.sort(key=lambda c: (c["time"], c["kind"]))

    return {
        "version": CUES_VERSION,
        "config": cfg,
        "config_hash": cues_hash(digest, vad_features.get("config_hash"), cfg, backends),
        "audio_sha256": digest,
        "vad_hash": vad_features.get("config_hash"),
        "backends": backends.identity(),
        "duration": round(duration, 3),
        "streams": streams,
        "asr": asr_record,
        "tagger": tagger_record,
        "clap": clap_record,
        "acoustic": acoustic_record,
        "events": sorted(events, key=lambda e: (e["time"], e["label"])),
        "speech_cues": cues,
    }


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def malformed_cues(record: Any) -> str | None:
    """First structural defect a consumer would trip on, or None."""
    if not isinstance(record, dict):
        return "not an object"
    if record.get("version") != CUES_VERSION:
        return "unknown version"
    if not _finite(record.get("duration")) or record["duration"] < 0:
        return "duration must be a finite non-negative number"
    if not isinstance(record.get("streams"), dict):
        return "streams is not an object"
    for name in ("events", "speech_cues"):
        rows = record.get(name)
        if not isinstance(rows, list):
            return f"{name} is not a list"
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or not _finite(row.get("time")) or not _finite(row.get("strength")):
                return f"{name}[{i}] needs finite time and strength"
            if row["time"] < 0:
                return f"{name}[{i}] has a negative time"
    for name, keys in (("tagger", ("times",)), ("clap", ("times", "change")), ("acoustic", ("times", "loudness_jump", "onset_strength"))):
        block = record.get(name)
        if block is None:
            continue
        if not isinstance(block, dict):
            return f"{name} is not an object"
        lengths = set()
        for key in keys:
            values = block.get(key)
            if not isinstance(values, list) or not all(_finite(v) for v in values):
                return f"{name}.{key} must be a list of finite numbers"
            lengths.add(len(values))
        if len(lengths) > 1:
            return f"{name} tracks differ in length"
    tagger = record.get("tagger")
    if tagger is not None:
        post = tagger.get("posteriors")
        if not isinstance(post, list) or len(post) != len(tagger["times"]):
            return "tagger.posteriors must have one row per time"
        width = len(tagger.get("labels") or [])
        if any(not isinstance(r, list) or len(r) != width or not all(_finite(v) for v in r) for r in post):
            return "tagger.posteriors rows must match labels and be finite"
    asr = record.get("asr")
    if asr is not None:
        if not isinstance(asr, dict) or not isinstance(asr.get("segments"), list):
            return "asr.segments is not a list"
    return None


def write_cues(cache_dir: str | Path, record: dict, name: str = CUES_NAME) -> Path:
    """Atomic write through a private staging file (same pattern as audio features)."""
    from src.utils.audio_features import _private_file

    path = Path(cache_dir) / name
    staging, handle = _private_file(prefix=f".{name}.", suffix=".tmp", dir=path.parent, mode="w")
    try:
        with handle:
            handle.write(json.dumps(record))
        staging.replace(path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    return path


def read_cues(cache_dir: str | Path, name: str = CUES_NAME) -> tuple[dict | None, str | None]:
    """(record, None) if readable and well-formed; otherwise (None, reason). Never raises."""
    path = Path(cache_dir) / name
    if not path.is_file():
        return None, NO_CUES
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, UNREADABLE_CUES
    if malformed_cues(record) is not None:
        return None, MALFORMED_CUES
    return record, None


def load_cues_for_selection(
    cache_dir: str | Path, audio_path: str | Path | None, vad_features: dict | None, name: str = CUES_NAME,
) -> tuple[dict | None, str | None]:
    """Consumer-side check (selectors): well-formed and computed from this exact WAV
    and VAD result. Backend identity is recorded in the file and reported, not
    re-derived, because selectors do not construct backends."""
    from src.utils.audio_features import wav_digest

    record, reason = read_cues(cache_dir, name)
    if record is None:
        return None, reason
    if audio_path is None or vad_features is None:
        return None, "no_vad_features"
    try:
        digest = wav_digest(audio_path)
    except OSError:
        return None, "missing_wav"
    if record.get("audio_sha256") != digest or record.get("vad_hash") != vad_features.get("config_hash"):
        return None, STALE_CUES
    return record, None


def load_current_cues(
    cache_dir: str | Path, audio_path: str | Path, vad_features: dict, backends: CueBackends, config: dict | None = None,
    name: str = CUES_NAME,
) -> tuple[dict | None, str | None]:
    """Cues only if well-formed and stamped for this WAV, VAD result, config and backends."""
    from src.utils.audio_features import wav_digest

    record, reason = read_cues(cache_dir, name)
    if record is None:
        return None, reason
    try:
        digest = wav_digest(audio_path)
    except OSError:
        return None, "missing_wav"
    expected = cues_hash(digest, vad_features.get("config_hash"), merged_config(config), backends)
    if record.get("config_hash") != expected:
        return None, STALE_CUES
    return record, None
