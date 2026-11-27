"""All cue streams for one video on one time grid.

A CueBundle holds salience tracks in [0, 1] sampled every `hop` seconds:

- vn: visual novelty (CLIP change to the previous frame, robust-normalized)
  plus a shot-boundary bonus, as in V;
- nse[component]: non-speech event salience per AudioSet superclass (onset
  rises), plus clap_change, loudness and onset (robust-normalized);
- sc[component]: deictic, semantic_shift, speaker_change, long_silence;

and the feasibility inputs (VAD speech and usable gaps). A stream that is not
available is absent from the dict and named in `missing` with the reason, so a
selector can never mistake a missing stream for a quiet one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from src.audio import salience
from src.audio.speech_cues import long_silence_track
from src.audio.taxonomy import SUPERCLASSES

DEFAULT_BUNDLE_CONFIG = {
    "hop_seconds": 0.5,
    "shot_weight": 0.5,
    "event_spread_seconds": 1.0,
    "speech_cue_spread_seconds": 1.5,
    "loudness_saturation_db": 12.0,
    "long_silence_seconds": 8.0,
    "normalize_percentile": 100.0,  # max; a lower percentile saturates sparse peak tracks
}


@dataclass
class CueBundle:
    duration: float
    hop: float
    grid: list[float]
    vn: list[float] | None = None
    frame_times: list[float] | None = None
    embeddings: Any = None  # (frames x dims) unit float32, aligned with frame_times
    shots: list[float] = field(default_factory=list)
    nse: dict[str, list[float]] = field(default_factory=dict)
    sc: dict[str, list[float]] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    speech_cues: list[dict] = field(default_factory=list)
    speech: list[tuple[float, float]] | None = None
    gaps: list[tuple[float, float]] | None = None
    missing: dict[str, str] = field(default_factory=dict)

    def index(self, t: float) -> int:
        return min(len(self.grid) - 1, max(0, int(round(t / self.hop))))

    def window_max(self, track: list[float] | None, t: float, half_width: float) -> float:
        if not track:
            return 0.0
        lo, hi = self.index(t - half_width), self.index(t + half_width)
        return max(track[lo:hi + 1])

    def audio_change(self) -> list[float]:
        """Max over non-speech salience components (0 where none is available)."""
        if not self.nse:
            return [0.0] * len(self.grid)
        return [max(values) for values in zip(*self.nse.values())]

    def embedding_near(self, t: float):
        if self.embeddings is None or not self.frame_times:
            return None
        import bisect

        i = bisect.bisect_left(self.frame_times, t)
        near = [j for j in (i - 1, i) if 0 <= j < len(self.frame_times)]
        return self.embeddings[min(near, key=lambda j: abs(self.frame_times[j] - t))]


def make_grid(duration: float, hop: float) -> list[float]:
    if not (math.isfinite(duration) and duration > 0):
        raise ValueError("duration must be positive and finite")
    if hop <= 0:
        raise ValueError("hop must be positive")
    return [round(i * hop, 6) for i in range(int(math.floor(duration / hop)) + 1)]


def build_cue_bundle(
    duration: float,
    audio_features: dict | None = None,
    audio_cues: dict | None = None,
    visual_features: dict | None = None,
    embeddings=None,
    config: dict | None = None,
    missing_reasons: dict[str, str] | None = None,
) -> CueBundle:
    """Assemble a CueBundle from validated cache records (each may be None)."""
    cfg = {**DEFAULT_BUNDLE_CONFIG, **(config or {})}
    grid = make_grid(duration, cfg["hop_seconds"])
    bundle = CueBundle(duration=duration, hop=cfg["hop_seconds"], grid=grid, missing=dict(missing_reasons or {}))
    pct = cfg["normalize_percentile"]

    if visual_features is not None:
        novelty = salience.resample_to_grid(visual_features["frame_times"], visual_features["novelty"], grid, "max")
        shots = [float(s) for s in visual_features.get("shots", [])]
        shot_track = salience.events_to_track(
            [{"time": s, "strength": 1.0} for s in shots], grid, cfg["hop_seconds"],
        )
        bundle.vn = [min(1.0, n + cfg["shot_weight"] * s) for n, s in zip(salience.normalize(novelty, pct), shot_track)]
        bundle.shots = shots
        bundle.frame_times = [float(t) for t in visual_features["frame_times"]]
        if embeddings is not None and len(embeddings) == len(bundle.frame_times):
            bundle.embeddings = embeddings
        else:
            bundle.missing.setdefault("embeddings", "no embeddings for running memory / MMR")
    else:
        bundle.missing.setdefault("visual", "no visual features")
        bundle.missing.setdefault("embeddings", "no visual features")

    if audio_features is not None:
        bundle.speech = [(float(s), float(e)) for s, e in audio_features["speech"]]
        bundle.gaps = [(float(g["start"]), float(g["end"])) for g in audio_features["gaps"]]
        bundle.sc["long_silence"] = long_silence_track(bundle.gaps, grid, cfg["long_silence_seconds"])
    else:
        bundle.missing.setdefault("vad", "no VAD features")

    if audio_cues is not None:
        streams = audio_cues.get("streams", {})
        events = list(audio_cues.get("events", []))
        bundle.events = events
        if (str(streams.get("tagger", "")).startswith("ok")
                or str(streams.get("whisper_tags", "")).startswith("ok")
                or any(e.get("source") == "whisper_tag" for e in events)):
            for superclass in SUPERCLASSES:
                bundle.nse[superclass] = [
                    min(1.0, v) for v in salience.events_to_track(events, grid, cfg["event_spread_seconds"], superclass)
                ]
        else:
            bundle.missing.setdefault("tagger", streams.get("tagger", "missing"))
        clap = audio_cues.get("clap")
        if clap:
            bundle.nse["clap_change"] = salience.normalize(
                salience.resample_to_grid(clap["times"], clap["change"], grid, "max"), pct)
        else:
            bundle.missing.setdefault("clap", streams.get("clap", "missing"))
        acoustic = audio_cues.get("acoustic")
        if acoustic:
            jump = salience.resample_to_grid(acoustic["times"], acoustic["loudness_jump"], grid, "max")
            bundle.nse["loudness"] = [min(1.0, v / cfg["loudness_saturation_db"]) for v in jump]
            if any(acoustic["onset_strength"]):
                bundle.nse["onset"] = salience.normalize(
                    salience.resample_to_grid(acoustic["times"], acoustic["onset_strength"], grid, "max"), pct)
        else:
            bundle.missing.setdefault("acoustic", streams.get("acoustic", "missing"))
        cues = list(audio_cues.get("speech_cues", []))
        bundle.speech_cues = cues
        for kind, stream in (("deictic", "asr"), ("semantic_shift", "semantic"), ("speaker_change", "diarization")):
            if str(streams.get(stream, "")).startswith("ok"):
                bundle.sc[kind] = [
                    min(1.0, v) for v in salience.events_to_track(
                        [c for c in cues if c["kind"] == kind], grid, cfg["speech_cue_spread_seconds"])
                ]
            else:
                bundle.missing.setdefault(kind, streams.get(stream, "missing"))
    else:
        bundle.missing.setdefault("audio_cues", "no audio_cues.json")
    return bundle
