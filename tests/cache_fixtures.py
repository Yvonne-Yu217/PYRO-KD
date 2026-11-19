"""Synthetic media caches for tests of cache validation (no ffmpeg required).

make_media_cache writes a source file, a frame, a WAV and a manifest that pass
media.cache_is_complete and media.cache_is_current, the way preprocess_video
would leave them. patched_audio_helpers stands in for the fingerprint helpers
from the audio-validation work (#16): the hash covers the manifest hash and the
WAV's size, so replacing the WAV or the manifest makes stored features stale.
"""

from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.utils import cache_checks
from src.utils import visual_features as vf
from src.utils.media import _source_fingerprint


def fake_features_config_hash(audio_path, config=None, manifest_hash=None):
    return f"{manifest_hash}:{Path(audio_path).stat().st_size}"


def fake_features_are_current(features, expected_hash):
    return bool(features) and features.get("config_hash") == expected_hash


@contextmanager
def patched_audio_helpers():
    with (
        patch.object(cache_checks, "features_config_hash", fake_features_config_hash),
        patch.object(cache_checks, "features_are_current", fake_features_are_current),
    ):
        yield


def make_media_cache(root, video_id, audio=True, manifest_hash="m1", frame_times=(0.0,)):
    """Create <root>/<video_id>/ with a complete, current media cache; returns its paths.

    One frame file is written per entry of frame_times, and the manifest records them
    at those timestamps, so feature timelines can be aligned with (or deliberately
    different from) the cached frames.
    """
    root = Path(root)
    source = root / "sources" / f"{video_id}.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source video bytes " + video_id.encode())
    cache = root / video_id
    (cache / "frames").mkdir(parents=True, exist_ok=True)
    frames = []
    for index, timestamp in enumerate(frame_times):
        frame_path = cache / "frames" / f"{index:06d}.jpg"
        frame_path.write_bytes(b"jpg")
        frames.append({"index": index, "timestamp": float(timestamp), "path": str(frame_path)})
    frame = cache / "frames" / "000000.jpg"
    wav = cache / "audio_16k.wav"
    if audio:
        wav.write_bytes(b"RIFF....audio")
    manifest = {
        "source": str(source.resolve()),
        "settings": {"source_fingerprint": _source_fingerprint(source)},
        "config_hash": manifest_hash,
        "audio_path": str(wav) if audio else None,
        "frames": frames,
    }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {"cache": cache, "source": source, "wav": wav, "frame": frame, "manifest": manifest}


def make_visual_cache(root, video_id, visual, embeddings=None, audio=False, manifest_hash="m1"):
    """A complete, current media cache plus visual features published as a pair.

    visual is {"frame_times", "novelty", "shots"}; the manifest's frames sit at the same
    timestamps, and the record is stamped for this manifest, as the caching script would.
    """
    paths = make_media_cache(
        root, video_id, audio=audio, manifest_hash=manifest_hash, frame_times=visual["frame_times"])
    features = {**visual, "config": vf.DEFAULT_CONFIG,
                "config_hash": vf.features_hash(paths["manifest"], vf.DEFAULT_CONFIG)}
    count = len(visual["frame_times"])
    embeddings = np.eye(count)[:, :4] if embeddings is None else embeddings
    vf.write_features(paths["cache"], features, embeddings)
    return paths


def write_audio_features(
    cache, manifest, speech=(), gaps=(), hash_override=None, raw_speech=None, raw_gaps=None,
):
    """Write audio_features.json stamped with the (fake) current hash of this cache.

    raw_speech / raw_gaps replace the payload verbatim (for malformed-entry tests).
    """
    stamped = hash_override or fake_features_config_hash(
        manifest["audio_path"], None, manifest.get("config_hash"))
    features = {
        "config_hash": stamped,
        "speech": raw_speech if raw_speech is not None else [list(s) for s in speech],
        "gaps": raw_gaps if raw_gaps is not None else [
            {"start": a, "end": b, "rms": 0.0, "onset": 0.0} for a, b in gaps],
    }
    (Path(cache) / "audio_features.json").write_text(json.dumps(features), encoding="utf-8")
    return features
