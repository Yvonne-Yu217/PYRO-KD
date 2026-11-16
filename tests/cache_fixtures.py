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

from src.utils import cache_checks
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


def make_media_cache(root, video_id, audio=True, manifest_hash="m1"):
    """Create <root>/<video_id>/ with a complete, current media cache; returns its paths."""
    root = Path(root)
    source = root / "sources" / f"{video_id}.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source video bytes " + video_id.encode())
    cache = root / video_id
    (cache / "frames").mkdir(parents=True, exist_ok=True)
    frame = cache / "frames" / "000000.jpg"
    frame.write_bytes(b"jpg")
    wav = cache / "audio_16k.wav"
    if audio:
        wav.write_bytes(b"RIFF....audio")
    manifest = {
        "source": str(source.resolve()),
        "settings": {"source_fingerprint": _source_fingerprint(source)},
        "config_hash": manifest_hash,
        "audio_path": str(wav) if audio else None,
        "frames": [{"index": 0, "timestamp": 0.0, "path": str(frame)}],
    }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {"cache": cache, "source": source, "wav": wav, "frame": frame, "manifest": manifest}


def write_audio_features(cache, manifest, speech=(), gaps=(), hash_override=None):
    """Write audio_features.json stamped with the (fake) current hash of this cache."""
    stamped = hash_override or fake_features_config_hash(
        manifest["audio_path"], None, manifest.get("config_hash"))
    features = {
        "config_hash": stamped,
        "speech": [list(s) for s in speech],
        "gaps": [{"start": a, "end": b, "rms": 0.0, "onset": 0.0} for a, b in gaps],
    }
    (Path(cache) / "audio_features.json").write_text(json.dumps(features), encoding="utf-8")
    return features
