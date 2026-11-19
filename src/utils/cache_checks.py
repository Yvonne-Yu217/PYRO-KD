"""Decide whether cached derived data may be trusted before a consumer uses it.

Every function returns a (value, reason) pair: the value is usable only when
reason is None. Otherwise the reason is a short, stable code that consumers
count and report per video, so stale or unverifiable features are never scored
silently. Reading is best effort: unreadable files become reasons, not
exceptions, so one broken cache cannot abort an unrelated evaluation.

Order of checks for audio features: manifest readable, media cache complete and
current (media.cache_is_complete / cache_is_current, which catch a rebuild that
failed and left an older complete cache), an audio stream exists, the features
file is readable and every speech pair and gap record is well-formed, and only
then its fingerprint matches the current WAV and manifest. Structure is checked
before the fingerprint on purpose: a correctly stamped file can still carry a
payload that would crash a consumer, and that must be reported whether or not
the fingerprint helpers are available.

The WAV, its fingerprint and the structural rules come from
src/utils/audio_features.py, the same helpers build_moments uses, so the
selector evaluation and the moment builder accept and reject the same caches.
In particular the WAV is the one the video's cache dir owns
(audio_features.resolve_audio_path), never a recorded relative path resolved
against the current working directory, and that same resolved WAV is used for
the completeness check and the fingerprint. If the fingerprint helpers are
unavailable, features are reported as "unverifiable_audio_features" rather
than trusted on faith.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.utils.media import MANIFEST_NAME, cache_is_complete, cache_is_current

from src.utils.audio_features import (
    features_are_current, features_config_hash, malformed_features, resolve_audio_path,
)

AUDIO_FEATURES_NAME = "audio_features.json"

NO_MANIFEST = "no_manifest"
UNREADABLE_MANIFEST = "unreadable_manifest"
MEDIA_INCOMPLETE = "media_cache_incomplete"
MEDIA_STALE = "media_cache_stale"
NO_AUDIO_STREAM = "no_audio_stream"
NO_AUDIO_FEATURES = "no_audio_features"
UNREADABLE_AUDIO_FEATURES = "unreadable_audio_features"
MALFORMED_AUDIO_FEATURES = "malformed_audio_features"
MISSING_WAV = "missing_wav"
UNVERIFIABLE_AUDIO_FEATURES = "unverifiable_audio_features"
STALE_AUDIO_FEATURES = "stale_audio_features"


def read_manifest(cache_dir: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    """The video's media manifest, or (None, reason) if absent or unreadable."""
    path = Path(cache_dir) / MANIFEST_NAME
    if not path.is_file():
        return None, NO_MANIFEST
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, UNREADABLE_MANIFEST
    if not isinstance(manifest, dict):
        return None, UNREADABLE_MANIFEST
    return manifest, None


def media_cache_problem(manifest: dict[str, Any]) -> str | None:
    """Why the cached frames/audio cannot be trusted for this source, or None."""
    if not cache_is_complete(manifest):
        return MEDIA_INCOMPLETE
    if not cache_is_current(manifest):
        return MEDIA_STALE
    return None


def malformed_audio_features(features: Any) -> str | None:
    """A description of the first structural defect that would break a consumer, or None.

    Delegates to audio_features.malformed_features (speech [start, end] pairs and
    exact Gap records, finite, 0 <= start <= end), so there is one validator.
    """
    return malformed_features(features)


def load_audio_features(cache_dir: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    """Audio features for one cached video, validated; (None, reason) if unusable."""
    manifest, reason = read_manifest(cache_dir)
    if manifest is None:
        return None, reason
    # Check completeness against the WAV this cache owns, the same file that is
    # fingerprinted below; the frame and source checks are unchanged.
    audio_path = resolve_audio_path(manifest, cache_dir)
    problem = media_cache_problem(
        {**manifest, "audio_path": str(audio_path) if audio_path is not None else None},
    )
    if problem:
        return None, problem
    if audio_path is None:
        return None, NO_AUDIO_STREAM
    path = Path(cache_dir) / AUDIO_FEATURES_NAME
    if not path.is_file():
        return None, NO_AUDIO_FEATURES
    try:
        features = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, UNREADABLE_AUDIO_FEATURES
    if malformed_audio_features(features) is not None:
        return None, MALFORMED_AUDIO_FEATURES
    if features_config_hash is None or features_are_current is None:
        return None, UNVERIFIABLE_AUDIO_FEATURES
    try:
        expected = features_config_hash(audio_path, None, manifest.get("config_hash"))
    except OSError:
        return None, MISSING_WAV
    if not features_are_current(features, expected):
        return None, STALE_AUDIO_FEATURES
    return features, None
