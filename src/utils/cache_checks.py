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

The fingerprint helpers come from the audio-validation work (PR #16). Until
that is merged they are missing and audio features are reported as
"unverifiable_audio_features" rather than trusted on faith; once it is merged
the same code verifies them with no change here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.utils.media import MANIFEST_NAME, cache_is_complete, cache_is_current

try:  # provided by the audio-validation change (#16)
    from src.utils.audio_features import features_are_current, features_config_hash
except ImportError:  # pragma: no cover - exercised by patching in tests
    features_are_current = features_config_hash = None

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


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # a JSON integer too large for a float, e.g. 10**400
        return False


_GAP_FIELDS = {"start", "end", "rms", "onset"}


def malformed_audio_features(features: Any) -> str | None:
    """A description of the first structural defect that would break a consumer, or None.

    Every speech entry must be a [start, end] pair of finite numbers with
    0 <= start <= end, and every gap a record with exactly the fields start, end,
    rms and onset, finite, with 0 <= start <= end. (Gap(**record) is what
    consumers do, so unknown or missing fields would raise there.)
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
        if not (_is_number(start) and _is_number(end)) or not 0 <= start <= end:
            return f"speech[{index}] needs finite numbers with 0 <= start <= end"
    for index, gap in enumerate(gaps):
        if not isinstance(gap, dict) or set(gap) != _GAP_FIELDS:
            return f"gaps[{index}] must be a record with exactly {sorted(_GAP_FIELDS)}"
        if not all(_is_number(gap[name]) for name in _GAP_FIELDS):
            return f"gaps[{index}] has a non-numeric or non-finite value"
        if not 0 <= gap["start"] <= gap["end"]:
            return f"gaps[{index}] needs 0 <= start <= end"
    return None


def load_audio_features(cache_dir: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    """Audio features for one cached video, validated; (None, reason) if unusable."""
    manifest, reason = read_manifest(cache_dir)
    if manifest is None:
        return None, reason
    problem = media_cache_problem(manifest)
    if problem:
        return None, problem
    audio_path = manifest.get("audio_path")
    if not audio_path:
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
