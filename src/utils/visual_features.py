"""Visual cue features for one video: shot boundaries and CLIP frame novelty.

For each cached video this stores, next to the manifest:

- visual_features.json: the frame timestamps, a novelty score per frame
  (1 - cosine similarity to the previous frame's CLIP embedding), and shot
  boundary times from PySceneDetect run on the original video (dense decode,
  not the sparse cached frames).
- visual_embeddings.<digest>.npy: the unit-normalized frame embeddings (float16),
  kept for the visual-redundancy diagnostic. The file name carries the content
  digest and the JSON record names it (with its sha256), so the pair is
  published together: the JSON is the commit record, a new embeddings
  generation is written under a new name first, and the previous pair stays
  valid until the commit succeeds. A mixed pair cannot pass validation.

The V selector ranks frames by novelty plus a bonus for the frame at each shot
boundary, then takes the top K with the shared minimum separation. Embedding
change is not an AD-importance label: camera motion raises it and subtle
actions can miss it, and a shot cut can be visually large but narratively
unimportant. The weights are untuned defaults; tune them on dev only.

Heavy dependencies (numpy, PySceneDetect, open_clip) are imported lazily so the
selection logic and its tests run without them.
"""

from __future__ import annotations

import bisect
import contextlib
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import uuid
from typing import Any, Callable, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

from src.utils import cache_checks
from src.utils.audio_features import InvalidAudioPath, owned_media_manifest
from src.utils.media import config_hash
from src.utils.selection import MIN_SEPARATION_SECONDS, select_top_k

FEATURES_NAME = "visual_features.json"
EMBEDDINGS_PREFIX = "visual_embeddings"  # generation files: visual_embeddings.<digest16>.npy
# Other writers may be mid-publish; only embeddings generations older than this are removed.
GENERATION_GRACE_SECONDS = 300.0
FEATURE_VERSION = 1

DEFAULT_CONFIG: dict[str, Any] = {
    "clip_model": "ViT-B-32",
    "clip_pretrained": "laion2b_s34b_b79k",
    "shot_detector": "ContentDetector",
    "shot_threshold": 27.0,
    "shot_min_scene_seconds": 1.0,
}

Embedder = Callable[[Sequence[str]], Any]
ShotDetector = Callable[[str, float, float], list[float]]


def novelty_curve(embeddings) -> list[float]:
    """1 - cosine similarity to the previous frame; the first frame scores 0."""
    import numpy as np

    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("embeddings must be a 2-D array (frames x dimensions)")
    if len(array) == 0:
        return []
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    unit = array / np.where(norms == 0.0, 1.0, norms)
    similarity = np.sum(unit[:-1] * unit[1:], axis=1)
    return [0.0] + [float(max(0.0, 1.0 - s)) for s in similarity]


def detect_shots(
    video_path: str | Path,
    threshold: float = DEFAULT_CONFIG["shot_threshold"],
    min_scene_seconds: float = DEFAULT_CONFIG["shot_min_scene_seconds"],
) -> list[float]:
    """Shot boundary times in seconds (start of every scene after the first).

    Runs PySceneDetect's ContentDetector on the original video. Boundaries closer
    than min_scene_seconds to the previous kept boundary are dropped.
    """
    from scenedetect import ContentDetector, detect

    scenes = detect(str(video_path), ContentDetector(threshold=threshold))
    boundaries: list[float] = []
    for start, _end in scenes[1:]:
        t = float(start.seconds if hasattr(start, "seconds") else start.get_seconds())
        if not boundaries or t - boundaries[-1] >= min_scene_seconds:
            boundaries.append(t)
    return boundaries


def clip_embedder(
    model_name: str = DEFAULT_CONFIG["clip_model"],
    pretrained: str = DEFAULT_CONFIG["clip_pretrained"],
    batch_size: int = 64,
    device: str | None = None,
) -> Embedder:
    """Build an embed(paths) -> (frames x dims) array function around OpenCLIP.

    Reuses the team's VisualSelector (frame preprocessing and normalization).
    The model loads once, when this function is called.
    """
    import numpy as np
    from PIL import Image

    from src.utils.visual_selector import VisualSelector

    selector = VisualSelector(model_name, pretrained, device)

    def embed(paths: Sequence[str]):
        chunks = []
        for start in range(0, len(paths), batch_size):
            frames = [Image.open(p).convert("RGB") for p in paths[start:start + batch_size]]
            chunks.append(selector.extract_frame_embeddings(frames).float().cpu().numpy())
        return np.concatenate(chunks) if chunks else np.zeros((0, 0))

    return embed


def features_hash(manifest: dict[str, Any], config: dict[str, Any]) -> str:
    """Identity of the inputs: feature version, config, and the frame cache they came from."""
    return config_hash({
        "version": FEATURE_VERSION,
        "config": config,
        "manifest": manifest.get("config_hash"),
    })


def is_fresh(features: dict | None, manifest: dict[str, Any], config: dict[str, Any] | None = None) -> bool:
    """True if cached features were computed from this manifest with this config."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    return bool(features) and features.get("config_hash") == features_hash(manifest, cfg)


def compute_visual_features(
    manifest: dict[str, Any],
    embed: Embedder,
    detect: ShotDetector | None = None,
    config: dict[str, Any] | None = None,
):
    """Return (features record, unit-normalized embeddings) for one cached video."""
    import numpy as np

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    frames = manifest.get("frames") or []
    if not frames:
        raise ValueError("manifest has no cached frames")
    embeddings = np.asarray(embed([f["path"] for f in frames]), dtype=np.float64)
    if embeddings.ndim != 2 or len(embeddings) != len(frames):
        raise ValueError(f"expected {len(frames)} embeddings, got shape {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0.0, 1.0, norms)
    shots = (detect or detect_shots)(
        manifest["source"], cfg["shot_threshold"], cfg["shot_min_scene_seconds"])
    features = {
        "version": FEATURE_VERSION,
        "config": cfg,
        "config_hash": features_hash(manifest, cfg),
        "num_frames": len(frames),
        "embedding_dim": int(embeddings.shape[1]),
        "frame_times": [float(f["timestamp"]) for f in frames],
        "novelty": novelty_curve(embeddings),
        "shots": [round(float(t), 3) for t in shots],
    }
    return features, embeddings


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # a JSON integer too large for a float, e.g. 10**400
        return False


def malformed_visual_features(features: Any) -> str | None:
    """A description of the first structural defect that would break a consumer, or None.

    frame_times must be finite, non-negative and strictly increasing; novelty a list
    of finite non-negative numbers of the same length; shots a list of finite
    non-negative numbers. Wrong types (for example a null novelty) are defects.
    """
    if not isinstance(features, dict):
        return "features are not an object"
    for name in ("frame_times", "novelty", "shots"):
        if not isinstance(features.get(name), list):
            return f"{name} is not a list"
    times, novelty, shots = features["frame_times"], features["novelty"], features["shots"]
    if len(times) != len(novelty):
        return "frame_times and novelty differ in length"
    if not all(_is_number(t) and t >= 0 for t in times):
        return "frame_times must be finite non-negative numbers"
    if any(b <= a for a, b in zip(times, times[1:])):
        return "frame_times must be strictly increasing"
    if not all(_is_number(v) and v >= 0 for v in novelty):
        return "novelty must be finite non-negative numbers"
    if not all(_is_number(t) and t >= 0 for t in shots):
        return "shots must be finite non-negative numbers"
    return None


def features_match_manifest(features: dict, manifest: dict[str, Any], tolerance: float = 1e-6) -> bool:
    """True if the feature rows line up one-to-one with the manifest's cached frames.

    A matching hash only says which inputs the features were computed from; this
    checks the rows themselves, so a record whose timeline was edited, truncated or
    swapped cannot change which frames a moment is built from.
    """
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != len(features["frame_times"]):
        return False
    try:
        return all(
            abs(float(frame["timestamp"]) - t) <= tolerance
            for frame, t in zip(frames, features["frame_times"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _unique_tmp(directory: Path, name: str) -> Path:
    """A staging path no other writer will use (concurrent cache jobs are not hypothetical)."""
    return directory / f".{name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"


def _generation_files(cache_dir: Path) -> list[Path]:
    return [p for p in cache_dir.glob(f"{EMBEDDINGS_PREFIX}.*.npy") if p.is_file()]


LOCK_NAME = ".visual_features.lock"


@contextlib.contextmanager
def cache_lock(cache_dir: Path):
    """Exclusive advisory lock on one video's visual cache (blocks until it is free).

    flock is released when the descriptor closes, including if the process dies, so a
    killed job cannot leave the cache locked. The lock file itself is left in place:
    removing it would let two writers lock different files. Where flock is unavailable
    (non-POSIX) this does nothing and concurrent writers are not serialized.
    """
    if fcntl is None:
        yield
        return
    with open(cache_dir / LOCK_NAME, "a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _generation_intact(path: Path, digest: str) -> bool:
    """True if the file exists and its bytes hash to the digest it is named for."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() == digest
    except OSError:
        return False


def write_features(cache_dir: str | Path, features: dict, embeddings) -> Path:
    """Publish the features record and its embeddings as one unit.

    1. Stage the embeddings under a unique temporary name, then publish them as a
       new immutable generation named by content digest.
    2. Stage the JSON record (which names that generation and its sha256), then
       commit it with an atomic rename. Until this succeeds the previous JSON and
       the previous embeddings generation are untouched and still a valid pair.
    3. After the commit, remove older generations past a grace period, never the one the
       committed record names.
    Steps 1-3 run under a per-cache lock (see cache_lock), so writers and collection
    are serialized.
    On any failure before the commit, staging files are removed and the previous pair
    stays valid; the published generation is kept (another writer may have committed
    a record that references the same content-named file) and collected later.
    An existing file of that name is verified against the digest and rewritten if its
    bytes are wrong, so repeating a failed run repairs the cache. The input record is not
    modified.
    """
    import numpy as np

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    array = np.asarray(embeddings, dtype=np.float16)
    problem = malformed_visual_features(features)
    if problem:
        raise ValueError(f"refusing to write malformed visual features: {problem}")
    if array.ndim != 2 or len(array) != len(features["frame_times"]) or not np.isfinite(array).all():
        raise ValueError(
            f"embeddings must be a finite ({len(features['frame_times'])} frames x dims) matrix, "
            f"got shape {array.shape}")
    buffer = io.BytesIO()
    np.save(buffer, array)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    name = f"{EMBEDDINGS_PREFIX}.{digest[:16]}.npy"
    generation = cache_dir / name
    record = {**features, "embeddings": {"file": name, "sha256": digest, "shape": list(array.shape)}}
    path = cache_dir / FEATURES_NAME
    with cache_lock(cache_dir):
        # Verification/adoption of the generation, the JSON commit and collection all run
        # under one per-cache lock, so no writer can collect a generation that another is
        # adopting or has just committed (a file's age says nothing about that).
        staged: list[Path] = []
        try:
            if not _generation_intact(generation, digest):
                # Absent, or present with different bytes (a corrupt or truncated earlier
                # write): publish the correct bytes atomically over it.
                staging = _unique_tmp(cache_dir, name)
                staged.append(staging)
                staging.write_bytes(data)
                staging.replace(generation)
            record_tmp = _unique_tmp(cache_dir, FEATURES_NAME)
            staged.append(record_tmp)
            record_tmp.write_text(json.dumps(record, indent=2), encoding="utf-8")
            record_tmp.replace(path)  # the commit
        except BaseException:
            # Only staging files are removed; the immutable generation stays and an
            # unreferenced one is collected after a later commit.
            for leftover in staged:
                leftover.unlink(missing_ok=True)
            raise
        cutoff = time.time() - GENERATION_GRACE_SECONDS
        # Keep what the committed record names (re-read under the lock, so it is current)
        # and this call's generation; remove other generations past the grace period.
        keep = {generation.name}
        committed, _ = _read_features(cache_dir)
        reference = committed.get("embeddings") if committed else None
        if isinstance(reference, dict) and isinstance(reference.get("file"), str):
            keep.add(reference["file"])
        for other in _generation_files(cache_dir):
            try:
                if other.name not in keep and other.stat().st_mtime <= cutoff:
                    other.unlink()
            except OSError:
                pass
    return path


NO_VISUAL_FEATURES = "no_visual_features"
UNREADABLE_VISUAL_FEATURES = "unreadable_visual_features"
MALFORMED_VISUAL_FEATURES = "malformed_visual_features"
STALE_VISUAL_FEATURES = "stale_visual_features"
INCONSISTENT_VISUAL_FEATURES = "inconsistent_visual_features"


def _read_features(cache_dir: str | Path) -> tuple[dict | None, str | None]:
    """The features record, or (None, reason): absent, unreadable, or structurally malformed."""
    path = Path(cache_dir) / FEATURES_NAME
    if not path.is_file():
        return None, NO_VISUAL_FEATURES
    try:
        features = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, UNREADABLE_VISUAL_FEATURES
    if malformed_visual_features(features) is not None:
        return None, MALFORMED_VISUAL_FEATURES
    return features, None


def load_features(cache_dir: str | Path) -> dict | None:
    """Cached features, or None if absent, unreadable, or structurally malformed (never raises)."""
    return _read_features(cache_dir)[0]


def _check_current(
    cache_dir: str | Path, manifest: dict[str, Any], config: dict[str, Any] | None,
) -> tuple[dict | None, str | None]:
    features, reason = _read_features(cache_dir)
    if reason:
        return None, reason
    if not is_fresh(features, manifest, config):
        return None, STALE_VISUAL_FEATURES
    if not features_match_manifest(features, manifest):
        return None, INCONSISTENT_VISUAL_FEATURES
    return features, None


def load_current_features(
    cache_dir: str | Path,
    manifest: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict | None:
    """Cached features only if they are well-formed and still match this video.

    Valid means: structurally well-formed, computed from this manifest and config,
    and one row per cached frame at the manifest's timestamps. A changed config,
    re-extracted frames, a new version, or an edited timeline is treated as missing
    instead of silently reused. With no manifest passed it reads
    <cache_dir>/manifest.json; an unreadable manifest means the features cannot be
    verified and count as missing. Never raises.
    """
    if manifest is None:
        try:
            manifest = json.loads((Path(cache_dir) / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
    if not isinstance(manifest, dict):
        return None
    return _check_current(cache_dir, manifest, config)[0]


def media_cache_problem(cache_dir: str | Path, manifest: dict[str, Any]) -> str | None:
    """Check source and frames using the same WAV ownership rule as audio consumers."""
    try:
        owned = owned_media_manifest(manifest, cache_dir)
    except InvalidAudioPath:
        return cache_checks.MEDIA_INCOMPLETE
    return cache_checks.media_cache_problem(owned)


def load_validated_features(cache_dir: str | Path) -> tuple[dict | None, str | None]:
    """Visual features for one cached video, validated; (None, reason) if unusable.

    Same contract as cache_checks.load_audio_features: reading never raises, and a
    reason code says why the features cannot be trusted. The media cache must be
    complete and current (so a failed rebuild that left a complete cache for an older
    source is caught), and the features well-formed, computed from this manifest and
    config, and aligned with the cached frames.
    """
    manifest, reason = cache_checks.read_manifest(cache_dir)
    if manifest is None:
        return None, reason
    problem = media_cache_problem(cache_dir, manifest)
    if problem:
        return None, problem
    return _check_current(cache_dir, manifest, None)


def load_embeddings(cache_dir: str | Path, features: dict | None = None):
    """The embeddings generation the features record commits to, as float32, or None.

    Pass the already-validated features record when you have one, so the matrix
    returned is the one that record names. None (never an exception) if the record
    names no embeddings, the file is missing or its digest does not match, or the
    array is not a finite (frames x dims) matrix with one row per frame.
    """
    import numpy as np

    if features is None:
        features = load_features(cache_dir)
        if features is None:
            return None
    reference = features.get("embeddings")
    if not isinstance(reference, dict):
        return None
    name, digest = reference.get("file"), reference.get("sha256")
    if (
        not isinstance(name, str) or not isinstance(digest, str)
        or name != Path(name).name or not name.startswith(f"{EMBEDDINGS_PREFIX}.")
    ):
        return None
    try:
        data = (Path(cache_dir) / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            return None
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception:  # EOF, BadZipFile, ValueError, ...: any undecodable payload is unusable
        return None
    if not isinstance(array, np.ndarray):  # np.load also returns an NpzFile archive handle
        close = getattr(array, "close", None)
        if callable(close):
            close()
        return None
    if (
        array.ndim != 2 or array.shape[1] < 1 or len(array) != len(features["frame_times"])
        or not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all()
    ):
        return None
    return array.astype(np.float32)


def mean_pairwise_similarity(embeddings, frame_times: Sequence[float], times: Sequence[float]) -> float | None:
    """Mean pairwise cosine similarity of the frames nearest to the selected times.

    Higher means more redundant selections. None for fewer than two selections or
    when the embeddings are not a finite 2-D matrix lining up with the frame
    timestamps (an optional diagnostic never raises).
    """
    import numpy as np

    if embeddings is None or len(times) < 2:
        return None
    try:
        vectors = np.asarray(embeddings, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if vectors.ndim != 2 or len(vectors) != len(frame_times) or not np.isfinite(vectors).all():
        return None
    chosen = []
    for t in times:
        i = bisect.bisect_left(frame_times, t)
        candidates = [j for j in (i - 1, i) if 0 <= j < len(frame_times)]
        chosen.append(min(candidates, key=lambda j: abs(frame_times[j] - t)))
    vectors = vectors[chosen]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0.0, 1.0, norms)
    similarity = vectors @ vectors.T
    upper = similarity[np.triu_indices(len(chosen), k=1)]
    return float(upper.mean())


@dataclass(frozen=True)
class VisualCandidate:
    time: float
    score: float
    novelty: float
    shot_boundary: bool


class VisualCueSelector:
    """V: frames ranked by CLIP novelty plus a shot-boundary bonus, top K with separation.

    score(t) = novelty_weight * novelty(t) / max novelty + shot_weight * [t is the
    cached frame nearest a shot boundary]. Novelty is scaled by the video's own
    maximum so videos are comparable; a video with no change at all scores 0
    everywhere, and ties go to earlier times.
    """

    def __init__(self, novelty_weight: float = 1.0, shot_weight: float = 0.5):
        self.novelty_weight = novelty_weight
        self.shot_weight = shot_weight

    def candidates(self, features: dict) -> list[VisualCandidate]:
        times = [float(t) for t in features["frame_times"]]
        novelty = [float(v) for v in features["novelty"]]
        if not times:
            return []
        peak = max(novelty) or 1.0
        boundary_frames: set[int] = set()
        for shot in features["shots"]:
            i = bisect.bisect_left(times, float(shot))
            near = [j for j in (i - 1, i) if 0 <= j < len(times)]
            boundary_frames.add(min(near, key=lambda j: abs(times[j] - float(shot))))
        return [
            VisualCandidate(
                time=times[i],
                score=self.novelty_weight * novelty[i] / peak
                + (self.shot_weight if i in boundary_frames else 0.0),
                novelty=novelty[i],
                shot_boundary=i in boundary_frames,
            )
            for i in range(len(times))
        ]

    def select(
        self,
        features: dict,
        k: int,
        min_separation: float = MIN_SEPARATION_SECONDS,
    ) -> list[VisualCandidate]:
        """Up to K moments in chronological order; fewer only if separation leaves no room."""
        return select_top_k(
            self.candidates(features), k, min_separation,
            time_of=lambda c: c.time, score_of=lambda c: c.score,
        )
