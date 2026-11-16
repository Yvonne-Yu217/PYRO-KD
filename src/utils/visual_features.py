"""Visual cue features for one video: shot boundaries and CLIP frame novelty.

For each cached video this stores, next to the manifest:

- visual_features.json: the frame timestamps, a novelty score per frame
  (1 - cosine similarity to the previous frame's CLIP embedding), and shot
  boundary times from PySceneDetect run on the original video (dense decode,
  not the sparse cached frames).
- visual_embeddings.npy: the unit-normalized frame embeddings (float16), kept
  for the visual-redundancy diagnostic.

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
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from src.utils.media import config_hash
from src.utils.selection import MIN_SEPARATION_SECONDS, select_top_k

FEATURES_NAME = "visual_features.json"
EMBEDDINGS_NAME = "visual_embeddings.npy"
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


def write_features(cache_dir: str | Path, features: dict, embeddings) -> Path:
    """Write embeddings, then the JSON record, each atomically (the JSON marks completion)."""
    import numpy as np

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_tmp = cache_dir / (EMBEDDINGS_NAME + ".tmp.npy")
    np.save(embeddings_tmp, np.asarray(embeddings, dtype=np.float16))
    embeddings_tmp.replace(cache_dir / EMBEDDINGS_NAME)
    path = cache_dir / FEATURES_NAME
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(features, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def load_features(cache_dir: str | Path) -> dict | None:
    """Cached features, or None if absent or unreadable (treated as missing)."""
    path = Path(cache_dir) / FEATURES_NAME
    try:
        features = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    required = ("frame_times", "novelty", "shots")
    if not isinstance(features, dict) or any(key not in features for key in required):
        return None
    if len(features["frame_times"]) != len(features["novelty"]):
        return None
    return features


def load_current_features(
    cache_dir: str | Path,
    manifest: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict | None:
    """Cached features only if they still match this video's manifest and config.

    This is what consumers (build_moments, evaluate_selectors) should call, so a
    changed config, re-extracted frames, or a new feature version is treated as
    missing instead of silently reusing stale values. With no manifest passed it
    reads <cache_dir>/manifest.json; an unreadable manifest means the features
    cannot be verified and count as missing.
    """
    features = load_features(cache_dir)
    if features is None:
        return None
    if manifest is None:
        try:
            manifest = json.loads((Path(cache_dir) / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
    return features if is_fresh(features, manifest, config) else None


def load_embeddings(cache_dir: str | Path):
    """Cached unit-normalized embeddings as float32, or None."""
    import numpy as np

    path = Path(cache_dir) / EMBEDDINGS_NAME
    try:
        return np.load(path).astype(np.float32)
    except (OSError, ValueError):
        return None


def mean_pairwise_similarity(embeddings, frame_times: Sequence[float], times: Sequence[float]) -> float | None:
    """Mean pairwise cosine similarity of the frames nearest to the selected times.

    Higher means more redundant selections. None for fewer than two selections or
    when the embeddings do not line up with the frame timestamps.
    """
    import numpy as np

    if embeddings is None or len(times) < 2 or len(embeddings) != len(frame_times):
        return None
    chosen = []
    for t in times:
        i = bisect.bisect_left(frame_times, t)
        candidates = [j for j in (i - 1, i) if 0 <= j < len(frame_times)]
        chosen.append(min(candidates, key=lambda j: abs(frame_times[j] - t)))
    vectors = np.asarray(embeddings, dtype=np.float64)[chosen]
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
