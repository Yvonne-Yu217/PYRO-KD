"""Videos and validated per-video inputs for an experiment.

Every cache read goes through the existing validators (media cache complete
and current; audio features stamped for the current WAV; visual features
aligned with the cached frames; cues stamped for this WAV and VAD result), so
stale or broken caches are reported per video, never used.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.audio.cues_io import CUES_NAME, load_cues_for_selection
from src.datasets.adapters import VideoRecord, build_adapter
from src.experiments.config import ConfigError, get
from src.selection.registry import SelectionContext
from src.utils.audio_features import resolve_audio_path
from src.utils.cache_checks import load_audio_features, read_manifest
from src.utils.selection import k_for_duration


def load_records(cfg: dict) -> list[VideoRecord]:
    """The videos named by dataset.split and dataset.subset (all | dev | test_sample | ids)."""
    ds = get(cfg, "dataset")
    adapter = build_adapter(ds)
    split = ds.get("split", "validation")
    records = adapter.load(split) if ds["name"] != "egoschema" else adapter.load()
    subset = ds.get("subset", "all")
    if subset == "all":
        chosen = records
    else:
        ids = _subset_ids(ds, subset)
        by_id = {r.video_id: r for r in records}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ConfigError(f"subset {subset!r} names videos absent from {ds['name']} {split}: {missing[:5]}")
        chosen = [by_id[i] for i in ids]
    limit = ds.get("limit")
    return chosen[: int(limit)] if limit else chosen


def _subset_ids(ds: dict, subset) -> list[str]:
    if isinstance(subset, list):
        return [str(i) for i in subset]
    splits_path = ds.get("splits_config")
    if subset in ("dev", "test_sample"):
        if not splits_path:
            raise ConfigError("dataset.splits_config is required for subset dev/test_sample")
        splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))
        if ds["name"] == "mtg-ad":
            return [v["video_id"] for v in splits["dev"]["videos"]] if subset == "dev" else list(splits["test"]["video_ids"])
        if ds["name"] == "reframed":
            return [c["video_id"] for c in splits["clips"]] if subset == "dev" else list(splits["test"]["video_ids"])
    if isinstance(subset, str) and subset.startswith("ids:"):
        return [line.strip() for line in Path(subset[4:]).read_text().splitlines() if line.strip()]
    raise ConfigError(f"unknown dataset.subset {subset!r}")


def k_rate(cfg: dict) -> float:
    rate = get(cfg, "k_rule.rate_per_minute")
    if rate is None:
        splits_path = get(cfg, "dataset.splits_config")
        if not splits_path:
            raise ConfigError("set k_rule.rate_per_minute or dataset.splits_config")
        rate = json.loads(Path(splits_path).read_text(encoding="utf-8"))["k_rule"]["rate_per_minute"]
    return float(rate)


def video_duration(record: VideoRecord, manifest: dict | None) -> float | None:
    if record.duration:
        return float(record.duration)
    if manifest and isinstance(manifest.get("media_info"), dict):
        d = manifest["media_info"].get("duration")
        return float(d) if d else None
    return None


def ground_truth(record: VideoRecord, duration: float, slack: float) -> tuple[list[float], list[float]]:
    """(scored GT starts, starts beyond duration + slack). Out-of-bounds starts are excluded for every selector."""
    starts = sorted(record.ad_starts)
    limit = duration + slack
    return [t for t in starts if t <= limit], [t for t in starts if t > limit]


_MEMO: dict[tuple, tuple[SelectionContext, dict]] = {}


def clear_context_cache() -> None:
    """Forget memoized contexts (call when caches may have changed, e.g. a new run)."""
    _MEMO.clear()


def load_context(record: VideoRecord, cfg: dict, cue_set: str = CUES_NAME, need_visual: bool = True) -> tuple[SelectionContext, dict]:
    """SelectionContext for one video and an availability report {stream: "ok" | reason}.

    Memoized per process on (cache root, video, cue set, visual): validation
    hashes the WAV, and every selector and the evaluator need the same inputs.
    """
    key = (str(get(cfg, "paths.cache_root")), record.video_id, cue_set, need_visual,
           repr(get(cfg, "selection.bundle")), float(get(cfg, "selection.duration_slack_seconds", 1.0)))
    if key not in _MEMO:
        _MEMO[key] = _load_context(record, cfg, cue_set, need_visual)
    return _MEMO[key]


def _load_context(record: VideoRecord, cfg: dict, cue_set: str, need_visual: bool) -> tuple[SelectionContext, dict]:
    cache = Path(get(cfg, "paths.cache_root")) / record.video_id
    manifest, manifest_reason = read_manifest(cache)
    duration = video_duration(record, manifest)
    available: dict[str, str] = {"manifest": "ok" if manifest else manifest_reason}
    features, reason = load_audio_features(cache)
    available["vad"] = "ok" if features is not None else reason
    visual = embeddings = None
    if need_visual:
        from src.utils.visual_features import load_embeddings, load_validated_features

        visual, vreason = load_validated_features(cache)
        available["visual"] = "ok" if visual is not None else vreason
        if visual is not None:
            embeddings = load_embeddings(cache, visual)
            available["embeddings"] = "ok" if embeddings is not None else "unavailable_embeddings"
    cues = None
    if features is not None and manifest is not None:
        try:
            wav = resolve_audio_path(manifest, cache)
        except ValueError:
            wav = None
        cues, creason = load_cues_for_selection(cache, wav, features, cue_set)
        available["audio_cues"] = "ok" if cues is not None else creason
    else:
        available["audio_cues"] = "no_vad_features"
    missing = {k: v for k, v in available.items() if v != "ok"}
    gt = None
    if record.ad is not None and duration:
        gt, _ = ground_truth(record, duration, float(get(cfg, "selection.duration_slack_seconds", 1.0)))
    context = SelectionContext(
        video_id=record.video_id, duration=duration or 0.0, audio_features=features, audio_cues=cues,
        visual_features=visual, embeddings=embeddings, ground_truth=gt, missing=missing,
        bundle_config=get(cfg, "selection.bundle") or None,
    )
    return context, available


def k_for(record: VideoRecord, duration: float, cfg: dict) -> int:
    return k_for_duration(duration, k_rate(cfg))
