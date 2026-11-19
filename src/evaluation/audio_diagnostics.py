"""Audio-cue diagnostics (proposal §7B).

1. Detector quality on the hand-labeled subset (§6 check): event onsets
   matched one-to-one within a tolerance, per superclass and pooled.
2. Trigger/suppress analysis: per superclass, how often human AD starts fall
   near an onset of that class (hit rate) versus how much of the timeline is
   near such onsets at all (chance rate); lift > 1 suggests the class
   triggers description, < 1 that describers avoid it (suppress).
3. Silent-visual-event coverage: GT AD starts with no audio change nearby,
   and each selector's recall on that subset.

Hand-label file format (JSONL, one event per line, documented in
docs/data_formats.md): {"video_id": str, "time": seconds, "label": str} or
with "superclass" instead of "label". No hand labels ship with the repo;
creating them is a §6 task for the team.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from src.audio.taxonomy import superclass_of
from src.evaluation.metrics import temporal_precision_recall_f1


def load_hand_labels(path: str | Path) -> tuple[dict[str, list[dict]], list[str]]:
    """({video_id: events}, problems). Malformed lines are reported, not fatal."""
    by_video: dict[str, list[dict]] = {}
    problems: list[str] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            time = float(row["time"])
            if time < 0:
                raise ValueError("negative time")
            superclass = row.get("superclass") or superclass_of(str(row["label"]))
            by_video.setdefault(str(row["video_id"]), []).append(
                {"time": time, "label": row.get("label"), "superclass": superclass})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            problems.append(f"line {n}: {exc}")
    return by_video, problems


def detector_precision_recall(
    predicted: dict[str, Sequence[dict]],
    labeled: dict[str, Sequence[dict]],
    tolerance: float = 1.0,
    classes: Sequence[str] | None = None,
) -> dict:
    """Per-superclass and pooled onset P/R/F1 over the labeled videos only."""
    classes = list(classes) if classes is not None else sorted(
        {e["superclass"] for events in labeled.values() for e in events})
    out: dict[str, dict] = {}
    pooled = {"tp": 0, "fp": 0, "fn": 0}
    for c in classes:
        tp = fp = fn = 0
        for video_id, gold in labeled.items():
            g = [e["time"] for e in gold if e["superclass"] == c]
            p = [float(e["time"]) for e in predicted.get(video_id, []) if e.get("superclass") == c]
            r = temporal_precision_recall_f1(p, g, tolerance)
            tp, fp, fn = tp + r["tp"], fp + r["fp"], fn + r["fn"]
        out[c] = _prf(tp, fp, fn)
        pooled = {"tp": pooled["tp"] + tp, "fp": pooled["fp"] + fp, "fn": pooled["fn"] + fn}
    return {"tolerance_seconds": tolerance, "per_class": out, "pooled": _prf(**pooled),
            "n_videos": len(labeled)}


def _prf(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None}


def _near_fraction(times: Sequence[float], duration: float, window: float, step: float = 0.1) -> float:
    if duration <= 0:
        return 0.0
    points = int(duration / step) + 1
    ordered = sorted(times)
    import bisect

    near = 0
    for i in range(points):
        x = i * step
        j = bisect.bisect_left(ordered, x - window)
        if j < len(ordered) and ordered[j] <= x + window:
            near += 1
    return near / points


def trigger_suppress(
    videos: Sequence[dict], window: float = 2.0, classes: Sequence[str] | None = None,
) -> dict:
    """videos: [{"video_id", "duration", "gt_times", "events": [{"time", "superclass"}]}].

    Per class: pooled hit rate (GT starts within +/- window of an onset), the
    duration-weighted chance rate, lift = hit / chance, and per-video
    (hit - chance) differences for paired statistics.
    """
    classes = list(classes) if classes is not None else sorted(
        {e["superclass"] for v in videos for e in v["events"]})
    result: dict[str, dict] = {}
    for c in classes:
        hits = total = 0
        chance_num = chance_den = 0.0
        per_video = []
        for v in videos:
            onsets = sorted(float(e["time"]) for e in v["events"] if e.get("superclass") == c)
            gt = list(v["gt_times"])
            if not gt:
                continue
            import bisect

            h = 0
            for t in gt:
                j = bisect.bisect_left(onsets, t - window)
                h += j < len(onsets) and onsets[j] <= t + window
            chance = _near_fraction(onsets, float(v["duration"]), window)
            hits, total = hits + h, total + len(gt)
            chance_num += chance * float(v["duration"])
            chance_den += float(v["duration"])
            per_video.append({"video_id": v["video_id"], "hit_rate": h / len(gt), "chance_rate": chance,
                              "excess": h / len(gt) - chance})
        hit_rate = hits / total if total else None
        chance_rate = chance_num / chance_den if chance_den else None
        result[c] = {
            "n_gt": total, "hit_rate": hit_rate, "chance_rate": chance_rate,
            "lift": (hit_rate / chance_rate) if hit_rate is not None and chance_rate else None,
            "per_video": per_video,
        }
    return {"window_seconds": window, "classes": result}


def silent_gt_times(
    gt_times: Sequence[float], grid: Sequence[float], audio_change: Sequence[float],
    threshold: float = 0.2, window: float = 2.0,
) -> list[float]:
    """GT AD starts with no audio change >= threshold within +/- window."""
    import bisect

    out = []
    for t in gt_times:
        lo, hi = bisect.bisect_left(grid, t - window), bisect.bisect_right(grid, t + window)
        if max(audio_change[lo:hi], default=0.0) < threshold:
            out.append(t)
    return out


def silent_visual_coverage(
    selections: dict[str, Sequence[float]], silent_gt: Sequence[float], all_gt: Sequence[float], tolerance: float = 2.0,
) -> dict:
    """Share of GT starts that are silent, and each selector's recall on them."""
    return {
        "n_gt": len(all_gt),
        "n_silent": len(silent_gt),
        "silent_fraction": len(silent_gt) / len(all_gt) if all_gt else None,
        "recall_on_silent": {
            name: (temporal_precision_recall_f1(times, silent_gt, tolerance)["recall"] if silent_gt else None)
            for name, times in selections.items()
        },
    }
