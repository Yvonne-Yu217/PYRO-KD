"""Dev-only weight tuning for the two-stage selector (proposal §5.1, Risk 6).

Grid search over a few weights, scored by macro F1 at the primary tolerance,
with k-fold cross-validation over videos: for each fold the best grid point on
the training videos is scored on the held-out videos, which estimates how much
the tuned weights overfit. The final weights are the best on all dev videos.
Never runs on test (refuses unless dataset.split is "validation").
"""

from __future__ import annotations

import copy
import itertools
import math
import random
from typing import Any

from src.evaluation.metrics import temporal_precision_recall_f1
from src.experiments.config import ConfigError, get
from src.experiments.data import k_for, load_context
from src.selection.registry import resolve_spec, run_selector


def _set(options: dict, dotted: str, value: Any) -> None:
    node = options
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def grid_points(grid: dict[str, list]) -> list[dict[str, Any]]:
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def tune(cfg: dict, records, base: str = "AV", grid: dict[str, list] | None = None, folds: int = 5) -> dict:
    if get(cfg, "dataset.split") != "validation":
        raise ConfigError("tuning runs on the validation split (dev) only")
    grid = grid or get(cfg, "tuning.grid") or {"weights.nse": [-0.5, 0.0, 0.5], "weights.sc": [0.0, 0.5], "weights.gf": [0.0, 0.5]}
    tol = float(get(cfg, "selection.tolerance_seconds", 2.0))
    points = grid_points(grid)
    contexts = []
    for r in records:
        ctx, _ = load_context(r, cfg)
        if ctx.duration and ctx.ground_truth is not None:
            contexts.append((r.video_id, ctx, k_for(r, ctx.duration, cfg)))
    scores: dict[int, dict[str, float | None]] = {}
    for i, point in enumerate(points):
        spec = resolve_spec(base)
        options = copy.deepcopy(spec.get("options", {}))
        for key, value in point.items():
            _set(options, key, value)
        spec = {**spec, "name": f"{base}-grid{i}", "options": options}
        per_video = {}
        for vid, ctx, k in contexts:
            sel = run_selector(spec, ctx, k)
            per_video[vid] = (temporal_precision_recall_f1(sel.times, ctx.ground_truth, tol)["f1"]
                              if sel.status in ("ok", "ok_degraded") else None)
        scores[i] = per_video

    def mean_f1(i: int, ids) -> float | None:
        vals = [scores[i][v] for v in ids if scores[i].get(v) is not None]
        return math.fsum(vals) / len(vals) if vals else None

    def rank(i: int, ids):
        value = mean_f1(i, ids)
        return (-math.inf if value is None else value, -i)

    ids = sorted(v for v, _, _ in contexts)
    rng = random.Random(int(get(cfg, "experiment.seed", 0)))
    shuffled = ids[:]
    rng.shuffle(shuffled)
    folds = max(2, min(folds, len(ids))) if len(ids) >= 2 else 1
    cv = []
    for f in range(folds if len(ids) >= 2 else 0):
        held = set(shuffled[f::folds])
        train = [v for v in ids if v not in held]
        best = max(range(len(points)), key=lambda i: rank(i, train))
        cv.append({"fold": f, "held_out": sorted(held), "best_point": points[best],
                   "train_f1": mean_f1(best, train), "held_out_f1": mean_f1(best, held)})
    best_all = max(range(len(points)), key=lambda i: rank(i, ids))
    held_scores = [c["held_out_f1"] for c in cv if c["held_out_f1"] is not None]
    return {
        "base": base, "grid": grid, "n_videos": len(ids),
        "n_videos_scored": sum(scores[best_all].get(v) is not None for v in ids),
        "objective": f"macro F1 at +/-{tol} s",
        "best": points[best_all], "best_f1": mean_f1(best_all, ids),
        "cv": cv, "cv_held_out_mean_f1": (math.fsum(held_scores) / len(held_scores)) if held_scores else None,
        "all_points": [{"point": p, "f1": mean_f1(i, ids)} for i, p in enumerate(points)],
        "note": "dev only; apply the result through [selectors.AV] in the config, then evaluate test once",
    }
