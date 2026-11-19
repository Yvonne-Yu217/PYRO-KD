"""Selection and selector-only evaluation. No MLLM is imported or called here.

run_selection writes one row per video:
    {"video_id", "selector", "run_id", "k", "duration", "times", "status",
     "available", "details", "diagnostics", "oracle"}
and skips videos already written (resume). A failure on one video is recorded
as status "error" and the batch continues; --retry-failed recomputes them.

evaluate_selections scores every selection run against the GT AD starts with
the shared one-to-one matcher, plus speech overlap (rendered at 1.2x with the
full word budget), silence utilization, placement diagnostics and silent-
visual-event coverage. Comparisons are paired over the videos both runs
scored: bootstrap CI and Wilcoxon signed-rank, Holm-adjusted across the
comparison family. Oracle rows are labeled as upper bounds and are never
compared as if they were deployable.
"""

from __future__ import annotations

import math
from pathlib import Path
import traceback
from typing import Any, Sequence

from src.evaluation import audio_diagnostics, placement
from src.evaluation.metrics import temporal_precision_recall_f1
from src.evaluation.significance import holm, wilcoxon_signed_rank
from src.evaluation.stats import bootstrap_mean_ci, paired_bootstrap_diff
from src.experiments.config import get
from src.experiments.data import ground_truth, k_for, load_context
from src.experiments.io import append_jsonl, latest_by_key, read_jsonl, update_status, write_json
from src.experiments.matrix import SelectionRun
from src.selection.registry import run_selector

TOLERANCE_SWEEP = (0.0, 1.0, 2.0, 3.0, 5.0)


def selections_path(exp_dir: Path, run: SelectionRun) -> Path:
    return exp_dir / "selections" / f"{run.run_id}.jsonl"


def run_selection(cfg: dict, run: SelectionRun, records, exp_dir: Path, retry_failed: bool = False) -> Path:
    path = selections_path(exp_dir, run)
    done = latest_by_key(read_jsonl(path), "video_id")
    spec = run.spec
    cue_set = spec.get("cue_set", "audio_cues.json")
    needs_visual = spec["kind"] in ("V", "two_stage")
    counts = {"written": 0, "skipped": 0, "errors": 0}
    for record in records:
        previous = done.get(record.video_id)
        if previous and (previous["status"] != "error" or not retry_failed):
            counts["skipped"] += 1
            continue
        try:
            context, available = load_context(record, cfg, cue_set, need_visual=needs_visual)
            if not context.duration:
                row = {"video_id": record.video_id, "status": "missing_duration", "times": [], "k": None,
                       "available": available}
            else:
                k = k_for(record, context.duration, cfg)
                selection = run_selector(spec, context, k)
                row = {"video_id": record.video_id, "k": k, "duration": context.duration, "times": selection.times,
                       "status": selection.status, "available": available, "details": selection.details,
                       "diagnostics": selection.diagnostics}
        except Exception as exc:  # recorded, batch continues
            row = {"video_id": record.video_id, "status": "error", "times": [], "k": None,
                   "error": f"{exc!r}", "traceback": traceback.format_exc(limit=3)}
            counts["errors"] += 1
        row.update(selector=run.name, run_id=run.run_id, oracle=run.is_oracle, ablation=run.ablation)
        append_jsonl(path, [row])
        counts["written"] += 1
    update_status(exp_dir, "select", run.run_id, {"selector": run.name, **counts})
    return path


def _video_metrics(row: dict, record, cfg: dict) -> dict | None:
    """Per-video scores for one selection row; None if the selector did not run."""
    if row["status"] not in ("ok", "ok_degraded"):
        return None
    duration = row["duration"]
    tol = float(get(cfg, "selection.tolerance_seconds", 2.0))
    slack = float(get(cfg, "selection.duration_slack_seconds", 1.0))
    truth, out_of_bounds = ground_truth(record, duration, slack)
    times = row["times"]
    primary = temporal_precision_recall_f1(times, truth, tol)
    context, _ = load_context(record, cfg, need_visual=False)
    speech = [tuple(s) for s in context.audio_features["speech"]] if context.audio_features else None
    gaps = [(g["start"], g["end"]) for g in context.audio_features["gaps"]] if context.audio_features else None
    out: dict[str, Any] = {
        "video_id": row["video_id"], "k": row["k"], "num_selected": len(times), "shortfall": row["k"] - len(times),
        "num_gt": len(truth), "num_gt_out_of_bounds": len(out_of_bounds),
        "tp": primary["tp"], "fp": primary["fp"], "fn": primary["fn"],
        "precision": primary["precision"], "recall": primary["recall"], "f1": primary["f1"],
        "mean_temporal_error_seconds": primary["mean_temporal_error_seconds"],
        "f1_by_tolerance": {str(t): temporal_precision_recall_f1(times, truth, t)["f1"] for t in TOLERANCE_SWEEP},
        "status": row["status"], "group": record.group,
        **placement.placement_diagnostics(times, duration, speech, gaps),
        "gt_in_gap_fraction": placement.gt_in_gap_fraction(truth, speech, duration),
        "overlap_seconds": None, "silence_utilization": None,
    }
    if speech is not None:
        placements = placement.budget_placements(times, speech, duration)
        wps, speed = float(get(cfg, "evaluation.tts_words_per_second", 2.5)), float(get(cfg, "evaluation.tts_speed", 1.2))
        out["overlap_seconds"] = placement.speech_overlap(placements, speech, wps, speed)["overlap_seconds"]
        out["silence_utilization"] = placement.silence_utilization(placements, speech, duration, wps, speed)["mean"]
        out["zero_budget_picks"] = sum(1 for _, n in placements if n == 0)
    if context.audio_cues is not None:
        bundle = context.bundle()
        silent = audio_diagnostics.silent_gt_times(truth, bundle.grid, bundle.audio_change())
        out["n_silent_gt"] = len(silent)
        out["recall_on_silent_gt"] = (temporal_precision_recall_f1(times, silent, tol)["recall"] if silent else None)
    return out


def _summary(rows: Sequence[dict], resamples: int, seed: int) -> dict:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return math.fsum(vals) / len(vals) if vals else None

    return {
        "n_videos": len(rows),
        "macro_f1": bootstrap_mean_ci([r["f1"] for r in rows], resamples, seed),
        "macro_precision": bootstrap_mean_ci([r["precision"] for r in rows], resamples, seed),
        "macro_recall": bootstrap_mean_ci([r["recall"] for r in rows], resamples, seed),
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "overlap_seconds": bootstrap_mean_ci([r["overlap_seconds"] for r in rows], resamples, seed),
        "silence_utilization": mean("silence_utilization"),
        "span_fraction": mean("span_fraction"),
        "densest_decile_share": mean("densest_decile_share"),
        "busiest_gap_share": mean("busiest_gap_share"),
        "recall_on_silent_gt": mean("recall_on_silent_gt"),
        "videos_with_shortfall": sum(r["shortfall"] > 0 for r in rows),
        "f1_by_tolerance": {t: mean_of([r["f1_by_tolerance"][t] for r in rows]) for t in (rows[0]["f1_by_tolerance"] if rows else {})},
    }


def mean_of(values):
    kept = [v for v in values if v is not None]
    return math.fsum(kept) / len(kept) if kept else None


def evaluate_selections(cfg: dict, runs: Sequence[SelectionRun], records, exp_dir: Path, baseline: str = "U") -> dict:
    by_id = {r.video_id: r for r in records}
    resamples = int(get(cfg, "evaluation.resamples", 10000))
    seed = int(get(cfg, "experiment.seed", 0))
    per_run: dict[str, dict] = {}
    for run in runs:
        rows = latest_by_key(read_jsonl(selections_path(exp_dir, run)), "video_id")
        scored, not_run = [], {}
        for video_id, row in rows.items():
            if video_id not in by_id:
                continue
            metrics = _video_metrics(row, by_id[video_id], cfg)
            if metrics is None:
                not_run[row["status"]] = not_run.get(row["status"], 0) + 1
            else:
                scored.append(metrics)
        per_run[run.name] = {"run_id": run.run_id, "oracle": run.is_oracle, "ablation": run.ablation,
                             "not_run": not_run, "summary": _summary(scored, resamples, seed), "videos": scored}
    # Summary rows are only comparable on the same videos: also summarize every run
    # on the videos that every core run (ablation None) scored.
    core_sets = [{v["video_id"] for v in r["videos"]} for r in per_run.values() if r["ablation"] is None]
    common = set.intersection(*core_sets) if core_sets else set()
    for r in per_run.values():
        r["summary_common"] = _summary([v for v in r["videos"] if v["video_id"] in common], resamples, seed)
    comparisons = []
    names = [n for n in per_run if n != baseline]
    for name in names:
        if baseline not in per_run:
            break
        a = {v["video_id"]: v for v in per_run[name]["videos"]}
        b = {v["video_id"]: v for v in per_run[baseline]["videos"]}
        shared = sorted(a.keys() & b.keys())
        for metric in ("f1", "overlap_seconds"):
            xa = [a[i][metric] for i in shared]
            xb = [b[i][metric] for i in shared]
            comparisons.append({
                "selector": name, "baseline": baseline, "metric": metric, "n_shared_videos": len(shared),
                "oracle_upper_bound": per_run[name]["oracle"],
                "bootstrap": paired_bootstrap_diff(xa, xb, resamples, seed),
                "wilcoxon": wilcoxon_signed_rank(xa, xb),
            })
    deployable = [c for c in comparisons if not c["oracle_upper_bound"]]
    for metric in ("f1", "overlap_seconds"):
        family = [c for c in deployable if c["metric"] == metric]
        for c, p in zip(family, holm([c["wilcoxon"]["p_value"] for c in family])):
            c["wilcoxon"]["p_holm"] = p
    result = {"tolerance_seconds": get(cfg, "selection.tolerance_seconds"), "baseline": baseline,
              "n_videos": len(records), "common_video_ids": sorted(common), "runs": per_run, "comparisons": comparisons,
              "note": "O is an oracle upper bound (human AD timestamps), not a deployable selector."}
    write_json(exp_dir / "eval" / "selection_metrics.json", result)
    return result


def write_timelines(cfg: dict, runs: Sequence[SelectionRun], records, exp_dir: Path, max_videos: int = 20) -> Path:
    """SVG lanes per video and selector: speech shaded, GT ticks above, picks below."""
    rows = []
    selections = {run.name: latest_by_key(read_jsonl(selections_path(exp_dir, run)), "video_id") for run in runs}
    slack = float(get(cfg, "selection.duration_slack_seconds", 1.0))
    for record in records[:max_videos]:
        context, _ = load_context(record, cfg, need_visual=False)
        speech = context.audio_features["speech"] if context.audio_features else []
        for name, by_video in selections.items():
            row = by_video.get(record.video_id)
            if not row or not row.get("duration"):
                continue
            gt, _ = ground_truth(record, row["duration"], slack)
            rows.append({"label": f"{record.video_id} {name}", "duration": row["duration"], "speech": speech,
                         "gt": gt, "times": row["times"]})
    path = exp_dir / "plots" / "selection_timelines.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(placement.timeline_svg(rows, title="Selections (red) vs human AD starts (green); speech shaded"),
                    encoding="utf-8")
    return path
