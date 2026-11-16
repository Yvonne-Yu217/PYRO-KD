#!/usr/bin/env python
"""Selector-only timing evaluation on MTG-AD: no VLM, no GPU.

Scores where each selector places its K moments against where the professional
describers started their descriptions (temporal precision/recall/F1 within
+/- tau seconds, one-to-one matching). This answers RQ1(a) and, with the
oracle, RQ4 (headroom) before any model is run.

    python scripts/evaluate_selectors.py --set dev
    python scripts/evaluate_selectors.py --set validation --selectors U O
    python scripts/evaluate_selectors.py --set test --output outputs/selector_eval_test.json

Selectors: U (uniform), O (oracle: human start times capped at K) and A
(audio-only; needs data/cache/mtg-ad/<video_id>/audio_features.json, built by
scripts/cache_audio_features.py, so videos without it are skipped and counted).
V and AV are added to SELECTORS as they are implemented. U and O need only the
annotations, so they can be scored on all 332 validation videos. K comes from
the same duration rule as generation (configs/mtg_ad_splits.json), so the
comparison uses the same budget everywhere. Tune nothing on --set test.

Each per-video value is one observation: confidence intervals are paired
percentile bootstraps over videos, never over moments.

Cached features are used only after validation (src/utils/cache_checks.py): the
media cache must be complete and current, and audio features readable and
matching the current WAV. A video whose features fail any check is skipped for
A with an explicit reason, and counted in the report; reading never raises, so a
broken audio cache cannot stop a U/O-only run.

Out-of-duration annotations. Some MTG-AD videos list AD starts after their
declared (whole-second) duration, which no selector working inside the clip can
place. The default policy EXCLUDES such starts (later than the declared duration
plus --duration-slack, default 1 s for second rounding) from the ground truth,
for every selector including O, and reports the affected videos and counts.
--keep-out-of-bounds instead scores against every annotated start, which is the
unrestricted annotation upper bound: O may then place moments outside the clip,
and the report says so. Starts are never clamped, and nothing is dropped silently.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import mtg_ad  # noqa: E402
from src.evaluation.metrics import temporal_precision_recall_f1  # noqa: E402
from src.evaluation.stats import DEFAULT_RESAMPLES, bootstrap_mean_ci, paired_bootstrap_diff  # noqa: E402
from src.utils.audio_features import available_gap, speech_of  # noqa: E402
from src.utils.audio_selector import AudioSelector  # noqa: E402
from src.utils.cache_checks import load_audio_features  # noqa: E402
from src.utils.selection import k_for_duration, oracle_timestamps, uniform_timestamps  # noqa: E402

TOLERANCE_SWEEP = (0.0, 1.0, 2.0, 3.0, 5.0)
# (selector, baseline) pairs reported as paired differences in F1.
COMPARISONS = (("A", "U"), ("O", "U"), ("O", "A"))
DURATION_SLACK_SECONDS = 1.0  # declared durations are whole seconds
BOUNDS_POLICIES = ("exclude", "keep")


def ground_truth_times(
    video: mtg_ad.MTGVideo, policy: str = "exclude", slack: float = DURATION_SLACK_SECONDS,
) -> tuple[list[float], list[float]]:
    """(ground truth to score against, annotated starts beyond the declared duration).

    policy "exclude" removes starts later than duration + slack from the scored
    ground truth; "keep" scores against all of them. The second list is the same
    either way, so the anomaly is always reported.
    """
    if policy not in BOUNDS_POLICIES:
        raise ValueError(f"policy must be one of {BOUNDS_POLICIES}")
    starts = sorted(float(item.timestamp) for item in video.ad)
    limit = video.duration_seconds + slack
    out_of_bounds = [t for t in starts if t > limit]
    return (starts if policy == "keep" else [t for t in starts if t <= limit]), out_of_bounds


def _times_u(video, k, features, truth):
    return uniform_timestamps(video.duration_seconds, k)


def _times_o(video, k, features, truth):
    return oracle_timestamps(truth, k)


def _times_a(video, k, features, truth):
    if features is None:
        return None
    return [candidate.time for candidate in AudioSelector().select(features, k)]


SELECTORS: dict[str, Callable[..., list[float] | None]] = {"U": _times_u, "O": _times_o, "A": _times_a}


def load_video_set(name: str, splits: dict, data_dir: Path) -> list[mtg_ad.MTGVideo]:
    """dev = the fixed 10 dev videos, test = the fixed 50, validation = all 332."""
    if name == "test":
        by_id = {v.video_id: v for v in mtg_ad.load_split(data_dir, "test")}
        return [by_id[i] for i in splits["test"]["video_ids"]]
    validation = mtg_ad.load_split(data_dir, "validation")
    if name == "validation":
        return validation
    by_id = {v.video_id: v for v in validation}
    return [by_id[v["video_id"]] for v in splits["dev"]["videos"]]


def in_speech_rate(times: Sequence[float], features: dict | None, duration: float) -> float | None:
    """Fraction of moments that start inside detected speech (None without VAD features)."""
    if features is None or not times:
        return None
    speech = speech_of(features)
    return sum(available_gap(t, speech, duration) == 0.0 for t in times) / len(times)


def evaluate_video(
    selector: str,
    video: mtg_ad.MTGVideo,
    k: int,
    features: dict | None,
    tolerance: float,
    sweep: Sequence[float] = TOLERANCE_SWEEP,
    *,
    bounds: str = "exclude",
    slack: float = DURATION_SLACK_SECONDS,
) -> dict[str, Any] | None:
    """Score one selector on one video; None if the selector cannot run (no features for A).

    Ground truth follows the out-of-duration policy (see the module docstring);
    both the annotated and the scored counts are recorded.
    """
    truth, out_of_bounds = ground_truth_times(video, bounds, slack)
    times = SELECTORS[selector](video, k, features, truth)
    if times is None:
        return None
    limit = video.duration_seconds + slack
    primary = temporal_precision_recall_f1(times, truth, tolerance)
    return {
        "video_id": video.video_id,
        "selector": selector,
        "k": k,
        "num_gt": len(video.ad),
        "num_gt_scored": len(truth),
        "num_gt_out_of_bounds": len(out_of_bounds),
        "out_of_bounds_gt_times": out_of_bounds,
        "num_selected": len(times),
        "num_selected_out_of_bounds": sum(t > limit for t in times),
        "shortfall": k - len(times),
        "tp": primary["tp"], "fp": primary["fp"], "fn": primary["fn"],
        "precision": primary["precision"],
        "recall": primary["recall"],
        "f1": primary["f1"],
        "mean_temporal_error_seconds": primary["mean_temporal_error_seconds"],
        "f1_by_tolerance": {
            str(tau): temporal_precision_recall_f1(times, truth, tau)["f1"] for tau in sweep
        },
        "in_speech_rate": in_speech_rate(times, features, video.duration_seconds),
        "times": times,
    }


def _mean(values: Sequence[float | None]) -> float | None:
    kept = [v for v in values if v is not None]
    return math.fsum(kept) / len(kept) if kept else None


def summarize(rows: Sequence[dict], resamples: int, seed: int) -> dict[str, Any]:
    """Macro (per-video mean) and micro (pooled) scores with bootstrap intervals."""
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    matched_error = math.fsum(
        r["mean_temporal_error_seconds"] * r["tp"] for r in rows if r["tp"]
    )
    return {
        "n_videos": len(rows),
        "macro_f1": bootstrap_mean_ci([r["f1"] for r in rows], resamples, seed),
        "macro_precision": bootstrap_mean_ci([r["precision"] for r in rows], resamples, seed),
        "macro_recall": bootstrap_mean_ci([r["recall"] for r in rows], resamples, seed),
        "micro_precision": tp / (tp + fp) if tp + fp else None,
        "micro_recall": tp / (tp + fn) if tp + fn else None,
        "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "mean_temporal_error_seconds": matched_error / tp if tp else None,
        "total_selected": sum(r["num_selected"] for r in rows),
        "total_gt": sum(r["num_gt_scored"] for r in rows),
        "total_gt_annotated": sum(r["num_gt"] for r in rows),
        "total_selected_out_of_bounds": sum(r["num_selected_out_of_bounds"] for r in rows),
        "videos_with_shortfall": sum(r["shortfall"] > 0 for r in rows),
        "total_shortfall": sum(max(r["shortfall"], 0) for r in rows),
        "macro_f1_by_tolerance": {
            tau: _mean([r["f1_by_tolerance"][tau] for r in rows])
            for tau in (rows[0]["f1_by_tolerance"] if rows else {})
        },
        "in_speech_rate": _mean([r["in_speech_rate"] for r in rows]),
        "n_with_audio_features": sum(r["in_speech_rate"] is not None for r in rows),
    }


def compare(
    by_selector: dict[str, list[dict]], resamples: int, seed: int,
) -> list[dict[str, Any]]:
    """Paired F1 differences on the videos both selectors ran on."""
    out = []
    for selector, baseline in COMPARISONS:
        if selector not in by_selector or baseline not in by_selector:
            continue
        a = {r["video_id"]: r["f1"] for r in by_selector[selector]}
        b = {r["video_id"]: r["f1"] for r in by_selector[baseline]}
        shared = sorted(a.keys() & b.keys())
        diff = paired_bootstrap_diff([a[i] for i in shared], [b[i] for i in shared], resamples, seed)
        out.append({"selector": selector, "baseline": baseline, "n_shared_videos": len(shared), **diff})
    return out


def _fmt(value: float | None, digits: int = 3) -> str:
    return "  n/a" if value is None else f"{value:.{digits}f}"


def format_report(result: dict) -> str:
    lines = [
        f"Selector-only timing evaluation ({result['set']}, tau=+/-{result['tolerance']:g}s, "
        f"K = round({result['k_rate_per_minute']} x minutes), {result['n_videos_in_set']} videos)",
        "",
        f"{'sel':<4}{'n':>5}  {'macro F1 [95% CI]':<24}{'P':>7}{'R':>7}{'mean err s':>12}"
        f"{'short':>7}{'in-speech':>11}",
    ]
    for name, s in result["selectors"].items():
        f1 = s["macro_f1"]
        ci = f"{_fmt(f1['mean'])} [{_fmt(f1['ci_low'])}, {_fmt(f1['ci_high'])}]"
        lines.append(
            f"{name:<4}{s['n_videos']:>5}  {ci:<24}{_fmt(s['macro_precision']['mean']):>7}"
            f"{_fmt(s['macro_recall']['mean']):>7}{_fmt(s['mean_temporal_error_seconds'], 2):>12}"
            f"{s['videos_with_shortfall']:>7}{_fmt(s['in_speech_rate']):>11}"
        )
    for name, count in result["skipped"].items():
        reasons = ", ".join(
            f"{reason} x{n}" for reason, n in sorted(result["skipped_reasons"].get(name, {}).items()))
        lines.append(f"{name}: skipped {count} video(s) without usable audio features ({reasons})")
    audio = result["audio_features"]
    if audio["unavailable"]:
        reasons = ", ".join(f"{r} x{n}" for r, n in sorted(audio["unavailable"].items()))
        lines.append(
            f"audio features usable for {audio['usable']} of {result['n_videos_in_set']} videos "
            f"(not usable: {reasons}); in-speech is computed only where usable"
        )
    bounds = result["bounds"]
    if bounds["videos"]:
        ids = ", ".join(v["video_id"] for v in bounds["videos"])
        if bounds["policy"] == "exclude":
            lines.append(
                f"Out-of-duration AD starts: {bounds['total_gt_out_of_bounds']} in "
                f"{len(bounds['videos'])} video(s) ({ids}) EXCLUDED from the ground truth for every "
                f"selector (declared duration + {bounds['slack_seconds']:g}s)"
            )
        else:
            lines.append(
                f"Out-of-duration AD starts: {bounds['total_gt_out_of_bounds']} in "
                f"{len(bounds['videos'])} video(s) ({ids}) INCLUDED: unrestricted annotation upper "
                f"bound; selected out of bounds: {bounds['selected_out_of_bounds']}"
            )
    else:
        lines.append("Out-of-duration AD starts: none")
    if result["comparisons"]:
        lines += ["", "Paired F1 differences (selector minus baseline, per video):"]
        for c in result["comparisons"]:
            lines.append(
                f"  {c['selector']} - {c['baseline']}: {_fmt(c['mean_diff'])} "
                f"[{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}]  (n={c['n_shared_videos']})"
            )
    lines += ["", "Macro F1 by tolerance (seconds):"]
    taus = next(iter(result["selectors"].values()))["macro_f1_by_tolerance"].keys() if result["selectors"] else []
    lines.append(f"{'sel':<4}" + "".join(f"{'+/-' + t:>9}" for t in taus))
    for name, s in result["selectors"].items():
        lines.append(f"{name:<4}" + "".join(f"{_fmt(s['macro_f1_by_tolerance'][t]):>9}" for t in taus))
    return "\n".join(lines)


def run(
    videos: Sequence[mtg_ad.MTGVideo],
    rate_per_minute: float,
    selectors: Sequence[str],
    cache_dir: Path,
    tolerance: float,
    resamples: int,
    seed: int,
    *,
    bounds: str = "exclude",
    slack: float = DURATION_SLACK_SECONDS,
) -> dict[str, Any]:
    rows: dict[str, list[dict]] = {name: [] for name in selectors}
    skipped: dict[str, int] = {}
    skipped_reasons: dict[str, dict[str, int]] = {}
    unavailable: list[dict[str, str]] = []
    audio_reasons: dict[str, int] = {}
    anomalies: list[dict[str, Any]] = []
    usable_audio = 0
    for video in videos:
        k = k_for_duration(video.duration_seconds, rate_per_minute)
        # Validated and best effort: a missing, stale, or unreadable cache is a reason,
        # never an exception, so it cannot stop selectors that do not need it.
        features, reason = load_audio_features(cache_dir / video.video_id)
        if reason is None:
            usable_audio += 1
        else:
            audio_reasons[reason] = audio_reasons.get(reason, 0) + 1
        _, out_of_bounds = ground_truth_times(video, bounds, slack)
        if out_of_bounds:
            anomalies.append({
                "video_id": video.video_id, "declared_duration_seconds": video.duration_seconds,
                "num_gt": len(video.ad), "num_gt_out_of_bounds": len(out_of_bounds),
                "out_of_bounds_gt_times": out_of_bounds,
            })
        for name in selectors:
            row = evaluate_video(name, video, k, features, tolerance, bounds=bounds, slack=slack)
            if row is None:
                skipped[name] = skipped.get(name, 0) + 1
                why = reason or "unavailable"
                skipped_reasons.setdefault(name, {})[why] = skipped_reasons.get(name, {}).get(why, 0) + 1
                unavailable.append({"video_id": video.video_id, "selector": name, "reason": why})
            else:
                rows[name].append(row)
    by_selector = {name: r for name, r in rows.items() if r}
    return {
        "bounds": {
            "policy": bounds,
            "slack_seconds": slack,
            "unrestricted_annotation_upper_bound": bounds == "keep",
            "videos": anomalies,
            "total_gt_out_of_bounds": sum(a["num_gt_out_of_bounds"] for a in anomalies),
            "selected_out_of_bounds": {
                name: sum(r["num_selected_out_of_bounds"] for r in rs) for name, rs in by_selector.items()
            },
        },
        "audio_features": {
            "usable": usable_audio,
            "unavailable": audio_reasons,
        },
        "skipped_reasons": skipped_reasons,
        "unavailable": unavailable,
        "tolerance": tolerance,
        "k_rate_per_minute": rate_per_minute,
        "resamples": resamples,
        "seed": seed,
        "n_videos_in_set": len(videos),
        "selectors": {name: summarize(r, resamples, seed) for name, r in by_selector.items()},
        "skipped": skipped,
        "comparisons": compare(by_selector, resamples, seed),
        "videos": [{k_: v for k_, v in row.items()} for r in by_selector.values() for row in r],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", dest="video_set", choices=["dev", "test", "validation"], default="dev")
    parser.add_argument("--selectors", nargs="+", choices=sorted(SELECTORS), default=["U", "O", "A"])
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--splits", type=Path, default=Path("configs/mtg_ad_splits.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/mtg-ad"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/mtg-ad"))
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-out-of-bounds", action="store_true",
                        help="score against AD starts beyond the declared duration too "
                             "(unrestricted annotation upper bound); default excludes them")
    parser.add_argument("--duration-slack", type=float, default=DURATION_SLACK_SECONDS,
                        help="seconds past the declared duration still counted as inside the clip")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    splits = json.loads(args.splits.read_text())
    try:
        videos = load_video_set(args.video_set, splits, args.data_dir)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nDownload the annotations first: python scripts/prepare_mtg_ad.py"
        ) from exc
    result = run(
        videos, splits["k_rule"]["rate_per_minute"], args.selectors, args.cache_dir,
        args.tolerance, args.resamples, args.seed,
        bounds="keep" if args.keep_out_of_bounds else "exclude", slack=args.duration_slack,
    )
    result["set"] = args.video_set
    output = args.output or Path(f"outputs/selector_eval_{args.video_set}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(format_report(result))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
