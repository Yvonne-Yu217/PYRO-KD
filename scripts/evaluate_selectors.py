#!/usr/bin/env python
"""Selector-only timing evaluation on MTG-AD: no VLM, no GPU.

Scores where each selector places its K moments against where the professional
describers started their descriptions (temporal precision/recall/F1 within
+/- tau seconds, one-to-one matching). This answers RQ1(a) and, with the
oracle, RQ4 (headroom) before any model is run.

    python scripts/evaluate_selectors.py --set dev
    python scripts/evaluate_selectors.py --set validation --selectors U O
    python scripts/evaluate_selectors.py --set test --output outputs/selector_eval_test.json

Selectors: U (uniform), O (oracle: human start times capped at K), A (audio-only;
needs audio_features.json from scripts/cache_audio_features.py) and V (video-only;
needs visual_features.json from scripts/cache_visual_features.py). Videos without
a selector's features are skipped for that selector and counted. AV is added to
SELECTORS when it is implemented. U and O need only the
annotations, so they can be scored on all 332 validation videos. K comes from
the same duration rule as generation (configs/mtg_ad_splits.json), so the
comparison uses the same budget everywhere. Tune nothing on --set test.

Each per-video value is one observation: confidence intervals are paired
percentile bootstraps over videos, never over moments.
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
from src.utils.audio_features import available_gap, load_features, speech_of  # noqa: E402
from src.utils.audio_selector import AudioSelector  # noqa: E402
from src.utils.selection import k_for_duration, oracle_timestamps, uniform_timestamps  # noqa: E402
from src.utils.visual_features import (  # noqa: E402
    VisualCueSelector, load_current_features as load_visual_features, load_embeddings,
    mean_pairwise_similarity,
)

TOLERANCE_SWEEP = (0.0, 1.0, 2.0, 3.0, 5.0)
# (selector, baseline) pairs reported as paired differences in F1.
COMPARISONS = (("A", "U"), ("V", "U"), ("O", "U"), ("V", "A"), ("O", "A"), ("O", "V"))


def _times_u(video, k, features, visual=None):
    return uniform_timestamps(video.duration_seconds, k)


def _times_o(video, k, features, visual=None):
    return oracle_timestamps([item.timestamp for item in video.ad], k)


def _times_a(video, k, features, visual=None):
    if features is None:
        return None
    return [candidate.time for candidate in AudioSelector().select(features, k)]


def _times_v(video, k, features, visual=None):
    if visual is None:
        return None
    return [candidate.time for candidate in VisualCueSelector().select(visual, k)]


SELECTORS: dict[str, Callable[..., list[float] | None]] = {
    "U": _times_u, "O": _times_o, "A": _times_a, "V": _times_v,
}


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
    visual: dict | None = None,
    embeddings=None,
) -> dict[str, Any] | None:
    """Score one selector on one video; None if it cannot run (no features for A or V).

    features are the audio features (also used for the in-speech rate); visual are
    the visual features; embeddings enable the visual-redundancy diagnostic.
    """
    times = SELECTORS[selector](video, k, features, visual)
    if times is None:
        return None
    truth = [item.timestamp for item in video.ad]
    primary = temporal_precision_recall_f1(times, truth, tolerance)
    return {
        "video_id": video.video_id,
        "selector": selector,
        "k": k,
        "num_gt": len(truth),
        "num_selected": len(times),
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
        "visual_redundancy": (
            mean_pairwise_similarity(embeddings, visual["frame_times"], times)
            if embeddings is not None and visual is not None else None
        ),
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
        "total_gt": sum(r["num_gt"] for r in rows),
        "videos_with_shortfall": sum(r["shortfall"] > 0 for r in rows),
        "total_shortfall": sum(max(r["shortfall"], 0) for r in rows),
        "macro_f1_by_tolerance": {
            tau: _mean([r["f1_by_tolerance"][tau] for r in rows])
            for tau in (rows[0]["f1_by_tolerance"] if rows else {})
        },
        "in_speech_rate": _mean([r["in_speech_rate"] for r in rows]),
        "n_with_audio_features": sum(r["in_speech_rate"] is not None for r in rows),
        "visual_redundancy": _mean([r["visual_redundancy"] for r in rows]),
        "n_with_embeddings": sum(r["visual_redundancy"] is not None for r in rows),
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
        f"{'short':>7}{'in-speech':>11}{'redund.':>9}",
    ]
    for name, s in result["selectors"].items():
        f1 = s["macro_f1"]
        ci = f"{_fmt(f1['mean'])} [{_fmt(f1['ci_low'])}, {_fmt(f1['ci_high'])}]"
        lines.append(
            f"{name:<4}{s['n_videos']:>5}  {ci:<24}{_fmt(s['macro_precision']['mean']):>7}"
            f"{_fmt(s['macro_recall']['mean']):>7}{_fmt(s['mean_temporal_error_seconds'], 2):>12}"
            f"{s['videos_with_shortfall']:>7}{_fmt(s['in_speech_rate']):>11}{_fmt(s['visual_redundancy']):>9}"
        )
    for name, count in result["skipped"].items():
        lines.append(f"{name}: skipped {count} video(s) without {'visual' if name == 'V' else 'audio'} features")
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
) -> dict[str, Any]:
    rows: dict[str, list[dict]] = {name: [] for name in selectors}
    skipped: dict[str, int] = {}
    for video in videos:
        k = k_for_duration(video.duration_seconds, rate_per_minute)
        features = load_features(cache_dir / video.video_id)
        visual = load_visual_features(cache_dir / video.video_id) if "V" in selectors else None
        embeddings = load_embeddings(cache_dir / video.video_id) if visual is not None else None
        for name in selectors:
            row = evaluate_video(name, video, k, features, tolerance, visual=visual, embeddings=embeddings)
            if row is None:
                skipped[name] = skipped.get(name, 0) + 1
            else:
                rows[name].append(row)
    by_selector = {name: r for name, r in rows.items() if r}
    return {
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
    parser.add_argument("--selectors", nargs="+", choices=sorted(SELECTORS), default=["U", "O", "A", "V"])
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--splits", type=Path, default=Path("configs/mtg_ad_splits.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/mtg-ad"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/mtg-ad"))
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
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
    )
    result["set"] = args.video_set
    output = args.output or Path(f"outputs/selector_eval_{args.video_set}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(format_report(result))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
