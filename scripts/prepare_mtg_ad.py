"""Download MTG-AD annotations, fix the dev/test videos and K rule, and
optionally fetch + cache videos.

Examples:
    # annotations + configs/mtg_ad_splits.json only (seconds, ~10 MB)
    python scripts/prepare_mtg_ad.py

    # also download (yt-dlp) and preprocess the dev videos; run on a CPU node
    python scripts/prepare_mtg_ad.py --download-videos --preprocess --videos dev

    # preprocess videos you already placed in data/mtg-ad/videos/<video_id>.mp4
    python scripts/prepare_mtg_ad.py --preprocess --videos dev
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.mtg_ad import (  # noqa: E402
    EXPECTED_SPLIT_COUNTS, download_annotations, download_video, load_split, select_dev,
    select_stratified,
)
from src.utils.media import MANIFEST_NAME, preprocess_video  # noqa: E402
from src.utils.selection import k_for_duration  # noqa: E402


def prepare_targets(
    targets, data_dir, cache_dir, download_videos, preprocess, fps,
    cookies=None, cookies_from_browser=None, extractor_args=None,
):
    """Download/preprocess a batch, continuing after per-video failures."""
    results = []
    for video in targets:
        path = data_dir / "videos" / video.video_filename
        item = {"video_id": video.video_id, "source": str(path)}
        existed = path.is_file() and path.stat().st_size > 0
        source_mtime = path.stat().st_mtime_ns if existed else None
        if download_videos:
            try:
                path = download_video(
                    video, data_dir / "videos",
                    cookies=cookies,
                    cookies_from_browser=cookies_from_browser,
                    extractor_args=extractor_args,
                )
                reused_source = (
                    source_mtime is not None and path.stat().st_mtime_ns == source_mtime
                )
                item["download"] = "reused" if reused_source else "downloaded"
            except (OSError, RuntimeError) as exc:
                item.update(status="unavailable", stage="download", error=str(exc))
                results.append(item)
                print(f"FAILED {video.video_id}: {exc}")
                continue
        elif not existed:
            item.update(status="unavailable", stage="source", error=f"missing {path}")
            results.append(item)
            print(f"MISSING {video.video_id}: expected {path}")
            continue
        else:
            item["download"] = "existing"

        if not preprocess:
            item.update(status="ready", preprocess="skipped")
            results.append(item)
            continue

        manifest_path = cache_dir / video.video_id / MANIFEST_NAME
        previous_mtime = manifest_path.stat().st_mtime_ns if manifest_path.exists() else None
        try:
            manifest = preprocess_video(path, cache_dir / video.video_id, fps=fps)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            item.update(status="failed", stage="preprocess", error=str(exc))
            results.append(item)
            print(f"FAILED {video.video_id} preprocessing: {exc}")
            continue
        reused = previous_mtime is not None and manifest_path.stat().st_mtime_ns == previous_mtime
        drift = manifest["media_info"]["duration"] - video.duration_seconds
        item.update(
            status="ready",
            preprocess="reused" if reused else "written",
            manifest=str(manifest_path),
            frames=len(manifest["frames"]),
            audio=bool(manifest["audio_path"]),
            duration_drift_seconds=round(drift, 3),
        )
        results.append(item)
        print(
            f"{video.video_id}: {len(manifest['frames'])} frames,"
            f" audio={bool(manifest['audio_path'])}, duration drift {drift:+.1f}s"
            f" ({item['preprocess']} cache)"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/mtg-ad"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/mtg-ad"))
    parser.add_argument("--splits-out", type=Path, default=Path("configs/mtg_ad_splits.json"))
    parser.add_argument("--num-dev", type=int, default=10)
    parser.add_argument("--num-test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download-videos", action="store_true")
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument(
        "--cookies", type=Path,
        help="yt-dlp Netscape cookie file; keep it outside the repository",
    )
    auth.add_argument(
        "--cookies-from-browser",
        help="read cookies from a local browser, e.g. chrome or firefox",
    )
    parser.add_argument(
        "--yt-dlp-extractor-args",
        help="optional yt-dlp extractor args, e.g. youtube:player_client=mweb,tv",
    )
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--videos", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument(
        "--report", type=Path, default=None,
        help="JSON status report (default: <data-dir>/preprocess_report.json)",
    )
    args = parser.parse_args()

    download_annotations(args.data_dir)
    validation = load_split(args.data_dir, "validation")
    test = load_split(args.data_dir, "test")
    print(f"validation: {len(validation)} videos, test: {len(test)} videos")
    expected_validation = EXPECTED_SPLIT_COUNTS["validation"]
    expected_test = EXPECTED_SPLIT_COUNTS["test"]
    if len(validation) != expected_validation or len(test) != expected_test:
        print(
            f"refusing to write {args.splits_out}: "
            f"validation={len(validation)} (expected {expected_validation}), "
            f"test={len(test)} (expected {expected_test})",
            file=sys.stderr,
        )
        sys.exit(1)

    # K rule and dev set come from validation only; test is sampled, not inspected.
    rate = round(statistics.median(len(v.ad) / (v.duration_seconds / 60) for v in validation), 2)
    dev = select_dev(validation, args.num_dev, seed=args.seed)
    test_sample = select_stratified(test, args.num_test, seed=args.seed)
    record = {
        "dataset": "google/mind-the-gap-inline-ad",
        "seed": args.seed,
        "k_rule": {
            "formula": "K = max(1, round(rate_per_minute * duration_seconds / 60))",
            "rate_per_minute": rate,
            "source": "median GT AD segments per minute over the 332 validation videos",
        },
        "dev": {
            "split": "validation",
            "selection": "category-stratified seeded sample of validation videos with >=3 AD segments",
            "videos": [
                {
                    "video_id": v.video_id,
                    "category": v.category,
                    "duration_seconds": v.duration_seconds,
                    "num_ad": len(v.ad),
                    "has_speech": bool(v.transcript),
                    "k": k_for_duration(v.duration_seconds, rate),
                }
                for v in dev
            ],
        },
        "exemplar_pool": {
            "split": "validation",
            "rule": "any validation video not in dev; never test",
        },
        "test": {
            "split": "test",
            "selection": "category-stratified seeded sample of the 101 official test videos",
            "video_ids": [v.video_id for v in test_sample],
        },
    }
    args.splits_out.parent.mkdir(parents=True, exist_ok=True)
    args.splits_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"K rate {rate}/min; wrote {args.splits_out} (dev {len(dev)}, test {len(test_sample)})")

    if not (args.download_videos or args.preprocess):
        return
    targets = {"dev": dev, "test": test_sample, "all": dev + test_sample}[args.videos]
    results = prepare_targets(
        targets, args.data_dir, args.cache_dir,
        args.download_videos, args.preprocess, args.fps,
        args.cookies, args.cookies_from_browser, args.yt_dlp_extractor_args,
    )
    failures = [item["video_id"] for item in results if item["status"] != "ready"]
    report_path = args.report or args.data_dir / "preprocess_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "dataset": "mtg-ad",
        "selection": args.videos,
        "num_requested": len(targets),
        "num_ready": len(targets) - len(failures),
        "num_failed": len(failures),
        "videos": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {report_path}")
    if failures:
        print(f"{len(failures)} videos unavailable: {' '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
