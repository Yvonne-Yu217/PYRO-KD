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
    download_annotations, download_video, load_split, select_dev, select_stratified,
)
from src.utils.selection import k_for_duration  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/mtg-ad"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/mtg-ad"))
    parser.add_argument("--splits-out", type=Path, default=Path("configs/mtg_ad_splits.json"))
    parser.add_argument("--num-dev", type=int, default=10)
    parser.add_argument("--num-test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download-videos", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--videos", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()

    download_annotations(args.data_dir)
    validation = load_split(args.data_dir, "validation")
    test = load_split(args.data_dir, "test")
    print(f"validation: {len(validation)} videos, test: {len(test)} videos")

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
    args.splits_out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"K rate {rate}/min; wrote {args.splits_out} (dev {len(dev)}, test {len(test_sample)})")

    if not (args.download_videos or args.preprocess):
        return
    from src.utils.media import preprocess_video

    targets = {"dev": dev, "test": test_sample, "all": dev + test_sample}[args.videos]
    failures = []
    for video in targets:
        path = args.data_dir / "videos" / video.video_filename
        if args.download_videos:
            try:
                path = download_video(video, args.data_dir / "videos")
            except RuntimeError as exc:
                failures.append(video.video_id)
                print(f"FAILED {video.video_id}: {exc}")
                continue
        elif not path.exists():
            failures.append(video.video_id)
            print(f"MISSING {video.video_id}: expected {path}")
            continue
        if args.preprocess:
            manifest = preprocess_video(path, args.cache_dir / video.video_id, fps=args.fps)
            drift = manifest["media_info"]["duration"] - video.duration_seconds
            print(
                f"{video.video_id}: {len(manifest['frames'])} frames,"
                f" audio={bool(manifest['audio_path'])}, duration drift {drift:+.1f}s"
            )
    if failures:
        print(f"{len(failures)} videos unavailable: {' '.join(failures)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
