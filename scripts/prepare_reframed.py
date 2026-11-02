"""Download REFRAMED metadata, fix the pilot, and optionally fetch + cache videos.

Examples:
    # metadata + pilot list only (seconds, no video traffic)
    uv run python scripts/prepare_reframed.py

    # also download and preprocess the 10 pilot videos (run on HPC)
    uv run python scripts/prepare_reframed.py --download-videos --preprocess
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.reframed import (  # noqa: E402
    download_metadata, download_video, load_split, merge_cues, select_pilot,
)
from src.utils.media import preprocess_video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/reframed"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/reframed"))
    parser.add_argument("--pilot-out", type=Path, default=Path("configs/reframed_pilot.json"))
    parser.add_argument("--num-pilot", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download-videos", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()

    download_metadata(args.data_dir)
    validation = load_split(args.data_dir, "validation")
    test = load_split(args.data_dir, "test")
    print(f"validation: {len(validation)} clips / {len({c.movie_id for c in validation})} movies")
    print(f"test:       {len(test)} clips / {len({c.movie_id for c in test})} movies")

    # K rule is derived from validation only (proposal: never set K from test).
    rates = [len(merge_cues(c.ad_us)) / (c.video_duration / 60) for c in validation]
    pilot = select_pilot(validation, args.num_pilot, seed=args.seed)
    record = {
        "dataset": "igorsterner/reframed",
        "split": "validation",
        "seed": args.seed,
        "selection": "one clip per movie, seeded shuffle; >=3 US AD cues; has subtitles",
        "k_rule": {
            "formula": "K = max(1, round(rate_per_minute * video_duration / 60))",
            "rate_per_minute": round(statistics.median(rates), 2),
            "source": "median merged US AD descriptions/min on validation (merge gap 0.5 s)",
        },
        "clips": [
            {
                "video_id": c.video_id,
                "movie": c.rottentomatoes_id,
                "imdb_id": c.imdb_id,
                "video_duration": c.video_duration,
                "ad_us_cues": len(c.ad_us),
                "ad_us_descriptions": len(merge_cues(c.ad_us)),
                "ad_uk_cues": len(c.ad_uk),
            }
            for c in pilot
        ],
    }
    args.pilot_out.parent.mkdir(parents=True, exist_ok=True)
    args.pilot_out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.pilot_out} ({len(pilot)} clips)")

    if not (args.download_videos or args.preprocess):
        return
    for clip in pilot:
        video_path = download_video(clip, args.data_dir / "videos")
        print(f"downloaded {video_path}")
        if args.preprocess:
            manifest = preprocess_video(video_path, args.cache_dir / clip.video_id, fps=args.fps)
            info = manifest["media_info"]
            drift = info["duration"] - clip.video_duration
            print(
                f"  cached {len(manifest['frames'])} frames, audio={bool(manifest['audio_path'])},"
                f" duration drift vs metadata {drift:+.2f}s"
            )


if __name__ == "__main__":
    main()
