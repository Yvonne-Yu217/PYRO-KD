"""Turn a selector's timestamps into generation inputs (one JSONL row per moment).

Each row carries everything the frozen VLM sees, so every selector gets an
identical frame window, transcript-context rule, and word limit:
    {"id", "video_id", "selector", "time", "frames", "frame_times",
     "transcript", "word_limit"}

Only the uniform selector (U) exists so far; V/A/AV/O plug in here in Week 3.

    python scripts/build_moments.py --selector U --videos dev
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.mtg_ad import load_split  # noqa: E402
from src.generation.prompts import transcript_context  # noqa: E402
from src.utils.media import frame_window, load_manifest  # noqa: E402
from src.utils.selection import k_for_duration, uniform_timestamps  # noqa: E402

SELECTORS = {"U": lambda video, k: uniform_timestamps(video.duration_seconds, k)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selector", choices=sorted(SELECTORS), default="U")
    parser.add_argument("--videos", choices=["dev", "test"], default="dev")
    parser.add_argument("--splits", type=Path, default=Path("configs/mtg_ad_splits.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/mtg-ad"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/mtg-ad"))
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--span", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    splits = json.loads(args.splits.read_text())
    rate = splits["k_rule"]["rate_per_minute"]
    if args.videos == "dev":
        wanted = [v["video_id"] for v in splits["dev"]["videos"]]
        videos = {v.video_id: v for v in load_split(args.data_dir, "validation")}
    else:
        wanted = splits["test"]["video_ids"]
        videos = {v.video_id: v for v in load_split(args.data_dir, "test")}

    output = args.output or Path(f"data/moments/{args.selector}_{args.videos}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, missing = 0, []
    with output.open("w") as handle:
        for video_id in wanted:
            video = videos[video_id]
            cache = args.cache_dir / video_id
            if not (cache / "manifest.json").exists():
                missing.append(video_id)
                continue
            manifest = load_manifest(cache)
            k = k_for_duration(video.duration_seconds, rate)
            for i, t in enumerate(SELECTORS[args.selector](video, k)):
                window = frame_window(manifest, t, args.num_frames, args.span)
                handle.write(json.dumps({
                    "id": f"{video_id}_{args.selector}_{i:03d}",
                    "video_id": video_id,
                    "selector": args.selector,
                    "time": t,
                    "frames": [f.path for f in window],
                    "frame_times": [f.timestamp for f in window],
                    "transcript": transcript_context(video.transcript, t),
                    "word_limit": None,  # gap-based limit arrives with the VAD cache
                }) + "\n")
                rows += 1
    print(f"wrote {rows} moments to {output}")
    if missing:
        print(f"no cache for {len(missing)} videos (run prepare_mtg_ad.py --preprocess): {' '.join(missing)}")


if __name__ == "__main__":
    main()
