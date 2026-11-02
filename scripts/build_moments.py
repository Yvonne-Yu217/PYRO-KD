"""Turn a selector's timestamps into generation inputs (one JSONL row per moment).

Each row carries everything the frozen VLM sees, so every selector gets an
identical frame window, transcript-context rule, and word limit:
    {"id", "video_id", "selector", "time", "frames", "frame_times",
     "transcript", "word_limit"}

Only the uniform selector (U) exists so far; V/A/AV/O plug in here in Week 3.

    python scripts/build_moments.py --selector U --videos dev                  # MTG-AD
    python scripts/build_moments.py --dataset reframed --videos dev            # REFRAMED
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import mtg_ad, reframed  # noqa: E402
from src.generation.prompts import transcript_context  # noqa: E402
from src.utils.media import frame_window, load_manifest  # noqa: E402
from src.utils.selection import k_for_duration, uniform_timestamps  # noqa: E402

SELECTORS = {"U": lambda duration, k: uniform_timestamps(duration, k)}


def load_targets(dataset: str, which: str, splits: dict, data_dir: Path):
    """Return [(video_id, duration_seconds, transcript_items)] for the dev or test set."""
    if dataset == "mtg-ad":
        pool_split = "validation" if which == "dev" else "test"
        videos = {v.video_id: v for v in mtg_ad.load_split(data_dir, pool_split)}
        wanted = ([v["video_id"] for v in splits["dev"]["videos"]]
                  if which == "dev" else splits["test"]["video_ids"])
        return [(i, videos[i].duration_seconds, videos[i].transcript) for i in wanted]
    pool_split = "validation" if which == "dev" else "test"
    clips = {c.video_id: c for c in reframed.load_split(data_dir, pool_split)}
    wanted = ([c["video_id"] for c in splits["clips"]]
              if which == "dev" else splits["test"]["video_ids"])
    return [(i, clips[i].video_duration, clips[i].subtitles) for i in wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=["mtg-ad", "reframed"], default="mtg-ad")
    parser.add_argument("--selector", choices=sorted(SELECTORS), default="U")
    parser.add_argument("--videos", choices=["dev", "test"], default="dev")
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--span", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    name = args.dataset
    splits_path = args.splits or Path(
        "configs/mtg_ad_splits.json" if name == "mtg-ad" else "configs/reframed_splits.json")
    data_dir = args.data_dir or Path(f"data/{name}")
    cache_dir = args.cache_dir or Path(f"data/cache/{name}")
    splits = json.loads(splits_path.read_text())
    rate = splits["k_rule"]["rate_per_minute"]
    targets = load_targets(name, args.videos, splits, data_dir)

    output = args.output or Path(f"data/moments/{name}_{args.selector}_{args.videos}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, missing = 0, []
    with output.open("w") as handle:
        for video_id, duration, transcript in targets:
            cache = cache_dir / video_id
            if not (cache / "manifest.json").exists():
                missing.append(video_id)
                continue
            manifest = load_manifest(cache)
            k = k_for_duration(duration, rate)
            for i, t in enumerate(SELECTORS[args.selector](duration, k)):
                window = frame_window(manifest, t, args.num_frames, args.span)
                handle.write(json.dumps({
                    "id": f"{video_id}_{args.selector}_{i:03d}",
                    "video_id": video_id,
                    "dataset": name,
                    "selector": args.selector,
                    "time": t,
                    "frames": [f.path for f in window],
                    "frame_times": [f.timestamp for f in window],
                    "transcript": transcript_context(transcript, t),
                    "word_limit": None,  # gap-based limit arrives with the VAD cache
                }) + "\n")
                rows += 1
    print(f"wrote {rows} moments to {output}")
    if missing:
        print(f"no cache for {len(missing)} videos (run the matching prepare script with --preprocess): {' '.join(missing)}")


if __name__ == "__main__":
    main()
