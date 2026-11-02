"""Download EgoSchema annotations and fix the 10-item pilot.

Videos are not fetched here: download the release zip per the EgoSchema
README (Google Drive or the lmms-lab Hugging Face mirror) into
data/egoschema/videos/<q_uid>.mp4, then pass --preprocess.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.egoschema import download_annotations, load_items, split_pilot  # noqa: E402
from src.utils.media import preprocess_video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/egoschema"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/egoschema"))
    parser.add_argument("--pilot-out", type=Path, default=Path("configs/egoschema_pilot.json"))
    parser.add_argument("--num-pilot", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()

    download_annotations(args.data_dir)
    items = load_items(args.data_dir)
    pilot, rest = split_pilot(items, args.num_pilot, seed=args.seed)
    record = {
        "dataset": "egoschema public subset",
        "seed": args.seed,
        "pilot_q_uids": [item.q_uid for item in pilot],
        "num_heldout": len(rest),
        "note": "pilot items are excluded from the reported local evaluation",
    }
    args.pilot_out.parent.mkdir(parents=True, exist_ok=True)
    args.pilot_out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"{len(items)} labeled items; wrote {args.pilot_out}")

    if args.preprocess:
        for item in pilot:
            video_path = args.data_dir / "videos" / item.video_filename
            if not video_path.exists():
                print(f"missing {video_path}; skipping")
                continue
            manifest = preprocess_video(video_path, args.cache_dir / item.q_uid, fps=args.fps)
            print(f"cached {item.q_uid}: audio={bool(manifest['audio_path'])}")


if __name__ == "__main__":
    main()
