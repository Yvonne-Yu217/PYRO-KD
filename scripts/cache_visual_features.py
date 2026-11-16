"""Compute shot boundaries and CLIP novelty for every cached video.

Writes visual_features.json and visual_embeddings.npy next to each video's
manifest. Needs torch, open_clip and scenedetect (the HPC env). Features
computed from the current manifest and config are reused; stale or missing ones
are recomputed. A failure on one video is recorded in
<cache-dir>/visual_features_report.json and does not stop the others; the
script exits 1 if any video failed, and a failed video's old features are
deleted so V reports it as missing instead of using outdated values.

    python scripts/cache_visual_features.py --cache-dir data/cache/mtg-ad
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import visual_features as vf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/mtg-ad"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    embed = None
    done, reused, failures = 0, 0, {}
    for manifest_path in sorted(args.cache_dir.glob("*/manifest.json")):
        video_dir = manifest_path.parent
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not args.overwrite and vf.is_fresh(vf.load_features(video_dir), manifest):
                reused += 1
                continue
            if embed is None:  # load CLIP once, on first use
                embed = vf.clip_embedder(batch_size=args.batch_size)
            features, embeddings = vf.compute_visual_features(manifest, embed)
            vf.write_features(video_dir, features, embeddings)
            done += 1
            print(
                f"{video_dir.name}: {features['num_frames']} frames, "
                f"{len(features['shots'])} shot boundaries, "
                f"max novelty {max(features['novelty']):.3f}"
            )
        except Exception as exc:  # one bad video must not stop the rest
            failures[video_dir.name] = f"{type(exc).__name__}: {exc}"
            (video_dir / vf.FEATURES_NAME).unlink(missing_ok=True)
            (video_dir / vf.EMBEDDINGS_NAME).unlink(missing_ok=True)
            print(f"FAILED {video_dir.name}: {failures[video_dir.name]}")

    report = args.cache_dir / "visual_features_report.json"
    report.write_text(json.dumps({"computed": done, "reused": reused, "failures": failures}, indent=2) + "\n")
    print(f"computed {done}, reused {reused}, failed {len(failures)}; report: {report}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
