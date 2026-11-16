"""Compute shot boundaries and CLIP novelty for every cached video.

Writes visual_features.json and visual_embeddings.npy next to each video's
manifest. Needs torch, open_clip and scenedetect (the HPC env).

- Features computed from the current manifest and config are reused; stale or
  missing ones are recomputed.
- A failure on one video is recorded in <cache-dir>/visual_features_report.json
  and does not stop the others. On failure, features that are still current are
  kept (a failed --overwrite or an environment problem must not destroy them);
  only stale or unreadable features are deleted, so V reports the video as
  missing instead of using outdated values.
- A missing dependency (ImportError) is systematic, not per-video: the run
  stops at once, touches no cached features, and exits 2.
- A missing or empty cache directory exits 1 instead of reporting success.
- If a video's frame cache changes while its features are being computed, the
  result is discarded and recorded as a failure.
- Exit status: 0 all good, 1 some video failed or nothing to do, 2 missing dependency.

    python scripts/cache_visual_features.py --cache-dir data/cache/mtg-ad
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import visual_features as vf  # noqa: E402
from src.utils.media import cache_is_complete  # noqa: E402

REPORT_NAME = "visual_features_report.json"


def _read_manifest(video_dir: Path) -> dict:
    return json.loads((video_dir / "manifest.json").read_text(encoding="utf-8"))


def _still_current(video_dir: Path) -> bool:
    """True if the cached features match the manifest on disk right now."""
    try:
        return vf.load_current_features(video_dir, _read_manifest(video_dir)) is not None
    except Exception:  # unreadable manifest: nothing can be verified
        return False


def _drop_cache(video_dir: Path) -> None:
    for name in (vf.FEATURES_NAME, vf.EMBEDDINGS_NAME):
        try:
            (video_dir / name).unlink(missing_ok=True)
        except OSError:
            pass


def process_video(
    video_dir: Path,
    get_embedder: Callable[[], Callable],
    overwrite: bool = False,
) -> tuple[str, dict | None]:
    """Cache one video; returns ("computed" | "reused", features). Raises on failure."""
    manifest = _read_manifest(video_dir)
    if not overwrite and vf.is_fresh(vf.load_features(video_dir), manifest):
        return "reused", vf.load_features(video_dir)
    if not cache_is_complete(manifest):
        raise RuntimeError(
            "frame cache is incomplete or its paths do not resolve from this working "
            "directory (run from the repo root, or re-run the prepare script)"
        )
    features, embeddings = vf.compute_visual_features(manifest, get_embedder())
    # The frames and manifest may have been re-extracted while CLIP was running.
    if _read_manifest(video_dir).get("config_hash") != manifest.get("config_hash"):
        raise RuntimeError("frame cache changed while features were being computed")
    vf.write_features(video_dir, features, embeddings)
    return "computed", features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/mtg-ad"))
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute even when cached features are current")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    manifests = sorted(args.cache_dir.glob("*/manifest.json")) if args.cache_dir.is_dir() else []
    if not manifests:
        print(
            f"no cached videos found under {args.cache_dir}; run the prepare script "
            "with --preprocess first",
            file=sys.stderr,
        )
        return 1

    embedder = None

    def get_embedder():
        nonlocal embedder
        if embedder is None:  # load CLIP once, on first use
            embedder = vf.clip_embedder(batch_size=args.batch_size)
        return embedder

    counts = {"computed": 0, "reused": 0}
    failures: dict[str, str] = {}
    try:
        for manifest_path in manifests:
            video_dir = manifest_path.parent
            try:
                status, features = process_video(video_dir, get_embedder, args.overwrite)
            except ImportError:
                raise  # missing dependency: systematic, not this video's fault
            except Exception as exc:  # one bad video must not stop the rest
                failures[video_dir.name] = f"{type(exc).__name__}: {exc}"
                kept = _still_current(video_dir)
                if not kept:
                    _drop_cache(video_dir)
                print(
                    f"FAILED {video_dir.name}: {failures[video_dir.name]}"
                    f" ({'kept current features' if kept else 'removed stale features'})",
                    file=sys.stderr,
                )
                continue
            counts[status] += 1
            if status == "computed":
                print(
                    f"{video_dir.name}: {features['num_frames']} frames, "
                    f"{len(features['shots'])} shot boundaries, "
                    f"max novelty {max(features['novelty']):.3f}"
                )
    except ImportError as exc:
        print(f"missing dependency: {exc}. Run this inside the HPC environment; "
              "no cached features were changed.", file=sys.stderr)
        return 2

    report = args.cache_dir / REPORT_NAME
    report.write_text(
        json.dumps({**counts, "failures": failures}, indent=2) + "\n", encoding="utf-8")
    print(
        f"computed {counts['computed']}, reused {counts['reused']}, "
        f"failed {len(failures)}; report: {report}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
