"""Run Silero VAD + librosa on every cached video and write audio_features.json.

Needs torch/silero-vad/librosa (the HPC env, not the macOS test env). Reads
<cache-dir>/<video_id>/manifest.json for the WAV path; videos without audio are
skipped and listed, so A/AV report them instead of silently dropping them.

Existing features are reused only while their config_hash matches the current
VAD config, feature version, manifest hash and WAV identity; anything else is
recomputed. A video that fails (corrupt WAV, broken manifest, VAD error) is
recorded and the batch continues; its stale audio_features.json is removed so
build_moments reports it as missing instead of reusing outdated gaps. The run
writes <cache-dir>/audio_features_report.json and exits 1 if any video failed.

    python scripts/cache_audio_features.py --cache-dir data/cache/mtg-ad
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.audio_features import (  # noqa: E402
    FEATURES_NAME, compute_audio_features, features_are_current, features_config_hash,
    load_features, write_features,
)
from src.utils.media import load_manifest  # noqa: E402

REPORT_NAME = "audio_features_report.json"


def _current_features(video_dir: Path, expected_hash: str) -> dict | None:
    """Cached features if readable and stamped with expected_hash, else None."""
    try:
        features = load_features(video_dir)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return features if features_are_current(features, expected_hash) else None


def cache_video(video_dir: Path, overwrite: bool = False) -> tuple[str, dict | None]:
    """Cache one video's features; returns (status, features).

    status is "computed", "current" (reused), or "no_audio". Raises on any
    failure so the caller can record it and move on.
    """
    manifest = load_manifest(video_dir)
    audio_path = manifest.get("audio_path")
    if not audio_path:
        # A re-extracted video may have lost its audio stream; drop old features.
        (video_dir / FEATURES_NAME).unlink(missing_ok=True)
        return "no_audio", None
    manifest_hash = manifest.get("config_hash")
    expected = features_config_hash(audio_path, None, manifest_hash)
    if not overwrite:
        features = _current_features(video_dir, expected)
        if features is not None:
            return "current", features
    features = compute_audio_features(audio_path, None, manifest_hash)
    write_features(video_dir, features)
    return "computed", features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/mtg-ad"))
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute even when cached features are current")
    args = parser.parse_args(argv)

    counts = {"computed": 0, "current": 0}
    no_audio: list[str] = []
    failed: list[dict[str, str]] = []
    for manifest_path in sorted(args.cache_dir.glob("*/manifest.json")):
        video_dir = manifest_path.parent
        try:
            status, features = cache_video(video_dir, args.overwrite)
        except Exception as exc:  # one bad video must not stop the batch
            error = f"{type(exc).__name__}: {exc}"
            failed.append({"video_id": video_dir.name, "error": error})
            try:
                (video_dir / FEATURES_NAME).unlink(missing_ok=True)
            except OSError:
                pass
            print(f"{video_dir.name}: FAILED {error}", file=sys.stderr)
            continue
        if status == "no_audio":
            no_audio.append(video_dir.name)
            continue
        counts[status] += 1
        if status == "computed":
            print(
                f"{video_dir.name}: speech {features['speech_seconds']:.0f}s of"
                f" {features['duration']:.0f}s, {len(features['gaps'])} usable gaps"
            )

    report = {
        "computed": counts["computed"], "reused": counts["current"],
        "no_audio": no_audio, "failed": failed,
    }
    if args.cache_dir.is_dir():
        report_path = args.cache_dir / REPORT_NAME
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"cached {counts['computed']} videos, reused {counts['current']};"
        f" no audio stream: {' '.join(no_audio) or 'none'}"
    )
    if failed:
        print(f"FAILED {len(failed)} videos: {' '.join(f['video_id'] for f in failed)}"
              f" (details: {args.cache_dir / REPORT_NAME})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
