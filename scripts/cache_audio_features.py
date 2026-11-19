"""Run Silero VAD + librosa on every cached video and write audio_features.json.

Needs torch/silero-vad/librosa (the HPC env, not the macOS test env). Reads
<cache-dir>/<video_id>/manifest.json for the WAV path; videos without audio are
skipped and listed, so A/AV report them instead of silently dropping them.

Existing features are reused only while their config_hash matches the current
VAD config, feature version and WAV contents; anything else is recomputed.
Consumers (build_moments) apply the same check, so stale features are never
used even if this script was not rerun.

A video that fails (corrupt WAV, broken manifest, VAD error) is recorded and
the batch continues. Nothing is ever deleted: a failed video's old features are
rejected downstream by the hash check if they no longer match its WAV, and are
still valid if they do. A missing dependency (ImportError) stops the run at
once, since every later video would fail the same way.

Writes <cache-dir>/audio_features_report.json. Exit status: 0 if every video
with audio has current features, 1 if any video failed or no manifests were
found (e.g. a misspelled --cache-dir), 2 if a dependency is missing.

    python scripts/cache_audio_features.py --cache-dir data/cache/mtg-ad
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.audio_features import (  # noqa: E402
    compute_audio_features, load_current_features, resolve_audio_path, write_features,
)
from src.utils.media import load_manifest  # noqa: E402

REPORT_NAME = "audio_features_report.json"


def cache_video(video_dir: Path, overwrite: bool = False) -> tuple[str, dict | None]:
    """Cache one video's features; returns (status, features).

    status is "computed", "current" (reused), or "no_audio". Raises on any
    failure so the caller can record it and move on.
    """
    manifest = load_manifest(video_dir)
    audio_path = resolve_audio_path(manifest, video_dir)
    if audio_path is None:
        return "no_audio", None
    if not overwrite:
        features, _ = load_current_features(video_dir, manifest)
        if features is not None:
            return "current", features
    features = compute_audio_features(audio_path)
    write_features(video_dir, features)
    return "computed", features


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/mtg-ad"))
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute even when cached features are current")
    args = parser.parse_args(argv)

    manifests = sorted(args.cache_dir.glob("*/manifest.json"))
    if not manifests:
        print(f"no */manifest.json under {args.cache_dir}; check --cache-dir and run"
              " preprocessing first", file=sys.stderr)
        return 1

    counts = {"computed": 0, "current": 0}
    no_audio: list[str] = []
    failed: list[dict[str, str]] = []
    aborted = None
    for manifest_path in manifests:
        video_dir = manifest_path.parent
        try:
            status, features = cache_video(video_dir, args.overwrite)
        except ImportError as exc:  # environment problem: every video would fail
            aborted = f"{type(exc).__name__}: {exc}"
            print(f"missing dependency, stopping before touching more videos: {aborted}",
                  file=sys.stderr)
            break
        except Exception as exc:  # one bad video must not stop the batch
            error = f"{type(exc).__name__}: {exc}"
            failed.append({"video_id": video_dir.name, "error": error})
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
        "no_audio": no_audio, "failed": failed, "aborted": aborted,
    }
    report_path = args.cache_dir / REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"cached {counts['computed']} videos, reused {counts['current']};"
        f" no audio stream: {' '.join(no_audio) or 'none'}"
    )
    if aborted:
        return 2
    if failed:
        print(f"FAILED {len(failed)} videos: {' '.join(f['video_id'] for f in failed)}"
              f" (details: {report_path})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
