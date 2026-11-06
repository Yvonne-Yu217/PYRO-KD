"""Run Silero VAD + librosa on every cached video and write audio_features.json.

Needs torch/silero-vad/librosa (the HPC env, not the macOS test env). Reads
<cache-dir>/<video_id>/manifest.json for the WAV path; videos without audio are
skipped and listed, so A/AV report them instead of silently dropping them.

    python scripts/cache_audio_features.py --cache-dir data/cache/mtg-ad
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.audio_features import FEATURES_NAME, compute_audio_features, write_features  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/mtg-ad"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    no_audio, done = [], 0
    for manifest_path in sorted(args.cache_dir.glob("*/manifest.json")):
        video_dir = manifest_path.parent
        if (video_dir / FEATURES_NAME).exists() and not args.overwrite:
            continue
        audio_path = json.loads(manifest_path.read_text()).get("audio_path")
        if not audio_path:
            no_audio.append(video_dir.name)
            continue
        features = compute_audio_features(audio_path)
        write_features(video_dir, features)
        done += 1
        print(
            f"{video_dir.name}: speech {features['speech_seconds']:.0f}s of {features['duration']:.0f}s,"
            f" {len(features['gaps'])} usable gaps"
        )
    print(f"cached {done} videos; no audio stream: {' '.join(no_audio) or 'none'}")


if __name__ == "__main__":
    main()
