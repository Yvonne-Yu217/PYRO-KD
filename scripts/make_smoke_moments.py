"""Build a tiny synthetic clip, cache it, and write moments for a GPU smoke test.

Checks model loading, multi-frame generation, and cost logging without any
dataset or YouTube access. The clip is a test pattern, so the descriptions are
meaningless: this validates the pipeline, not description quality.

    python scripts/make_smoke_moments.py
    bash hpc/submit.sh qwen --limit 4 --input data/moments/smoke_U.jsonl
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.media import frame_window, preprocess_video  # noqa: E402
from src.utils.selection import uniform_timestamps  # noqa: E402

DURATION = 12
K = 4


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found on PATH")
    video = Path("data/smoke/smoke.mp4")
    video.parent.mkdir(parents=True, exist_ok=True)
    if not video.exists():
        subprocess.run(
            [
                ffmpeg, "-v", "error", "-y",
                "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=25:duration={DURATION}",
                "-f", "lavfi", "-i", f"sine=frequency=440:duration={DURATION}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(video),
            ],
            check=True,
        )
    manifest = preprocess_video(video, Path("data/cache/smoke/smoke"), fps=2.0)

    output = Path("data/moments/smoke_U.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for i, t in enumerate(uniform_timestamps(DURATION, K)):
            window = frame_window(manifest, t, 4, 2.0)
            handle.write(json.dumps({
                "id": f"smoke_U_{i:03d}",
                "video_id": "smoke",
                "dataset": "smoke",
                "selector": "U",
                "time": t,
                "frames": [f.path for f in window],
                "frame_times": [f.timestamp for f in window],
                "transcript": "",
                "k_requested": K,
                "word_limit": 15,
            }) + "\n")
    print(f"wrote {K} moments to {output}")


if __name__ == "__main__":
    main()
