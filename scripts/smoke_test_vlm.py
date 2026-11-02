"""Smoke-test a served VLM on cached pilot frames and log latency/tokens.

Requires a running OpenAI-compatible server (scripts/hpc/vlm_smoke_test.sbatch
starts one). Writes one JSON record per call to outputs/smoke/.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.vlm_client import QWEN3_VL_8B, VLMClient  # noqa: E402
from src.utils.media import frame_window, load_manifest, preprocess_video  # noqa: E402

PROMPT = (
    "These {n} frames span about {span:.0f} seconds of a movie, in order. "
    "Write one audio description sentence (at most {words} words) describing "
    "the most important visible action. Describe only what is visible."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=QWEN3_VL_8B)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cache", type=Path, help="preprocessed clip dir with manifest.json")
    source.add_argument("--video", type=Path, help="video to preprocess first")
    parser.add_argument("--times", type=float, nargs="+", default=[10.0, 30.0, 60.0])
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--span", type=float, default=2.0)
    parser.add_argument("--words", type=int, default=15)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/smoke"))
    args = parser.parse_args()

    if args.video:
        cache = Path("data/cache/smoke") / args.video.stem
        manifest = preprocess_video(args.video, cache)
    else:
        manifest = load_manifest(args.cache)

    client = VLMClient(model=args.model, base_url=args.base_url)
    out_dir = args.out_dir / args.model.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for t in args.times:
        window = frame_window(manifest, t, args.num_frames, args.span)
        prompt = PROMPT.format(n=len(window), span=args.span, words=args.words)
        result = client.generate([f.path for f in window], prompt, max_tokens=4 * args.words)
        record = {
            "video_id": manifest["video_id"],
            "center_time": t,
            "frame_times": [f.timestamp for f in window],
            "text": result.text,
            "model": result.model,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "latency_seconds": round(result.latency_seconds, 3),
            "request": result.request,
        }
        records.append(record)
        print(json.dumps({k: record[k] for k in (
            "center_time", "text", "prompt_tokens", "completion_tokens", "latency_seconds",
        )}))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"{manifest['video_id']}-{stamp}.json"
    out_path.write_text(json.dumps(records, indent=2))
    latencies = [r["latency_seconds"] for r in records]
    print(f"wrote {out_path}; mean latency {sum(latencies) / len(latencies):.2f}s")


if __name__ == "__main__":
    main()
