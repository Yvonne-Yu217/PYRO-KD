"""Turn a selector's timestamps into generation inputs (one JSONL row per moment).

Each row carries everything the frozen VLM sees, so every selector gets an
identical frame window, transcript-context rule, and word limit:
    {"id", "video_id", "selector", "time", "frames", "frame_times",
     "transcript", "word_limit"}

Selectors so far: U (uniform) and A (audio-only). A needs audio_features.json
(scripts/cache_audio_features.py). The word limit is floor(gap * 3), capped,
where gap is the time from the moment to the next speech onset, computed the
same way for every selector from the cached VAD segments.
Zero-word moments are skipped and recorded as not safely speakable in a
<output>.report.json sidecar; they are never sent to the VLM. Missing audio
features produce zero A moments for that video, without aborting other videos.

    python scripts/build_moments.py --selector U --videos dev                  # MTG-AD
    python scripts/build_moments.py --dataset reframed --videos dev            # REFRAMED
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import mtg_ad, reframed  # noqa: E402
from src.generation.prompts import transcript_context, word_limit_for_gap  # noqa: E402
from src.utils.audio_features import available_gap, load_features, speech_of  # noqa: E402
from src.utils.audio_selector import AudioSelector  # noqa: E402
from src.utils.media import frame_window, load_manifest  # noqa: E402
from src.utils.selection import k_for_duration, uniform_timestamps  # noqa: E402


def _select_audio(duration, k, features):
    if features is None:
        return []
    return [c.time for c in AudioSelector().select(features, k)]


SELECTORS = {
    "U": lambda duration, k, features: uniform_timestamps(duration, k),
    "A": _select_audio,
}


def load_targets(dataset: str, which: str, splits: dict, data_dir: Path):
    """Return [(video_id, duration_seconds, transcript_items)] for the dev or test set."""
    if dataset == "mtg-ad":
        pool_split = "validation" if which == "dev" else "test"
        videos = {v.video_id: v for v in mtg_ad.load_split(data_dir, pool_split)}
        wanted = ([v["video_id"] for v in splits["dev"]["videos"]]
                  if which == "dev" else splits["test"]["video_ids"])
        return [(i, videos[i].duration_seconds, videos[i].transcript) for i in wanted]
    pool_split = "validation" if which == "dev" else "test"
    if which == "test" and splits["test"].get("selection_version") != reframed.TEST_SELECTION_VERSION:
        raise ValueError("legacy REFRAMED test selection; regenerate with scripts/prepare_reframed.py")
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
    rows, missing, shortfalls, no_audio, unsafe = 0, [], [], [], []
    reports = []
    with output.open("w") as handle:
        for video_id, duration, transcript in targets:
            k = k_for_duration(duration, rate)
            report = {
                "video_id": video_id, "k_requested": k, "num_selected": 0,
                "num_written": 0, "selection_shortfall": k, "k_shortfall": k,
                "status": "missing_cache", "skipped_moments": [],
            }
            reports.append(report)
            cache = cache_dir / video_id
            if not (cache / "manifest.json").exists():
                missing.append(video_id)
                continue
            manifest = load_manifest(cache)
            features = load_features(cache)
            if args.selector == "A" and "audio_path" in manifest and not manifest["audio_path"]:
                features = None
            speech = speech_of(features) if features else None
            if args.selector == "A" and features is None:
                no_audio.append(video_id)
                report["status"] = "missing_audio_features"
            else:
                report["status"] = "ok"
            times = SELECTORS[args.selector](duration, k, features)
            report["num_selected"] = len(times)
            report["selection_shortfall"] = (
                0 if report["status"] == "missing_audio_features" else k - len(times)
            )
            if report["selection_shortfall"]:
                shortfalls.append((video_id, k - len(times)))
            for i, t in enumerate(times):
                gap = available_gap(t, speech, duration) if speech is not None else None
                word_limit = word_limit_for_gap(gap) if gap is not None else None
                if word_limit == 0:
                    unsafe.append((video_id, t))
                    report["skipped_moments"].append({
                        "time": t, "gap_seconds": gap, "word_limit": 0,
                        "status": "not_safely_speakable",
                    })
                    continue
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
                    "k_requested": k,
                    "word_limit": word_limit,
                }) + "\n")
                rows += 1
                report["num_written"] += 1
            report["k_shortfall"] = k - report["num_written"]
            if report["k_shortfall"] and report["status"] == "ok":
                report["status"] = "shortfall"
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps({
        "dataset": name, "selector": args.selector, "videos": reports,
    }, indent=2) + "\n")
    print(f"wrote {rows} moments to {output}")
    if shortfalls:
        print(f"selection shortfall (fewer than K feasible moments): {', '.join(f'{v} -{n}' for v, n in shortfalls)}")
    if no_audio:
        print(f"no audio/features; zero A moments: {' '.join(no_audio)} (run scripts/cache_audio_features.py)")
    if unsafe:
        print(f"not safely speakable; skipped {len(unsafe)} zero-word moments; details: {report_path}")
    if missing:
        print(f"no cache for {len(missing)} videos (run the matching prepare script with --preprocess): {' '.join(missing)}")


if __name__ == "__main__":
    main()
