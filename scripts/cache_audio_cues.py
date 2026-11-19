#!/usr/bin/env python
"""Run the audio-cue streams (VAD-gated Whisper, AST tagger, CLAP, librosa
salience, speech cues) on every cached video and write audio_cues.json.

Needs audio_features.json (scripts/cache_audio_features.py) first: Whisper is
gated by its VAD speech, and long-silence cues come from its gaps. Models load
once per run; videos whose cues are current for this WAV, VAD result, config
and backend set are skipped (resume). A failed video is recorded and the
batch continues; a missing dependency stops the run (exit 2) because every
later video would fail the same way. Writes <cache-dir>/audio_cues_report.json.

    # GPU node (Whisper large-v3 + AST + CLAP):
    python scripts/cache_audio_cues.py --cache-dir "$PYRO_CACHE/mtg-ad" --device cuda
    # CPU-only streams (librosa salience, deictic phrases from the dataset ASR):
    python scripts/cache_audio_cues.py --cache-dir data/cache/mtg-ad --no-asr --no-tagger --no-clap \\
        --dataset-transcripts data/mtg-ad
    # Ablation 3 cue set: ungated Whisper tags instead of the tagger
    python scripts/cache_audio_cues.py --cache-dir ... --no-tagger --no-clap --whisper-ungated \\
        --output-name audio_cues.whisper_tags.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio import cues_io  # noqa: E402
from src.utils.audio_features import load_current_features, resolve_audio_path  # noqa: E402
from src.utils.media import load_manifest  # noqa: E402

REPORT_NAME = "audio_cues_report.json"


def dataset_transcripts(data_dir: Path | None) -> dict[str, list[dict]]:
    """MTG-AD released ASR segments by video id (fallback when Whisper is off)."""
    if data_dir is None:
        return {}
    from src.datasets import mtg_ad

    out = {}
    for split in mtg_ad.SPLITS:
        if (data_dir / f"{split}.jsonl").is_file():
            for v in mtg_ad.load_split(data_dir, split):
                out[v.video_id] = [{"start": t.start, "end": t.end, "text": t.text} for t in v.transcript]
    return out


def build_backends(args) -> cues_io.CueBackends:
    b = cues_io.CueBackends(acoustic=not args.no_acoustic)
    if not args.no_asr:
        from src.audio.asr import faster_whisper_transcriber

        b.transcriber = faster_whisper_transcriber(args.whisper_model, args.device, args.compute_type, args.language)
        b.transcriber_name = f"faster-whisper:{args.whisper_model}:{args.compute_type}"
    if not args.no_tagger:
        from src.audio.tagging import ast_model

        b.tagger, b.tagger_labels = ast_model(args.tagger_model, None if args.device == "auto" else args.device)
        b.tagger_name = args.tagger_model
    if not args.no_clap:
        from src.audio.tagging import clap_model

        b.clap = clap_model(args.clap_model, None if args.device == "auto" else args.device)
        b.clap_name = args.clap_model
    if args.text_encoder == "lexical":
        from src.audio.speech_cues import lexical_encoder

        b.text_encoder, b.text_encoder_name = lexical_encoder(), "lexical-hash-256"
    elif args.text_encoder:
        from src.audio.speech_cues import sentence_encoder

        b.text_encoder, b.text_encoder_name = sentence_encoder(args.text_encoder), args.text_encoder
    return b


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-name", default=cues_io.CUES_NAME)
    p.add_argument("--device", default="auto")
    p.add_argument("--compute-type", default="float16", help="faster-whisper compute type (int8 on CPU)")
    p.add_argument("--whisper-model", default="large-v3")
    p.add_argument("--language", default=None)
    p.add_argument("--tagger-model", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    p.add_argument("--clap-model", default="laion/clap-htsat-unfused")
    p.add_argument("--text-encoder", default="sentence-transformers/all-MiniLM-L6-v2",
                   help='sentence-transformers model, "lexical" (weak fallback) or "" to skip semantic shift')
    p.add_argument("--no-asr", action="store_true")
    p.add_argument("--no-tagger", action="store_true")
    p.add_argument("--no-clap", action="store_true")
    p.add_argument("--no-acoustic", action="store_true")
    p.add_argument("--whisper-ungated", action="store_true", help="ablation 3 only: Whisper hears non-speech too")
    p.add_argument("--dataset-transcripts", type=Path, default=None,
                   help="MTG-AD annotation dir; used for speech cues when --no-asr")
    p.add_argument("--videos", nargs="*", default=None, help="restrict to these video ids")
    p.add_argument("--shard", default=None, help="i/n: process every n-th video starting at i (Slurm arrays)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    manifests = sorted(args.cache_dir.glob("*/manifest.json"))
    if args.videos:
        manifests = [m for m in manifests if m.parent.name in set(args.videos)]
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        if not 0 <= index < count:
            p.error("--shard must be i/n with 0 <= i < n")
        manifests = manifests[index::count]
    if not manifests:
        print(f"no */manifest.json under {args.cache_dir}", file=sys.stderr)
        return 1
    config = {"asr": {"vad_gated": not args.whisper_ungated}}
    transcripts = dataset_transcripts(args.dataset_transcripts) if args.no_asr else {}
    try:
        base = build_backends(args)
    except ImportError as exc:
        print(f"missing dependency: {exc}", file=sys.stderr)
        return 2
    report = {"computed": [], "current": [], "skipped": {}, "failed": [], "backends": base.identity()}
    for manifest_path in manifests:
        video_dir = manifest_path.parent
        vid = video_dir.name
        try:
            manifest = load_manifest(video_dir)
            wav = resolve_audio_path(manifest, video_dir)
            if wav is None:
                report["skipped"][vid] = "no_audio_stream"
                continue
            features, reason = load_current_features(video_dir, manifest)
            if features is None:
                report["skipped"][vid] = f"no current VAD features ({reason}); run cache_audio_features.py"
                continue
            backends = cues_io.CueBackends(**{**base.__dict__, "dataset_transcript": transcripts.get(vid)})
            if not args.overwrite:
                current, _ = cues_io.load_current_cues(video_dir, wav, features, backends, config, args.output_name)
                if current is not None:
                    report["current"].append(vid)
                    continue
            started = time.time()
            record = cues_io.compute_audio_cues(wav, features, backends, config)
            cues_io.write_cues(video_dir, record, args.output_name)
            report["computed"].append(vid)
            print(f"{vid}: {len(record['events'])} events, {len(record['speech_cues'])} speech cues, "
                  f"streams {record['streams']} ({time.time() - started:.0f}s)", flush=True)
        except ImportError as exc:
            print(f"missing dependency, stopping: {exc}", file=sys.stderr)
            report["aborted"] = repr(exc)
            break
        except Exception as exc:  # one bad video must not stop the batch
            report["failed"].append({"video_id": vid, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{vid}: FAILED {exc!r}", file=sys.stderr)
    report_name = REPORT_NAME if not args.shard else REPORT_NAME.replace(".json", f".shard{args.shard.replace('/', 'of')}.json")
    (args.cache_dir / report_name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"computed {len(report['computed'])}, current {len(report['current'])}, "
          f"skipped {len(report['skipped'])}, failed {len(report['failed'])}")
    if report.get("aborted"):
        return 2
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
