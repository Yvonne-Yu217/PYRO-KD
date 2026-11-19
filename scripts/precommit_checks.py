#!/usr/bin/env python
"""Proposal §6 pre-commitment checks: record pass / fail / blocked with evidence.

    python scripts/precommit_checks.py --egoschema-dir data/egoschema \\
        --cue2narrate-dir "$CUE2NARRATE_ROOT" --cue2narrate-scores path/to/published_scores.json \\
        --longqa-jsonl "$LONGQA_JSONL" --hand-labels data/hand_labels/audio_events.jsonl \\
        --cache-dir "$PYRO_CACHE/mtg-ad" --output outputs/precommit_report.json

Missing inputs give "blocked", never "pass". The report lists the proposal's
fallback for each failed or blocked check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import precommit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--egoschema-dir", type=Path, default=None)
    p.add_argument("--cue2narrate-dir", type=Path, default=None)
    p.add_argument("--cue2narrate-scores", type=Path, default=None)
    p.add_argument("--longqa-jsonl", type=Path, default=None)
    p.add_argument("--hand-labels", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None, help="video caches holding audio_cues.json for the labeled clips")
    p.add_argument("--tolerance", type=float, default=1.0)
    p.add_argument("--output", type=Path, default=Path("outputs/precommit_report.json"))
    args = p.parse_args(argv)

    results = []
    videos = sorted((args.egoschema_dir / "videos").glob("*.mp4")) if args.egoschema_dir else []
    results.append(precommit.egoschema_audio(videos))
    results.append(precommit.cue2narrate_access(args.cue2narrate_dir, args.cue2narrate_scores))
    if args.longqa_jsonl and args.longqa_jsonl.is_file():
        from src.datasets.adapters import CanonicalAdapter

        results.append(precommit.longform_qa(CanonicalAdapter(args.longqa_jsonl).load()))
    else:
        results.append(precommit.longform_qa([]))
    pr = None
    if args.hand_labels and args.hand_labels.is_file() and args.cache_dir:
        from src.audio.cues_io import read_cues
        from src.evaluation.audio_diagnostics import detector_precision_recall, load_hand_labels

        labels, problems = load_hand_labels(args.hand_labels)
        predicted = {}
        for vid in labels:
            record, _ = read_cues(args.cache_dir / vid)
            if record is not None and str(record["streams"].get("tagger", "")).startswith("ok"):
                predicted[vid] = record["events"]
        covered = {v: e for v, e in labels.items() if v in predicted}
        if covered:
            pr = detector_precision_recall(predicted, covered, args.tolerance)
            pr["label_problems"] = problems
    results.append(precommit.detector_sanity(args.hand_labels, pr))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for r in results:
        print(f"{r['check']:20s} {r['status'].upper():8s} {json.dumps(r['evidence'])[:140]}")
        if r["status"] != "pass":
            print(f"{'':20s} fallback: {r['if_fail']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
