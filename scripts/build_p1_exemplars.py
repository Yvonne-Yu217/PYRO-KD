#!/usr/bin/env python
"""Pick the 3 P1 exemplars from real MTG-AD professional AD lines.

Only the exemplar pool is eligible: validation videos that are not dev, never
test (configs/mtg_ad_splits.json). The pick is seeded and recorded with each
line's source video and time, so it is reproducible and auditable.

    python scripts/build_p1_exemplars.py --data-dir "$MTG_AD_ROOT"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import mtg_ad  # noqa: E402
from src.generation.priors import GUIDELINES_VERSION, select_exemplars  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=Path("data/mtg-ad"))
    p.add_argument("--splits", type=Path, default=Path("configs/mtg_ad_splits.json"))
    p.add_argument("--output", type=Path, default=Path("configs/p1_exemplars.json"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-words", type=int, default=15)
    args = p.parse_args(argv)

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    forbidden = {v["video_id"] for v in splits["dev"]["videos"]}
    test_ids = {v.video_id for v in mtg_ad.load_split(args.data_dir, "test")}
    forbidden |= test_ids
    candidates = [
        {"video_id": v.video_id, "time": a.timestamp, "text": a.text, "category": v.category}
        for v in mtg_ad.load_split(args.data_dir, "validation") if v.video_id not in forbidden
        for a in v.ad
    ]
    chosen = select_exemplars(candidates, forbidden, args.seed, args.max_words)
    record = {
        "version": f"mtg-ad-exemplar-pool-seed{args.seed}", "guidelines_version": GUIDELINES_VERSION,
        "source": "google/mind-the-gap-inline-ad validation, exemplar pool (not dev, not test)",
        "license": "CC BY 4.0 (MTG-AD annotations)", "exemplars": chosen,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for e in chosen:
        print(f"{e['video_id']} @ {e['time']:.1f}s [{e['category']}]: {e['text']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
