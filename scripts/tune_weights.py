#!/usr/bin/env python
"""Tune two-stage selector weights on the dev split with cross-validation (selector-only, no MLLM).

    python scripts/tune_weights.py --config configs/experiments/mtg_ad_dev.toml --base AV
    python scripts/tune_weights.py --config ... --grid '{"weights.nse": [-1, 0, 1], "memory_blend": [0.5, 1.0]}'

Writes <experiment>/eval/tuning_<base>.json. Refuses to run on test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.config import load_config  # noqa: E402
from src.experiments.data import load_records  # noqa: E402
from src.experiments.io import experiment_dir, write_json  # noqa: E402
from src.experiments.tuning import tune  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--base", default="AV")
    p.add_argument("--grid", default=None, help="JSON {dotted option: [values]}")
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    result = tune(cfg, load_records(cfg), args.base, json.loads(args.grid) if args.grid else None, args.folds)
    path = write_json(experiment_dir(cfg, writes_outputs=False) / "eval" / f"tuning_{args.base}.json", result)
    print(f"best {result['best']} macroF1={result['best_f1']}; CV held-out mean {result['cv_held_out_mean_f1']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
