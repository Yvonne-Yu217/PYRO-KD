#!/usr/bin/env python
"""Run an experiment from a TOML config, one stage at a time or all of them.

Stages (each resumable; rerunning skips finished work):
    plan                 print the selection and generation runs the config defines
    validate-data        dataset adapter validation report (no caches needed)
    select               run every selector (core matrix + enabled ablations); no MLLM
    evaluate-selection   selector-only metrics, paired tests, timelines; no MLLM
    selector-only        select + evaluate-selection + report (the zero-model-cost RQ1/RQ2/RQ6 results)
    moments              generation inputs for the core selections
    generate             AD generation for the core matrix (or --runs subset)
    ablation5            timeline absent / present / audio-native generation runs
    evaluate-generation  placement with real word counts and text metrics
    qa                   multiple-choice QA for each core selection x model (timeline on and off)
    report               tables (CSV/Markdown) and SVG plots from eval/*.json

    python scripts/run_experiment.py --config configs/experiments/mtg_ad_dev.toml --stage selector-only
    python scripts/run_experiment.py --config configs/experiments/mtg_ad_dev.toml --stage generate \\
        --runs "ad-U-P0-qwen*" --set models.qwen.batch_size=8
    python scripts/run_experiment.py --config ... --stage generate --dry-run   # echo backend, no model; output goes to generations/dry_run/ (never resumed by a real run)

Selector-only stages never import a generation backend.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import matrix  # noqa: E402
from src.experiments.config import get, load_config  # noqa: E402
from src.experiments.data import clear_context_cache, load_records  # noqa: E402
from src.experiments.io import experiment_dir, write_json  # noqa: E402

STAGES = ("plan", "validate-data", "select", "evaluate-selection", "selector-only", "moments", "generate",
          "ablation5", "evaluate-generation", "qa", "report")


def _filter(runs, patterns):
    if not patterns:
        return runs
    return [r for r in runs if any(fnmatch.fnmatch(r.run_id, p) or fnmatch.fnmatch(getattr(r, "name", ""), p)
                                   for p in patterns)]


def _backend_factory(dry_run: bool, ledger_path: Path | None = None):
    def factory(spec: dict):
        from src.generation.backends import EchoBackend, build_backend

        if dry_run:
            return EchoBackend("echo", {"reply": "DRY RUN: no model was called."})
        if ledger_path is not None and spec.get("backend") in ("openai", "gemini") and not spec.get("ledger_path"):
            spec = {**spec, "ledger_path": str(ledger_path)}  # one ledger per experiment, whatever the cwd
        if spec.get("backend") in ("openai", "gemini") and spec.get("ledger_path"):
            spec = {**spec, "ledger_path": str(Path(spec["ledger_path"]).resolve())}
        return build_backend(spec)
    return factory


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="dotted override, e.g. dataset.limit=2")
    parser.add_argument("--runs", nargs="*", default=None, help="run_id or name glob patterns to restrict to")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="moments per generation run (smoke tests)")
    parser.add_argument("--dry-run", action="store_true", help="generation stages use the echo backend")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    _seed_everything(int(get(cfg, "experiment.seed", 0)))
    exp = experiment_dir(cfg)
    write_json(exp / "config.resolved.json", {k: v for k, v in cfg.items()})
    clear_context_cache()

    ledger = exp / "api_spend_ledger.json"
    selection_runs = _filter(matrix.all_selection_runs(cfg), args.runs)
    core = matrix.core_selections(cfg)
    task = get(cfg, "experiment.task", "ad")

    if args.stage == "plan":
        print("selection runs:")
        for r in matrix.all_selection_runs(cfg):
            print(f"  {r.run_id:40s} {r.ablation or 'core'}{'  [oracle upper bound]' if r.is_oracle else ''}")
        if get(cfg, "matrix.models"):
            gen = matrix.core_generation_runs(cfg, task)
            print(f"generation runs ({len(gen)}):")
            for g in gen:
                print(f"  {g.run_id}")
        return 0

    if args.stage == "validate-data":
        from src.datasets.adapters import build_adapter

        report = build_adapter(get(cfg, "dataset")).validate().to_dict()
        write_json(exp / "eval" / "dataset_validation.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1

    records = load_records(cfg)
    print(f"[experiment] {get(cfg, 'experiment.name')}: {len(records)} videos from {get(cfg, 'dataset.name')}"
          f" ({get(cfg, 'dataset.split')}/{get(cfg, 'dataset.subset', 'all')}) -> {exp}")

    if args.stage in ("select", "selector-only"):
        from src.experiments.selection_stage import run_selection

        for run in selection_runs:
            path = run_selection(cfg, run, records, exp, args.retry_failed)
            print(f"[select] {run.run_id} -> {path}")
    if args.stage in ("evaluate-selection", "selector-only"):
        from src.experiments.selection_stage import evaluate_selections, write_timelines

        result = evaluate_selections(cfg, selection_runs, records, exp, get(cfg, "evaluation.baseline", "U"))
        write_timelines(cfg, [r for r in selection_runs if r.ablation is None], records, exp)
        for name, run in result["runs"].items():
            f1 = run["summary"]["macro_f1"]
            print(f"[eval] {name:24s} n={run['summary']['n_videos']:3d} macroF1={f1['mean'] if f1['mean'] is not None else float('nan'):.3f}"
                  f"{' (oracle)' if run['oracle'] else ''} not_run={run['not_run']}")
    if args.stage == "selector-only":
        from src.experiments.report import make_report

        for path in make_report(exp):
            print(f"[report] {path}")
        return 0

    if args.stage == "moments":
        from src.experiments.generation_stage import build_moments

        for run in _filter(core + [s for s in selection_runs if s.ablation], args.runs) if args.runs else core:
            print(f"[moments] {build_moments(cfg, run, records, exp)}")
    if args.stage in ("generate", "ablation5"):
        from src.experiments.generation_stage import run_generation

        runs = matrix.core_generation_runs(cfg, "ad") if args.stage == "generate" else matrix.ablation5_runs(cfg, "ad")
        for run in _filter(runs, args.runs):
            path = run_generation(cfg, run, exp, _backend_factory(args.dry_run, ledger), records, args.limit, args.dry_run)
            print(f"[generate] {run.run_id} -> {path}")
    if args.stage == "evaluate-generation":
        from src.experiments.generation_stage import evaluate_generation, generations_path

        runs = matrix.core_generation_runs(cfg, "ad") + (matrix.ablation5_runs(cfg, "ad") if get(cfg, "ablations.timeline.selector") else [])
        for run in _filter(runs, args.runs):
            if generations_path(exp, run).exists():
                result = evaluate_generation(cfg, run, records, exp)
                print(f"[eval-gen] {run.run_id} macroF1={result['macro_f1']['mean']}")
    if args.stage == "qa":
        from src.experiments.generation_stage import evaluate_qa, run_qa

        qa_runs = []
        for run in matrix.core_generation_runs(cfg, "qa"):
            for timeline in (False, True):
                qa_runs.append(matrix.GenerationRun(run.selection, run.prior, run.model_label, run.model_json, timeline,
                                                    "qa", None, run.generation_json))
        qa_runs = [r for r in _filter(qa_runs, args.runs) if r.prior == "P0"]  # priors are an AD factor
        for run in qa_runs:
            print(f"[qa] {run.run_id} -> {run_qa(cfg, run, records, exp, _backend_factory(args.dry_run, ledger), args.dry_run)}")
        evaluate_qa(cfg, qa_runs, exp, args.dry_run)
    if args.stage == "report":
        from src.experiments.report import make_report

        for path in make_report(exp):
            print(f"[report] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
