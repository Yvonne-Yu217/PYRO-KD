"""Experiment output layout, atomic writes and resumable JSONL.

    <output_root>/<experiment>/
        config.resolved.json
        selections/<selection_run_id>.jsonl      one row per video
        moments/<selection_run_id>.jsonl         one row per moment (+ .report.json)
        generations/<generation_run_id>.jsonl    append-only, resumable (src/generation/outputs.py)
        eval/*.json                              machine-readable metrics
        tables/*.csv, tables/*.md, plots/*.svg
        status.json                              stage -> run -> {done, failed, ...}
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

from src.experiments.config import get


def experiment_dir(cfg: dict) -> Path:
    path = Path(get(cfg, "paths.output_root")) / get(cfg, "experiment.name")
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    # Refuse stale reuse even for options not represented in individual run IDs.
    identity = {k: v for k, v in cfg.items() if k not in ("_source", "evaluation")}
    identity = json.loads(json.dumps(identity, default=str))
    from src.experiments.matrix import NON_RESULT_MODEL_KEYS
    for model in identity.get("models", {}).values():
        for key in NON_RESULT_MODEL_KEYS:
            model.pop(key, None)
    import hashlib
    repo = Path(__file__).resolve().parents[2]
    inputs = list((repo / "src").rglob("*.py")) + list((repo / "scripts").glob("*.py"))
    for dotted in ("dataset.splits_config", "generation.exemplars_path"):
        value = get(cfg, dotted)
        if value and Path(value).is_file():
            inputs.append(Path(value).resolve())
    identity["source_and_input_hashes"] = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(inputs))
    }
    guard = path / "config.identity.json"
    with _locked(guard):
        if guard.exists() and json.loads(guard.read_text()) != identity:
            raise ValueError("experiment configuration changed; use a new experiment.name to avoid stale outputs")
        if not guard.exists():
            if any((path / stage).exists() for stage in ("selections", "moments", "generations")):
                raise ValueError("unverified legacy outputs; use a new experiment.name")
            write_json(guard, identity)
    return path


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, default=str)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def read_jsonl(path: Path) -> list[dict]:
    """Read strictly, allowing only a recoverable unterminated tail."""
    from src.generation.jsonl_recovery import read_resume_rows
    rows, _ = read_resume_rows(path)
    return rows


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    from src.generation.jsonl_recovery import read_resume_rows, apply_tail_recovery
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path):
        _, recovery = read_resume_rows(path)
        if recovery is not None:
            apply_tail_recovery(path, recovery)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def latest_by_key(rows: Iterable[dict], key: str) -> dict[str, dict]:
    """Last row per key (later rows supersede earlier attempts)."""
    out: dict[str, dict] = {}
    for row in rows:
        if key in row:
            out[row[key]] = row
    return out


@contextlib.contextmanager
def _locked(path: Path):
    """Exclusive advisory lock beside a shared file (released if the process dies)."""
    if fcntl is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(f".{path.name}.lock"), "a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def update_status(exp_dir: Path, stage: str, run_id: str, info: dict) -> None:
    """Record one run's state. Sharded or parallel jobs share status.json, so the read-modify-write is locked."""
    path = exp_dir / "status.json"
    with _locked(path):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {}
        status.setdefault(stage, {})[run_id] = info
        write_json(path, status)
