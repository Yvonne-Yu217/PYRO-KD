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
    path.mkdir(parents=True, exist_ok=True)
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
    """Rows in order; a truncated last line from a killed job is skipped."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
