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
import hashlib
import json
import os
from pathlib import Path
from stat import S_IMODE
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


def _inspect_jsonl(path: Path, raw: bytes) -> tuple[list[dict], int]:
    """Return records and the byte offset where a malformed final fragment starts."""
    rows = []
    offset = 0
    lines = raw.split(b"\n")
    for number, line in enumerate(lines, start=1):
        line_offset = offset
        offset += len(line) + 1
        if not line.strip():
            continue
        try:
            row = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if number == len(lines):  # Complete records always have a following split entry.
                return rows, line_offset
            raise ValueError(f"{path}: malformed JSONL at line {number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}: expected a JSON object at line {number}")
        rows.append(row)
    return rows, len(raw)


def read_jsonl(path: Path) -> list[dict]:
    """Read records in order, ignoring only a malformed unterminated final line."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return []
    rows, _ = _inspect_jsonl(path, raw)
    return rows


def _write_bytes_atomic(path: Path, data: bytes, mode: int) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _prepare_jsonl_append(path: Path) -> bool:
    """Repair an unterminated file under the caller's lock, or flag a missing separator."""
    try:
        with path.open("rb") as handle:
            if handle.seek(0, os.SEEK_END) == 0:
                return False
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) == b"\n":
                return False  # The stage's resume read validates complete records.
            handle.seek(0)
            raw = handle.read()
    except FileNotFoundError:
        return False

    _, valid_end = _inspect_jsonl(path, raw)
    if valid_end == len(raw):
        return True

    mode = S_IMODE(path.stat().st_mode)
    digest = hashlib.sha256(raw).hexdigest()
    backup = path.with_name(f"{path.name}.before-recovery-{digest}.bin")
    if not backup.exists():
        _write_bytes_atomic(backup, raw, mode)
    elif backup.read_bytes() != raw:
        raise ValueError(f"{path}: recovery backup {backup} differs from the original file")
    _write_bytes_atomic(path, raw[:valid_end], mode)
    return False


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Preserve any torn tail before appending complete records under a sidecar lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path):
        needs_separator = _prepare_jsonl_append(path)
        with path.open("ab") as handle:
            if needs_separator:
                handle.write(b"\n")
            for row in rows:
                handle.write((json.dumps(row, default=str) + "\n").encode("utf-8"))
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
