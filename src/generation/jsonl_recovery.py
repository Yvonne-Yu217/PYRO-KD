"""Lossless preparation of a single-writer prediction JSONL for append/resume.

Only an unterminated final line may be repaired. Interior or newline-terminated
corruption fails. The exact original bytes are preserved before atomic repair.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class TailRecovery:
    original: bytes
    recovered: bytes
    reason: str


def read_resume_rows(path):
    path=Path(path)
    if not path.exists(): return [],None
    raw=path.read_bytes();rows=[];offset=0
    lines=raw.splitlines(keepends=True)
    for index,line in enumerate(lines):
        final_unterminated=index==len(lines)-1 and not raw.endswith(b'\n')
        try:
            text=line.decode('utf-8')
            value=json.loads(text) if text.strip() else None
            if text.strip() and not isinstance(value,dict):
                raise ValueError('prediction records must be JSON objects')
        except (UnicodeDecodeError,json.JSONDecodeError) as exc:
            if not final_unterminated:
                raise ValueError(f'{path}: malformed complete/interior line {index+1}') from exc
            return rows,TailRecovery(raw,raw[:offset],'unterminated malformed final line')
        if value is not None: rows.append(value)
        offset+=len(line)
    if raw and not raw.endswith(b'\n'):
        return rows,TailRecovery(raw,raw+b'\n','valid final record missing newline')
    return rows,None


def apply_tail_recovery(path,recovery):
    path=Path(path)
    if path.read_bytes()!=recovery.original:
        raise ValueError('prediction file changed after inspection; refusing concurrent repair')
    digest=hashlib.sha256(recovery.original).hexdigest()
    backup=path.with_name(path.name+'.before-recovery-'+digest+'.bin')
    if backup.exists():
        if backup.read_bytes()!=recovery.original: raise ValueError('recovery backup mismatch')
    else:
        with backup.open('xb') as handle:
            handle.write(recovery.original);handle.flush();os.fsync(handle.fileno())
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.recover-',delete=False) as handle:
            temporary=Path(handle.name)
            handle.write(recovery.recovered);handle.flush();os.fsync(handle.fileno())
        os.chmod(temporary,path.stat().st_mode & 0o777)
        if path.read_bytes()!=recovery.original: raise ValueError('prediction file changed during repair')
        os.replace(temporary,path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)
    return {'reason':recovery.reason,'original_sha256':digest,'backup':str(backup),
            'original_bytes':len(recovery.original),'recovered_bytes':len(recovery.recovered)}
