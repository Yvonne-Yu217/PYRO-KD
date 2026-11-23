"""Record exact runtime sources, input JSONL and frame content for generation."""
import hashlib
import json
from pathlib import Path
import subprocess


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def capture_provenance(repo: Path,input_path: Path,rows,settings):
    sources={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in ('src','scripts','configs') for p in sorted((repo/folder).rglob('*'))
             if p.is_file() and p.suffix in ('.py','.json')}
    frames={}
    for name in sorted({f for row in rows for f in row['frames']}):
        try: frames[name]=hashlib.sha256(Path(name).read_bytes()).hexdigest()
        except OSError as exc: frames[name]={'unreadable':type(exc).__name__}
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True,stderr=subprocess.DEVNULL).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True,stderr=subprocess.DEVNULL).strip())
    except (OSError,subprocess.CalledProcessError):
        commit,dirty=None,None
    record={'source_commit':commit,'source_dirty':dirty,'source_file_sha256':sources,
            'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
            'frame_sha256':frames,'settings':settings}
    # Commit/dirty flags can change for documentation-only commits; source bytes
    # determine generation compatibility. Limit/batch/token cap are attempt data.
    record['run_fingerprint']=digest({k:v for k,v in record.items() if k not in ('source_commit','source_dirty')})
    return record


def check_resume_fingerprint(previous,fingerprint):
    if any(row.get('run_fingerprint')!=fingerprint for row in previous):
        raise ValueError('output input/source/frame/settings fingerprint differs or is missing; use a separate output file')
