"""Verify that incomplete transfers cannot receive a complete audit result."""
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_transferred_dataset.py'


def audit(tmp_path, *, missing=False, corrupt=False, duplicate=False, unexpected=False):
    root = tmp_path / 'data'
    (root / 'videos').mkdir(parents=True)
    row = {'id': 'clip', 'status': 'ok', 'bytes': 4}
    (root / 'download_manifest.jsonl').write_text(json.dumps(row) + '\n' + (json.dumps(row) + '\n' if duplicate else ''))
    (root / 'availability.json').write_text(json.dumps({'available': 1, 'bytes': 4, 'requested': 2, 'missing_ids': ['source_missing']}))
    if not missing:
        (root / 'videos/clip.mp4').write_bytes(b'bad' if corrupt else b'data')
    if unexpected:
        (root / 'videos/other.mp4').write_bytes(b'data')
    output = tmp_path / 'audit.json'
    result = subprocess.run([sys.executable, str(SCRIPT), '--root', str(root), '--output', str(output)], capture_output=True, text=True)
    return result, json.loads(output.read_text())


def test_complete_means_available_source_files_only(tmp_path):
    result, report = audit(tmp_path)
    assert result.returncode == 0
    assert report['complete']
    assert report['source_missing_ids'] == ['source_missing']
    assert 'content hashes' in report['not_verified']


def test_missing_transfer_not_complete(tmp_path):
    result, report = audit(tmp_path, missing=True)
    assert result.returncode == 1
    assert report['missing'] == ['clip']


def test_corrupt_size_not_complete(tmp_path):
    result, report = audit(tmp_path, corrupt=True)
    assert result.returncode == 1
    assert report['errors'][0]['reason'] == 'size_mismatch'


def test_duplicate_manifest_not_complete(tmp_path):
    result, report = audit(tmp_path, duplicate=True)
    assert result.returncode == 1
    assert report['duplicates'] == ['clip']


def test_unexpected_video_not_complete(tmp_path):
    result, report = audit(tmp_path, unexpected=True)
    assert result.returncode == 1
    assert report['unexpected'] == ['other']
