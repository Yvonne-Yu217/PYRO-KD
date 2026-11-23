"""Resume must not silently mix changed input, frame, source or prompt settings."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.generation.provenance import capture_provenance, check_resume_fingerprint


@pytest.fixture
def fixture(tmp_path):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src/model.py').write_text('version = 1')
    frame = tmp_path / 'frame.jpg'
    frame.write_bytes(b'frame one')
    rows = [{'id': 'v:1', 'time': 1, 'frames': [str(frame)]}]
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps(rows[0]) + '\n')
    return tmp_path, source, rows, {'temperature': 0, 'prior': 'P0'}


def test_content_changes_prevent_resume(fixture):
    repo, source, rows, settings = fixture
    original = capture_provenance(*fixture)['run_fingerprint']
    previous = [{'run_fingerprint': original}]
    check_resume_fingerprint(previous, capture_provenance(*fixture)['run_fingerprint'])
    for path in [source, Path(rows[0]['frames'][0]), repo / 'src/model.py']:
        saved = path.read_bytes()
        path.write_bytes(saved + b' ')
        with pytest.raises(ValueError, match='fingerprint'):
            check_resume_fingerprint(previous, capture_provenance(*fixture)['run_fingerprint'])
        path.write_bytes(saved)
    settings['prior'] = 'P1'
    with pytest.raises(ValueError, match='fingerprint'):
        check_resume_fingerprint(previous, capture_provenance(*fixture)['run_fingerprint'])


def test_documentation_commits_do_not_invalidate_sources(fixture):
    with patch('src.generation.provenance.subprocess.check_output', side_effect=['abc\n', '']):
        before = capture_provenance(*fixture)
    (fixture[0] / 'README.md').write_text('Documentation change')
    with patch('src.generation.provenance.subprocess.check_output', side_effect=['def\n', ' M README.md']):
        after = capture_provenance(*fixture)
    assert before['source_commit'] != after['source_commit']
    assert before['run_fingerprint'] == after['run_fingerprint']


def test_missing_frame_and_legacy_rows_are_explicit(fixture):
    Path(fixture[2][0]['frames'][0]).unlink()
    record = capture_provenance(*fixture)
    assert next(iter(record['frame_sha256'].values())) == {'unreadable': 'FileNotFoundError'}
    with pytest.raises(ValueError, match='missing'):
        check_resume_fingerprint([{'id': 'legacy'}], record['run_fingerprint'])
    check_resume_fingerprint([], record['run_fingerprint'])


def test_empty_input_never_loads_model(tmp_path):
    from scripts import run_ad_vlm
    source = tmp_path / 'empty.jsonl'
    source.write_text('')
    with patch('sys.argv', ['run_ad_vlm', '--input', str(source), '--output', str(tmp_path / 'out.jsonl')]), patch.object(run_ad_vlm, 'load_model') as load:
        run_ad_vlm.main()
    load.assert_not_called()
