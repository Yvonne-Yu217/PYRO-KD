import numpy as np
import pytest
from src.evaluation.event_detection import score_event_windows


def annotation(**extra):
    return dict(duration=30., reviewed_classes=['Music'], reviewed_intervals=[[0, 30]],
                events=[dict(label='Music', start=10., end=20.)], **extra)


def score(a=None, p=None):
    return score_event_windows([5, 15, 25], [[.9], [.8], [.1]] if p is None else p,
                               ['Music'], annotation() if a is None else a, {'Music': .5}, 10.)


def test_boundary_contact_not_event_presence_and_false_positive_counted():
    r = score()
    assert r['micro'] == dict(tp=1, fp=1, fn=0, tn=1, precision=.5, recall=1., f1=2/3)


def test_unreviewed_audio_never_becomes_negative():
    a = annotation(); a['reviewed_intervals'] = [[10, 20]]
    r = score(a)
    assert r['scored_indices'] == [1] and r['excluded_windows'] == 2
    assert r['micro']['fp'] == 0


def test_unknown_audio_excludes_windows():
    r = score(annotation(unknown_intervals=[[0, 1]]))
    assert r['scored_indices'] == [1, 2]


def test_adjacent_reviewed_intervals_merge_and_edge_padding_is_not_unknown():
    a = annotation(); a['reviewed_intervals'] = [[0, 15], [15, 30]]
    r = score_event_windows([0, 29], [[0], [0]], ['Music'], a, {'Music': .5}, 10.)
    assert r['scored_windows'] == 2 and r['reviewed_seconds'] == 30


def test_undefined_precision_and_recall_are_explicit():
    a = annotation(); a['events'] = []
    r = score(a, [[0], [0], [0]])
    assert r['micro']['precision'] is None and r['micro']['recall'] is None


@pytest.mark.parametrize('change', [
    {'reviewed_classes': []}, {'reviewed_intervals': [[-1, 30]]},
    {'reviewed_intervals': []}, {'duration': float('nan')},
    {'events': [dict(label='Unknown', start=1, end=2)]},
    {'reviewed_intervals': [[0, 10]]},
])
def test_invalid_or_unreviewed_annotations_rejected(change):
    a = annotation(); a.update(change)
    with pytest.raises(ValueError): score(a)


@pytest.mark.parametrize('p', [[[np.nan], [0], [0]], [[1.1], [0], [0]], [[0]]])
def test_invalid_probabilities_rejected(p):
    with pytest.raises(ValueError): score(p=p)


def test_cli_no_annotations_no_score(tmp_path, monkeypatch):
    import json
    import sys
    from scripts.evaluate_event_detector import main
    source = tmp_path/'human.json'; source.write_text(json.dumps({'origin':'human','videos':[]}))
    protocol = tmp_path/'protocol.json'; protocol.write_text(json.dumps({'video_ids':['a']}))
    out = tmp_path/'score.json'
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--annotations', str(source), '--protocol', str(protocol),
                                    '--cache-dir', str(tmp_path), '--output', str(out)])
    with pytest.raises(ValueError, match='coverage'): main()
    assert not out.exists()


def test_cli_records_hashes_and_rejects_changed_audio(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys
    from scripts import evaluate_event_detector as cli
    cache = tmp_path/'a'; cache.mkdir()
    audio = cache/'audio.wav'; audio.write_bytes(b'synthetic source identity fixture')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = dict(version=1, audio_sha256=digest(audio), models={'test-model':'fixed'},
                      window_seconds=10., hop_seconds=10., edge_padding='zeros')
    np.savez(cache/'event_features.npz', times=[0., 10., 20.], probabilities=[[.9], [.8], [.1]])
    meta = dict(complete=True, duration=30., windows=3, labels=['Music'], provenance=provenance,
                arrays_sha256=digest(cache/'event_features.npz'),
                config_hash=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest())
    (cache/'event_features.json').write_text(json.dumps(meta))
    row = dict(annotation(), video_id='a', annotator_id='synthetic-test-only', audio_sha256=digest(audio))
    source = tmp_path/'annotations.json'; source.write_text(json.dumps({'origin':'human','videos':[row]}))
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(dict(video_ids=['a'], split='development', models=provenance['models'],
                                        window_seconds=10., hop_seconds=10., thresholds={'Music':.5})))
    # Model/media extraction is outside this scoring test; identity hashes and
    # all on-disk annotation/cache/protocol parsing remain real.
    monkeypatch.setattr(cli, 'load_manifest', lambda _: {})
    monkeypatch.setattr(cli, 'cache_is_current', lambda _: True)
    monkeypatch.setattr(cli, 'resolve_audio_path', lambda *_: audio)
    def run(output):
        monkeypatch.setattr(sys, 'argv', ['evaluate','--annotations',str(source),'--protocol',str(protocol),
                                        '--cache-dir',str(tmp_path),'--output',str(output)])
        cli.main()
    output = tmp_path/'score.json'; run(output)
    result = json.loads(output.read_text())
    assert result['annotations_sha256'] == digest(source)
    assert result['micro']['tp'] == 1 and result['micro']['fn'] == 1
    audio.write_bytes(b'changed')
    rejected = tmp_path/'rejected.json'
    with pytest.raises(ValueError, match='provenance'): run(rejected)
    assert not rejected.exists()
