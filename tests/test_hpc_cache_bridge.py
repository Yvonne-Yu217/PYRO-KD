import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from src.audio.cues_io import load_cues_for_selection, STALE_CUES
from src.generation.priors import load_exemplars


def test_bridge_rejects_changed_source(tmp_path):
    source = tmp_path/'speech.json';source.write_text('original')
    record = {'audio_sha256': 'wav', 'vad_hash': 'vad',
              'bridge_sources': {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}}
    with patch('src.audio.cues_io.read_cues', return_value=(record, None)), patch('src.utils.audio_features.wav_digest', return_value='wav'):
        assert load_cues_for_selection(tmp_path, 'wav', {'config_hash': 'vad'}) == (record, None)
        source.write_text('changed')
        assert load_cues_for_selection(tmp_path, 'wav', {'config_hash': 'vad'}) == (None, STALE_CUES)
        source.unlink()
        assert load_cues_for_selection(tmp_path, 'wav', {'config_hash': 'vad'}) == (None, STALE_CUES)


def test_reframed_exemplar_bridge_retains_split_validation(tmp_path):
    import pytest
    p=tmp_path/'examples.json'
    bundle={'schema':'reframed-train-exemplars-v1','excluded_movie_ids':['test-movie'],
            'source_sha256': {'train':'hash'}, 'examples': [
                {'video_id':f'v{i}', 'movie_id':f'm{i}', 'split':'train', 'track':'us',
                 'start':0., 'end':1., 'text':'A person walks outside.'} for i in range(3)]}
    p.write_text(json.dumps(bundle))
    assert load_exemplars(p)['exemplars']==bundle['examples']
    bundle['examples'][0]['movie_id']='test-movie';p.write_text(json.dumps(bundle))
    with pytest.raises(ValueError,match='outside evaluation'):
        load_exemplars(p)


def test_bridge_records_the_speech_cue_method_that_actually_ran(tmp_path):
    import numpy as np
    from scripts import bridge_hpc_audio_caches as script
    from src.audio.cues_io import malformed_cues, merged_config
    from src.utils.speech_cues import DEICTIC_PHRASES, cue_tracks, speech_chunks

    words = [{'start': float(i), 'end': i + 0.5, 'text': t} for i, t in enumerate(('Look.', 'Stop.', 'Go.'))]
    windows = [{'start': 0.0, 'end': 4.0}]
    embeddings = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    tracks = cue_tracks(speech_chunks(words, windows), embeddings, [], min_silence=5, silence_scale=30)
    wav = tmp_path / 'a.wav'
    wav.write_bytes(b'wav')
    provenance = {'audio_sha256': script.sha(wav)}
    np.savez(tmp_path / 'event_features.npz', times=np.array([0.0, 1.0]), probabilities=np.zeros((2, 1)),
             clap_change=np.zeros(2), loudness_rise_db=np.zeros(2))
    (tmp_path / 'speech_features.json').write_text(json.dumps(
        {'complete': True, 'provenance': provenance, 'words': words, 'windows': windows}))
    (tmp_path / 'event_features.json').write_text(json.dumps(
        {'complete': True, 'provenance': provenance, 'duration': 4.0, 'labels': ['Music'],
         'arrays_sha256': script.sha(tmp_path / 'event_features.npz')}))
    (tmp_path / 'onset_features.json').write_text(json.dumps(
        {'complete': True, 'provenance': provenance, 'times': [0.0, 1.0], 'normalized': [0.0, 0.0]}))
    (tmp_path / 'audio_features.json').write_text('{}')
    per_video = {'tracks': tracks, 'speech_sha256': script.sha(tmp_path / 'speech_features.json'), 'vad_config_hash': 'vad'}
    report = {'model': 'mpnet', 'revision': 'r1', 'settings': {'max_words': 24, 'phrases': list(DEICTIC_PHRASES)}}
    with patch.object(script, 'load_manifest', return_value={}), patch.object(script, 'cache_is_current', return_value=True), \
            patch.object(script, 'load_current_features', return_value=({'config_hash': 'vad'}, None)), \
            patch.object(script, 'resolve_audio_path', return_value=wav):
        record = script.bridge(tmp_path, per_video, merged_config(None), report)

    assert malformed_cues(record) is None
    semantic = [c['strength'] for c in record['speech_cues'] if c['kind'] == 'semantic_shift']
    assert semantic == [0.0, 0.5, 0.0]  # one per chunk, halved distance, no threshold or peak step
    described = record['config']['sc']['semantic_shift']
    assert (described['window'], described['threshold'], described['local_maxima_only']) == (None, None, False)
    assert (described['model'], described['revision'], described['max_words_per_chunk']) == ('mpnet', 'r1', 24)
    assert record['config']['sc']['deictic']['phrases'] == list(DEICTIC_PHRASES)
    assert 'semantic_threshold' not in json.dumps(record['config'])  # the native settings are gone
    assert record['config']['nse'] == merged_config(None)['nse']
