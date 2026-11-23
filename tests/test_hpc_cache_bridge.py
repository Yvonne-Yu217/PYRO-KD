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
