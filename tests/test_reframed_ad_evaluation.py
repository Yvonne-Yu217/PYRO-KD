"""Synthetic REFRAMED fixtures; no model downloads or claimed experiment scores."""
import json
from unittest.mock import Mock
import pytest
from scripts.evaluate_ad import evaluate_ad


def fixture(tmp_path):
    cues = '1\n00:00:10,000 --> 00:00:11,000\nA person\n\n2\n00:00:11,300 --> 00:00:12,000\nwalks.\n'
    rows = [dict(rottentomatoes_id='movie', imdb_id='same-movie', video_id=ident,
                 audio_description_us=cues, movie_duration=1000, video_start=100, video_duration=30)
            for ident in ['a', 'b', 'outside']]
    (tmp_path/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    predictions = tmp_path/'predictions.jsonl'
    predictions.write_text(json.dumps(dict(id='a1', video_id='a', time=10, output='A person walks.',
                                          audio_context='native'))+'\n')
    bert = Mock(return_value=(([.8], [.7], [.75]), 'synthetic-test'))
    encoder = Mock()
    encoder.encode.side_effect = lambda texts, **kwargs: [[1., 0.] for text in texts]
    return predictions, bert, encoder


def test_merges_us_cues_preserves_movie_clusters_and_missing_videos(tmp_path):
    predictions, bert, encoder = fixture(tmp_path)
    result = evaluate_ad(predictions, tmp_path, 'validation', dataset='reframed',
                         evaluation_video_ids=['a','b'], all_videos=True,
                         bert_scorer=bert, encoder=encoder)
    assert result['scope'] == 'all_selected_videos'
    assert result['coverage']['num_selected_videos'] == 2
    assert result['coverage']['split_video_ids_without_input'] == ['b']
    assert [r['movie_id'] for r in result['videos']] == ['same-movie','same-movie']
    assert [r['num_ground_truth'] for r in result['videos']] == [1,1]
    assert result['aggregate']['temporal']['tp'] == 1
    assert result['aggregate']['temporal']['fn'] == 1
    assert bert.call_args.args[1] == ['A person walks.']


def test_rejects_prediction_outside_frozen_membership(tmp_path):
    predictions, bert, encoder = fixture(tmp_path)
    with pytest.raises(ValueError, match='membership'):
        evaluate_ad(predictions, tmp_path, 'validation', dataset='reframed', evaluation_video_ids=['b'])
    bert.assert_not_called()


@pytest.mark.parametrize('other', [{'audio_context':'none'}, {}])
def test_rejects_mixed_audio_conditions(tmp_path, other):
    predictions, bert, encoder = fixture(tmp_path)
    with predictions.open('a') as handle:
        handle.write(json.dumps(dict(id='b1', video_id='b', time=10, output='Other.', **other))+'\n')
    with pytest.raises(ValueError, match='Mixed audio_context'):
        evaluate_ad(predictions, tmp_path, 'validation', dataset='reframed', bert_scorer=bert, encoder=encoder)
    bert.assert_not_called()
