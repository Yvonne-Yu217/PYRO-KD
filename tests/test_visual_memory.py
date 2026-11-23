"""Running-memory selection should avoid repeatedly rewarding a recurring scene."""
import numpy as np
import pytest
from src.utils.visual_features import VisualCueSelector


def fixture():
    return ({'frame_times': [0, 2, 4, 6], 'novelty': [0, 1, 1, .5], 'shots': []},
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]))


def test_recurring_scene_is_scored_against_selected_memory():
    features, vectors = fixture()
    selector = VisualCueSelector()
    assert [c.time for c in selector.select(features, 3)] == [2, 4, 6]
    chosen = selector.select_memory(features, vectors, 3)
    assert [c.time for c in chosen] == [0, 2, 6]
    assert chosen[0].novelty == .5
    # Scaling vectors must not change cosine memory or selected times.
    assert selector.select_memory(features, vectors * np.array([[2], [3], [4], [5]]), 3) == chosen


def test_separation_and_shortfall_are_retained():
    features, vectors = fixture()
    selected = VisualCueSelector().select_memory(features, vectors, 10, min_separation=3)
    assert [c.time for c in selected] == [2, 6]


def test_shot_bonus_and_deterministic_ties():
    features, vectors = fixture()
    features['shots'] = [4]
    selected = VisualCueSelector(shot_weight=2).select_memory(features, vectors, 1)
    assert selected[0].time == 4
    assert selected[0].shot_boundary


@pytest.mark.parametrize('vectors', [None, [[0, 0]] * 4, [[float('nan'), 1]] * 4, [[1, 2]]])
def test_invalid_embeddings_fail_explicitly(vectors):
    features, _ = fixture()
    with pytest.raises(ValueError, match='embeddings'):
        VisualCueSelector().select_memory(features, vectors, 2)


@pytest.mark.parametrize('k,separation', [(0, 2), (True, 2), (2, -1), (2, float('nan'))])
def test_invalid_budgets_and_spacing(k, separation):
    features, vectors = fixture()
    with pytest.raises(ValueError):
        VisualCueSelector().select_memory(features, vectors, k, separation)
