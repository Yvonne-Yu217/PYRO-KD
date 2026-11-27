import math
import pytest
from src.utils.gap_snapping import snap_to_gap


def snap(event, gaps, words=3, advance=2, lag=2):
    return snap_to_gap(event, gaps, words, max_advance=advance, max_lag=lag)


def test_event_inside_gap_stays_and_keeps_full_capacity():
    result = snap(3, [(1, 7)])
    assert result.time == 3
    assert result.word_limit == 12
    assert result.lag_seconds == 0


def test_mid_speech_event_moves_to_legal_gap_with_bounded_lag():
    result = snap(5, [(0, 2), (6, 8)])
    assert result.time == 6
    assert result.lag_seconds == 1
    assert snap(5, [(0, 2), (8, 10)]) is None
    assert snap(5, []) is None


def test_tie_prefers_after_event_and_short_gap_is_rejected():
    assert snap(5, [(3, 5), (6, 8)]).time == 6
    assert snap(5, [(5, 5.3)], words=1) is None
    assert snap(5, [(5, 7)], words=7, advance=0) is None


def test_touching_gaps_merge_and_far_advance_cannot_occur():
    result = snap(2, [(3, 4), (1, 3)], words=6)
    assert result.time == 2
    assert result.word_limit == 6
    assert snap(8, [(1, 3)]) is None


def test_numerical_boundaries_never_overclaim_capacity():
    for end in [1, 2.1, 100.1, 10000.1]:
        for words in range(1, 8):
            result = snap(end, [(0, end)], words, advance=10)
            if result:
                assert math.floor((end - result.time) * 3) >= words
                assert result.word_limit >= words
                assert result.time < end


@pytest.mark.parametrize('gaps', [[(-1, 2)], [(2, 2)], [(3, 2)], [(0, float('inf'))]])
def test_invalid_gaps_rejected(gaps):
    with pytest.raises(ValueError):
        snap(1, gaps)


def test_invalid_budget_and_bounds_rejected():
    for words in [0, -1, True, 1.5]:
        with pytest.raises(ValueError):
            snap(1, [(0, 5)], words)
    with pytest.raises(ValueError):
        snap(1, [(0, 5)], lag=-1)
