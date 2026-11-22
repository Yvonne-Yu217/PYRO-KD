import pytest
from scripts.analyze_paired import analyze


def rows(values):
    return [dict(example_id=str(i), condition=c, score=v, cluster_id=group)
            for i, (a, b, group) in enumerate(values) for c, v in [('A', a), ('B', b)]]


def test_clusters_are_equal_weight_and_pairing_precedes_aggregation():
    data = rows([(1, 0, 'film1'), (1, 0, 'film1'), (0, 1, 'film2'), (None, 1, 'film2')])
    result = analyze(data, 'A', 'B', 'score', 'cluster', 100, 4)
    assert result['bootstrap']['n'] == 2
    assert result['bootstrap']['mean_diff'] == 0
    assert result['excluded'] == [{'example_id': '3', 'reason': 'missing_metric'}]
    assert result == analyze(list(reversed(data)), 'A', 'B', 'score', 'cluster', 100, 4)


def test_known_signed_rank_probability():
    result = analyze(rows([(i, 0, str(i)) for i in range(1, 6)]), 'A', 'B', 'score', 'example', 100)
    assert result['wilcoxon']['statistic'] == 0
    assert result['wilcoxon']['pvalue'] == pytest.approx(2 / 32)


def test_missing_single_and_zero_pairs():
    data = rows([(2, 2, 'a'), (3, 3, 'b')])
    assert analyze(data, 'A', 'B', 'score', 'example', 100)['wilcoxon']['pvalue'] == 1
    result = analyze(data[:3], 'A', 'B', 'score', 'example', 100)
    assert result['wilcoxon']['pvalue'] is None
    assert result['bootstrap']['ci_low'] is None
    assert result['excluded'][0]['reason'] == 'missing_condition'


def test_invalid_and_duplicate_records_rejected():
    data = rows([(1, 0, 'a')])
    with pytest.raises(ValueError, match='duplicate'):
        analyze(data + data, 'A', 'B', 'score', 'example')
    data[0]['score'] = float('nan')
    with pytest.raises(ValueError, match='invalid metric'):
        analyze(data, 'A', 'B', 'score', 'example')
    data = rows([(1, 0, 'a')])
    data[0]['cluster_id'] = 'b'
    with pytest.raises(ValueError, match='cluster'):
        analyze(data, 'A', 'B', 'score', 'cluster')
