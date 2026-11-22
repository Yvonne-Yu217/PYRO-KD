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


def test_holm_keeps_unavailable_hypotheses_in_family():
    from src.evaluation.stats import holm_adjust
    assert holm_adjust([0.04, 0.01, None, 0.03]) == pytest.approx([0.09, 0.04, None, 0.09])
    assert holm_adjust([None]) == [None]
    with pytest.raises(ValueError):
        holm_adjust([1.1])


def test_family_rejects_reverse_duplicates_and_reports_correction():
    from scripts.analyze_family import analyze_family
    data = rows([(i, 0, str(i)) for i in range(1, 6)])
    family = [{'a': 'A', 'b': 'B', 'metric': 'score'}, {'a': 'A', 'b': 'C', 'metric': 'score'}]
    result = analyze_family(data, family, 'example', 100)
    assert result['comparisons'][0]['wilcoxon']['holm_pvalue'] == pytest.approx(0.125)
    assert result['comparisons'][1]['wilcoxon']['holm_pvalue'] is None
    with pytest.raises(ValueError, match='duplicate'):
        analyze_family(data, [family[0], dict(a='B', b='A', metric='score')], 'example', 100)
