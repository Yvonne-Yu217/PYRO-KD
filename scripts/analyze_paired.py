"""Paired inference on JSONL metrics with explicit coverage and cluster means.

Rows: {"example_id": str, "condition": str, "metric": finite float or null,
       "cluster_id": str}. For REFRAMED cluster_id must be the movie ID.
Use --unit example for independent EgoSchema videos, --unit cluster for movies.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.stats import paired_bootstrap_diff


def analyze(rows, condition_a, condition_b, metric, unit, resamples=10000, seed=0):
    if condition_a == condition_b:
        raise ValueError('conditions must differ')
    if unit not in ('example', 'cluster') or resamples < 1:
        raise ValueError('valid unit and positive resamples required')
    indexed = {condition_a: {}, condition_b: {}}
    for row in rows:
        condition = row['condition']
        if condition not in indexed:
            continue
        key = row['example_id']
        if not isinstance(key, str) or not key:
            raise ValueError('example_id must be a nonempty string')
        if key in indexed[condition]:
            raise ValueError(f'duplicate example {condition}/{key}')
        value = row[metric]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f'invalid metric for {key}')
        indexed[condition][key] = row
    a, b = indexed[condition_a], indexed[condition_b]
    included, excluded, groups = [], [], {}
    for key in sorted(a.keys() | b.keys()):
        if key not in a or key not in b:
            excluded.append({'example_id': key, 'reason': 'missing_condition'})
            continue
        if a[key][metric] is None or b[key][metric] is None:
            excluded.append({'example_id': key, 'reason': 'missing_metric'})
            continue
        cluster = key
        if unit == 'cluster':
            cluster = a[key]['cluster_id']
            if not isinstance(cluster, str) or not cluster or cluster != b[key]['cluster_id']:
                raise ValueError(f'inconsistent or missing cluster for {key}')
        groups.setdefault(cluster, []).append((a[key][metric], b[key][metric]))
        included.append(key)
    units = []
    for key, values in sorted(groups.items()):
        units.append({'unit_id': key, 'n_examples': len(values),
                      'a': math.fsum(v[0] for v in values) / len(values),
                      'b': math.fsum(v[1] for v in values) / len(values)})
    result = {'condition_a': condition_a, 'condition_b': condition_b, 'metric': metric,
              'unit': unit, 'estimand': 'equal-weight mean across analysis units',
              'included_example_ids': included, 'excluded': excluded, 'units': units,
              'seed': seed, 'resamples': resamples, 'confidence_level': 0.95}
    result['bootstrap'] = paired_bootstrap_diff([u['a'] for u in units], [u['b'] for u in units], resamples, seed)
    # Round subtraction noise only for signed ranks; retain full precision for CI.
    diffs = [round(u['a'] - u['b'], 12) for u in units]
    result['wilcoxon'] = {'alternative': 'two-sided', 'zero_method': 'wilcox',
                          'difference_decimals': 12, 'pvalue_adjustment': 'none',
                          'statistic': None, 'pvalue': None, 'status': 'insufficient_units'}
    if len(diffs) >= 2:
        if not any(diffs):
            result['wilcoxon'].update(statistic=0.0, pvalue=1.0, status='all_zero_differences')
        else:
            import scipy
            from scipy.stats import PermutationMethod, wilcoxon
            test = wilcoxon(diffs, method=PermutationMethod(n_resamples=resamples, random_state=seed))
            result['wilcoxon'].update(statistic=float(test.statistic), pvalue=float(test.pvalue),
                                      status='ok', method='permutation', scipy_version=scipy.__version__)
    if len(units) < 2:
        result['bootstrap']['ci_low'] = result['bootstrap']['ci_high'] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--a', required=True)
    parser.add_argument('--b', required=True)
    parser.add_argument('--metric', required=True)
    parser.add_argument('--unit', choices=['example', 'cluster'], required=True)
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    result = analyze([json.loads(line) for line in payload.splitlines() if line.strip()],
                     args.a, args.b, args.metric, args.unit, args.resamples, args.seed)
    result['input_sha256'] = hashlib.sha256(payload).hexdigest()
    result['command'] = sys.argv
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
