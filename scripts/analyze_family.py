"""Analyze a predeclared comparison family and apply Holm correction.

Family JSON: {"comparisons": [{"a": "U_P0", "b": "V_P0", "metric": "f1"}]}.
Use one family for all confirmatory comparisons; freeze it before test scoring.
Input rows follow analyze_paired.py. REFRAMED requires --unit cluster.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_paired import analyze
from src.evaluation.stats import holm_adjust


def analyze_family(rows, comparisons, unit, resamples=10000, seed=0):
    if not comparisons:
        raise ValueError('comparison family cannot be empty')
    seen, reports = set(), []
    for comparison in comparisons:
        a, b, metric = (comparison[k] for k in ('a', 'b', 'metric'))
        key = (tuple(sorted((a, b))), metric)
        if key in seen:
            raise ValueError('duplicate comparison in family')
        seen.add(key)
        reports.append(analyze(rows, a, b, metric, unit, resamples, seed))
    corrected = holm_adjust([r['wilcoxon']['pvalue'] for r in reports])
    for report, pvalue in zip(reports, corrected):
        report['wilcoxon']['holm_pvalue'] = pvalue
        report['wilcoxon']['family_size'] = len(reports)
    return {'comparisons': reports, 'correction': 'Holm',
            'confidence_intervals': 'pointwise 95%; not simultaneous family intervals'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--family', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit', choices=['example', 'cluster'], required=True)
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    payload, family = args.input.read_bytes(), args.family.read_bytes()
    result = analyze_family([json.loads(line) for line in payload.splitlines() if line.strip()],
                            json.loads(family)['comparisons'], args.unit, args.resamples, args.seed)
    result.update(input_sha256=hashlib.sha256(payload).hexdigest(),
                  family_sha256=hashlib.sha256(family).hexdigest(), command=sys.argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
