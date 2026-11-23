"""Export complete REFRAMED clip temporal metrics for movie-clustered comparisons."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def export_reports(reports):
    if len(reports) < 2 or len({name for name, _ in reports}) != len(reports):
        raise ValueError('at least two uniquely named conditions required')
    reference = None
    output = []
    for condition, report in reports:
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError('nonempty condition required')
        if report['dataset'] != 'reframed' or report['scope'] != 'all_selected_videos':
            raise ValueError('complete frozen REFRAMED membership required')
        if not report.get('selection_config_sha256') or not report['metadata'].get('run_fingerprint'):
            raise ValueError('selection and actual generation provenance required')
        videos = report['videos']
        if not videos or len(videos) != report['coverage']['num_selected_videos']:
            raise ValueError('incomplete selected-video coverage')
        membership = {}
        for video in videos:
            ident, movie = video['video_id'], video['movie_id']
            if not ident or ident in membership or not movie or not video.get('reference_sha256'):
                raise ValueError('unique clips, movie IDs and reference hashes required')
            temporal = video['temporal']
            tp, fp, fn = (temporal[k] for k in ('tp','fp','fn'))
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (tp,fp,fn)):
                raise ValueError('nonnegative integer temporal counts required')
            if tp + fp != video['num_generated'] or tp + fn != video['num_ground_truth']:
                raise ValueError('temporal counts disagree with evaluated items')
            empty = tp + fp + fn == 0
            expected = dict(precision=None if empty else tp/(tp+fp) if tp+fp else 0.,
                            recall=None if empty else tp/(tp+fn) if tp+fn else 0.,
                            f1=None if empty else 2*tp/(2*tp+fp))
            for key, value in expected.items():
                actual = temporal[key]
                if (value is None and actual is not None) or (value is not None and
                    (isinstance(actual,bool) or not isinstance(actual,(int,float)) or
                     not math.isfinite(actual) or not math.isclose(value,actual,abs_tol=1e-12))):
                    raise ValueError('temporal summary differs from counts')
            membership[ident] = (movie, video['reference_sha256'], video['num_ground_truth'])
            output.append(dict(example_id=ident, cluster_id=movie, condition=condition,
                               **{'temporal_'+k:v for k,v in expected.items()},
                               num_generated=video['num_generated'],
                               num_generation_errors=video['num_generation_errors']))
        source = (report['split'], report['selection_config_sha256'],
                  {k:report['evaluation_config'][k] for k in
                   ('tolerance_seconds','reference_track','reference_merge_gap_seconds')}, membership)
        if reference is None:
            reference = source
        elif source != reference:
            raise ValueError('conditions differ in membership, movie grouping, references or temporal protocol')
    return sorted(output, key=lambda r:(r['example_id'],r['condition']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', nargs=2, action='append', required=True, metavar=('CONDITION','PATH'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix+'.provenance.json')
    if args.output.exists() or sidecar.exists():
        raise FileExistsError('preserve prior comparison')
    reports, sources = [], []
    for name, filename in args.report:
        path = Path(filename); raw = path.read_bytes()
        reports.append((name,json.loads(raw)))
        sources.append(dict(condition=name,path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest()))
    rows = export_reports(reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        for row in rows:
            f.write(json.dumps(row,allow_nan=False)+'\n')
    with sidecar.open('x') as f:
        json.dump(dict(sources=sources,output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                       analysis_unit='cluster',cluster='movie_id',
                       estimand='equal-movie mean of within-movie clip temporal metrics',
                       scope='temporal metrics only; no semantic quality or generated-count parity claim'),f,indent=2)
    print(json.dumps({'rows':len(rows),'conditions':len(reports)}))


if __name__ == '__main__':
    main()
