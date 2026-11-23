"""Export fixed-denominator QA outcomes for paired, video-clustered analysis."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def export_reports(reports, dataset):
    if dataset not in ('egoschema', 'video-mme') or len(reports) < 2:
        raise ValueError('supported dataset and at least two conditions required')
    if len({name for name, _ in reports}) != len(reports):
        raise ValueError('unique condition names required')
    keys = (('membership_sha256','subset_sha256','annotations_sha256') if dataset == 'video-mme'
            else ('config_sha256','answers_sha256','split'))
    reference_source = reference_membership = None
    result = []
    for condition, report in reports:
        if not isinstance(condition, str) or not condition.strip():
            raise ValueError('nonempty condition names required')
        source = {key:report.get(key) for key in keys}
        if any(not isinstance(value,str) or not value for value in source.values()):
            raise ValueError('frozen source provenance required')
        if not report.get('run_fingerprint'):
            raise ValueError('actual generation run fingerprint required')
        rows = report['rows']
        if not rows or len(rows) != report['expected']:
            raise ValueError('complete frozen denominator required')
        membership = {}
        for row in rows:
            ident, video = row['id'], row['video_id']
            if (not isinstance(ident,str) or not ident or ident in membership
                    or not isinstance(video,str) or not video):
                raise ValueError('unique question IDs and nonempty video IDs required')
            answer = row['answer_index']
            options = 4 if dataset == 'video-mme' else 5
            if isinstance(answer,bool) or not isinstance(answer,int) or not 0 <= answer < options:
                raise ValueError('invalid evaluation answer index')
            if (not isinstance(row['correct'],bool)
                    or row['status'] not in ('valid','missing','invalid_answer','generation_error')
                    or (row['status'] != 'valid' and row['correct'])):
                raise ValueError('invalid success/failure outcome')
            if row['status']=='valid':
                choice=row['predicted_index']
                if (isinstance(choice,bool) or not isinstance(choice,int) or not 0 <= choice < options
                        or row['correct'] != (choice==answer)):
                    raise ValueError('correctness differs from evaluated choice')
            membership[ident] = (video, answer)
            result.append(dict(example_id=ident, cluster_id=video, condition=condition,
                               accuracy=float(row['correct']), status=row['status']))
        correct = sum(row['correct'] for row in rows)
        if report['correct'] != correct or not math.isclose(report['accuracy'],correct/len(rows),abs_tol=1e-12):
            raise ValueError('summary differs from fixed-denominator outcomes')
        if reference_source is None:
            reference_source, reference_membership = source, membership
        elif source != reference_source or membership != reference_membership:
            raise ValueError('conditions differ in frozen provenance, question membership, video mapping or labels')
    return sorted(result,key=lambda r:(r['example_id'],r['condition']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=('egoschema','video-mme'),required=True)
    p.add_argument('--report',nargs=2,action='append',metavar=('CONDITION','PATH'),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    sidecar=args.output.with_suffix(args.output.suffix+'.provenance.json')
    if args.output.exists() or sidecar.exists():
        raise FileExistsError('preserve previous comparison export')
    reports=[];sources=[]
    for name,filename in args.report:
        path=Path(filename);raw=path.read_bytes()
        reports.append((name,json.loads(raw)))
        sources.append(dict(condition=name,path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest()))
    rows=export_reports(reports,args.dataset)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    with sidecar.open('x') as f:
        json.dump(dict(dataset=args.dataset,sources=sources,rows=len(rows),
                       output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                       analysis_unit='cluster',cluster='video_id',metric='accuracy',
                       denominator='all frozen questions, including missing/invalid/failed predictions',
                       estimand='equal-video mean accuracy difference; equals question accuracy difference when each video has equal question count'),f,indent=2)
    print(json.dumps({'rows':len(rows),'conditions':len(reports),'analysis_unit':'cluster'}))


if __name__=='__main__':main()
