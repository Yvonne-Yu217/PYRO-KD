"""Score real predictions against frozen long-video question membership.

Only this evaluator reads answers. Missing, invalid, truncated and failed
predictions remain in the frozen denominator. No score is made for empty input.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.qa import score_qa


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score_long_qa(predictions, membership, answer_rows, videos):
    if not predictions:
        raise ValueError('no predictions to evaluate')
    questions = {r['question_id']: r['video_id'] for r in membership['questions']}
    if (len(questions) != len(membership['questions']) or not questions
            or membership['expected_questions'] != len(questions)):
        raise ValueError('invalid frozen question membership')
    if {row['video_id'] for row in membership['questions']} != set(videos):
        raise ValueError('question/video membership mismatch')
    expected = {}
    for row in answer_rows:
        ident = row['question_id']
        if ident not in questions:
            raise ValueError('answer outside frozen questions')
        if ident in expected or row['video_id'] != questions[ident]:
            raise ValueError('duplicate answer or wrong video')
        answer = row['answer']
        if not isinstance(answer, str) or answer not in 'ABCD' or len(answer) != 1:
            raise ValueError('expected official A-D answer label')
        expected[ident] = ord(answer)-ord('A')
    if set(expected) != set(questions):
        raise ValueError('missing frozen answer labels')
    for row in predictions:
        if row.get('id') not in questions or row.get('video_id') != questions[row['id']]:
            raise ValueError('prediction question/video mismatch')
    for key in ('selection_condition', 'audio_context'):
        if len({row.get(key) for row in predictions}) != 1:
            raise ValueError('mixed QA conditions')
    result = score_qa(predictions, expected, 4)
    grouped = defaultdict(list)
    for row in result['rows']:
        video_id = questions[row['id']]
        row.update(video_id=video_id, domain=videos[video_id]['domain'])
        grouped[('video', video_id)].append(row)
        grouped[('domain', videos[video_id]['domain'])].append(row)
    result['by_video'] = {}
    result['by_domain'] = {}
    for (kind, key), rows in sorted(grouped.items()):
        result['by_'+kind][key] = {'questions': len(rows), 'correct': sum(r['correct'] for r in rows),
                                  'accuracy': sum(r['correct'] for r in rows)/len(rows)}
    result['scope'] = 'fixed-question accuracy on the selected long-video subset; no whole-benchmark claim'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--membership', type=Path, default=Path('configs/video_mme_long_questions.json'))
    p.add_argument('--subset', type=Path, default=Path('configs/video_mme_long_subset.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous scores')
    predictions = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not predictions:
        raise ValueError('no predictions to evaluate')
    membership = json.loads(args.membership.read_text())
    subset = json.loads(args.subset.read_text())
    membership_hash, subset_hash = sha(args.membership), sha(args.subset)
    if (membership['subset_sha256'] != subset_hash
            or membership['annotations_sha256'] != subset['annotations_sha256']
            or membership['revision'] != subset['revision']
            or membership['dataset'] != subset['dataset']
            or membership['official_split'] != 'test' or subset['official_split'] != 'test'):
        raise ValueError('frozen membership provenance mismatch')
    for row in predictions:
        if (row.get('dataset') != 'video-mme' or row.get('split') != 'long-heldout'
                or not row.get('run_fingerprint')
                or row.get('source_sha256', {}).get('question_membership') != membership_hash
                or row.get('source_sha256', {}).get('subset') != subset_hash):
            raise ValueError('prediction dataset/split/source mismatch')
    if sha(args.annotations) != membership['annotations_sha256']:
        raise ValueError('annotation source changed')
    # Labels are accessed only after real, nonempty prediction and provenance checks.
    import pyarrow.parquet as pq
    ids = [r['question_id'] for r in membership['questions']]
    answer_rows = pq.read_table(args.annotations, columns=['question_id','video_id','answer'],
                                filters=[('question_id','in',ids)]).to_pylist()
    videos = {r['video_id']:r for r in subset['videos']}
    if len(videos) != len(subset['videos']):
        raise ValueError('duplicate frozen videos')
    result = score_long_qa(predictions, membership, answer_rows, videos)
    result.update(input_sha256=sha(args.input), membership_sha256=membership_hash,
                  subset_sha256=subset_hash, annotations_sha256=membership['annotations_sha256'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k:result[k] for k in ('complete','expected','correct','accuracy','status_counts')}))
    raise SystemExit(0 if result['complete'] else 1)


if __name__ == '__main__':
    main()
