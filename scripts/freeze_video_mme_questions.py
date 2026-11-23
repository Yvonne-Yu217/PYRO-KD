"""Freeze exact QA membership using ID/metadata columns only, never QA content."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ID_COLUMNS = ['video_id', 'videoID', 'duration', 'domain', 'question_id']


def freeze_questions(rows, subset):
    videos = {row['video_id']: row for row in subset['videos']}
    if not videos or len(videos) != len(subset['videos']):
        raise ValueError('unique nonempty frozen video membership required')
    selected, seen = [], set()
    for row in rows:
        if set(row) != set(ID_COLUMNS):
            raise ValueError('only ID/metadata columns may enter membership freeze')
        if row['video_id'] not in videos:
            continue
        source = videos[row['video_id']]
        if row['duration'] != 'long' or any(row[k] != source[k] for k in ('videoID', 'duration', 'domain')):
            raise ValueError('video metadata differs from frozen subset')
        ident = row['question_id']
        if not isinstance(ident, str) or not ident.strip() or ident in seen:
            raise ValueError('unique nonempty question IDs required')
        seen.add(ident)
        selected.append({'question_id': ident, 'video_id': row['video_id']})
    counts = Counter(row['video_id'] for row in selected)
    if set(counts) != set(videos):
        raise ValueError('missing questions for a frozen video')
    return {'questions': sorted(selected, key=lambda r: (r['video_id'], r['question_id'])),
            'video_question_counts': dict(sorted(counts.items())), 'expected_questions': len(selected)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--subset', type=Path, default=Path('configs/video_mme_long_subset.json'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('question membership already frozen')
    subset = json.loads(args.subset.read_text())
    digest = hashlib.sha256(args.annotations.read_bytes()).hexdigest()
    if digest != subset['annotations_sha256']:
        raise ValueError('annotation source differs from frozen video selection')
    import pyarrow.parquet as pq
    rows = pq.read_table(args.annotations, columns=ID_COLUMNS).to_pylist()
    result = freeze_questions(rows, subset)
    result.update(dataset=subset['dataset'], revision=subset['revision'],
                  official_split=subset['official_split'],
                  subset_sha256=hashlib.sha256(args.subset.read_bytes()).hexdigest(),
                  annotations_sha256=digest, columns_read=ID_COLUMNS,
                  scope='held-out question membership only; no QA content or answer labels read; selection/inference protocol still requires freezing')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'videos': len(result['video_question_counts']),
                      'questions': result['expected_questions'], 'columns_read': ID_COLUMNS}))


if __name__ == '__main__':
    main()
