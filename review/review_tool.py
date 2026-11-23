#!/usr/bin/env python3
"""Prepare blinded reviewer folders and summarize independent rating exports. Standard library only."""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import shutil
import sys

TEAM = ['Yvonne', 'Daksh', 'Rishabh', 'Kiran', 'Prajanya']
CONTRASTS = {
    'fusion': ('AV_P0', 'A_cue_P0'),
    'visual': ('V_P0', 'U_P0'),
    'prompt': ('A_cue_P1', 'A_cue_P0'),
}
CRITERIA = ['accuracy', 'usefulness', 'non_redundancy', 'clarity_timing']
COLLISIONS = ['dialogue_overlap', 'narration_overlap']
ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def plan(clips, seed):
    """Two contrasts per clip; each contrast twice/movie and eight times/reviewer."""
    rng = random.Random(seed)
    movies = defaultdict(list)
    ids = set()
    for clip in clips:
        if not isinstance(clip, dict) or not isinstance(clip.get('clip_id'), str) or not clip['clip_id'] or clip['clip_id'] in ids:
            raise ValueError('Every clip needs a unique nonempty clip_id.')
        if not isinstance(clip.get('movie_id'), str) or not clip['movie_id']:
            raise ValueError('Every clip needs a verified movie_id.')
        ids.add(clip['clip_id'])
        movies[clip['movie_id']].append(clip)
    if len(movies) != 10 or any(len(v) != 3 for v in movies.values()):
        raise ValueError('The study requires exactly 10 movies with 3 clips each (30 total).')
    rows = []
    choices = [('fusion', 'visual'), ('fusion', 'prompt'), ('visual', 'prompt')]
    for movie in sorted(movies):
        group = sorted(movies[movie], key=lambda c: c['clip_id'])
        rng.shuffle(group)
        for clip, contrasts in zip(group, choices):
            for contrast in contrasts:
                rows.append({'clip_id': clip['clip_id'], 'movie_id': movie, 'contrast': contrast})
    for contrast in CONTRASTS:
        subset = [r for r in rows if r['contrast'] == contrast]
        rng.shuffle(subset)
        pairs = list(itertools.combinations(TEAM, 2)) * 2
        rng.shuffle(pairs)
        flips = [False] * 10 + [True] * 10
        rng.shuffle(flips)
        for row, pair, flip in zip(subset, pairs, flips):
            a, b = CONTRASTS[contrast]
            row.update(reviewers=list(pair), X=b if flip else a, Y=a if flip else b)
    rng.shuffle(rows)
    for n, row in enumerate(rows, 1):
        row['id'] = f'PAIR-{n:03d}'
    return rows


def prepare(input_path, output, study_id, seed, membership):
    if output.exists():
        raise ValueError('Output already exists. Use a new directory to preserve prior packages.')
    source = load(input_path)
    clips = source['clips']
    rows = plan(clips, seed)
    official = load(membership)['test']['video_ids']
    if {c['clip_id'] for c in clips} != set(official):
        raise ValueError('Input clips do not match the frozen 30-clip test membership.')
    by_id = {c['clip_id']: c for c in clips}
    assets = {}
    for clip in clips:
        paths = {'original': clip.get('original'), **clip.get('variants', {})}
        for label in ['original', *sorted({v for pair in CONTRASTS.values() for v in pair})]:
            raw = paths.get(label)
            if not isinstance(raw, str) or not raw:
                raise ValueError(f"{clip['clip_id']}: missing {label} video.")
            path = Path(raw)
            if not path.is_absolute():
                path = input_path.parent / path
            path = path.resolve()
            if not path.is_file() or path.stat().st_size == 0 or path.suffix.lower() not in ('.mp4', '.webm'):
                raise ValueError(f'{path}: expected a nonempty MP4 or WebM file.')
            assets[clip['clip_id'], label] = (path, digest(path))
    output.mkdir(parents=True)
    coordinator = output / 'coordinator-only'
    coordinator.mkdir()
    packages = {}
    for reviewer in TEAM:
        folder = output / 'reviewers' / reviewer
        folder.mkdir(parents=True)
        for name in ['index.html', 'styles.css', 'core.js', 'app.js']:
            shutil.copyfile(ROOT / 'web' / name, folder / name)
        # Real packages do not include practice footage or any method key.
        (folder / 'media').mkdir()
        items = []
        for row in rows:
            if reviewer not in row['reviewers']:
                continue
            media = {}
            for public_label, label in [('original', 'original'), ('X', row['X']), ('Y', row['Y'])]:
                path, sha = assets[row['clip_id'], label]
                filename = sha + path.suffix.lower()
                target = folder / 'media' / filename
                if not target.exists():
                    shutil.copyfile(path, target)
                media[public_label] = 'media/' + filename
            items.append({'id': row['id'], 'media': media})
        package = {'schema_version': 1, 'study_id': study_id, 'reviewer': reviewer, 'demo': False, 'items': items}
        package['package_id'] = hashlib.sha256(json.dumps(package, sort_keys=True).encode()).hexdigest()
        dump(folder / 'review-package.json', package)
        packages[reviewer] = package['package_id']
    private = {'schema_version': 1, 'study_id': study_id, 'seed': seed, 'packages': packages, 'items': rows,
               'membership_sha256': digest(membership), 'input_sha256': digest(input_path),
               'media': [{'clip_id': c, 'condition': label, 'source': str(path), 'sha256': sha}
                         for (c, label), (path, sha) in sorted(assets.items())]}
    dump(coordinator / 'study-key.json', private)
    dump(coordinator / 'allocation.json', {r: [x['id'] for x in rows if r in x['reviewers']] for r in TEAM})
    (output / 'READ_ME_FIRST.txt').write_text(
        'Send each person only their own reviewers/<Name> folder. Never serve this output root.\n'
        'Keep coordinator-only private; it reveals methods and assignments.\n'
        'Reviewers: start python3 -m http.server 8000 inside your folder, then open http://localhost:8000.\n'
        'Export ratings and send the JSON file back to the coordinator.\n', encoding='utf-8')
    return {'items': len(rows), 'ratings_expected': len(rows) * 2, 'reviewers': {r: 24 for r in TEAM}, 'output': str(output)}


def validate_rating(r):
    if r.get('preference') not in ('X', 'Y', 'Tie', 'Cannot assess'):
        raise ValueError('Invalid preference.')
    if not isinstance(r.get('evidence'), str) or not r['evidence'].strip() or r.get('independent') is not True:
        raise ValueError('Completed review needs evidence and independence confirmation.')
    if r['preference'] != 'Cannot assess':
        for v in ('X', 'Y'):
            for c in CRITERIA:
                value = r.get(v, {}).get(c)
                if type(value) is not int or not 1 <= value <= 5:
                    raise ValueError(f'Missing/invalid {v} score: {c}.')
            for c in COLLISIONS:
                if r.get(v, {}).get(c) not in ('Yes', 'No', 'Uncertain'):
                    raise ValueError(f'Missing/invalid overlap answer: {v} {c}.')


def summarize(key_path, exports, output):
    if output.exists():
        raise ValueError('Summary output exists; choose a new path.')
    key = load(key_path)
    items = {r['id']: r for r in key['items']}
    votes = defaultdict(list)
    seen = set()
    completion = Counter()
    drafts = Counter()
    for path in exports:
        data = load(path)
        reviewer = data.get('reviewer')
        if data.get('demo') is not False or data.get('schema_version') != 1 or data.get('study_id') != key['study_id'] or reviewer not in key['packages'] or data.get('package_id') != key['packages'][reviewer]:
            raise ValueError(f'{path}: wrong study/reviewer/package or practice export.')
        if not isinstance(data.get('ratings'), list):
            raise ValueError('Missing ratings array.')
        for rating in data['ratings']:
            ident = rating.get('item_id')
            if ident not in items or reviewer not in items[ident]['reviewers']:
                raise ValueError(f'{path}: unassigned item {ident}.')
            if (reviewer, ident) in seen:
                raise ValueError('Duplicate reviewer/item. Supply only one final export per reviewer.')
            seen.add((reviewer, ident))
            if type(rating.get('completed')) is not bool:
                raise ValueError('Missing completed boolean.')
            if not rating['completed']:
                drafts[reviewer] += 1
                continue
            validate_rating(rating)
            completion[reviewer] += 1
            votes[ident].append(rating)
    contrasts = {}
    for contrast, (a, b) in CONTRASTS.items():
        subset = [r for r in items.values() if r['contrast'] == contrast]
        counts = Counter()
        item_counts = Counter()
        agreement_n = agreement_d = 0
        for row in subset:
            choices = []
            for r in votes[row['id']]:
                pref = r['preference']
                choice = row[pref] if pref in ('X', 'Y') else pref
                counts[choice] += 1
                choices.append(choice)
            if len(choices) < 2:
                item_counts['awaiting_two_reviews'] += 1
            elif 'Cannot assess' in choices:
                item_counts['contains_cannot_assess'] += 1
            else:
                agreement_d += 1
                if choices[0] == choices[1]:
                    agreement_n += 1
                    item_counts['agreement_' + choices[0]] += 1
                else:
                    item_counts['disagreement'] += 1
        contrasts[contrast] = {'methods': [a,b], 'items_expected': 20, 'judgments_expected': 40,
                              'judgment_counts': {v: counts[v] for v in [a,b,'Tie','Cannot assess']},
                              'item_outcomes': dict(item_counts), 'agreement_numerator': agreement_n,
                              'agreement_denominator': agreement_d,
                              'raw_agreement': agreement_n/agreement_d if agreement_d else None}
    result = {'study_id': key['study_id'], 'completed_reviews': sum(completion.values()), 'expected_reviews': 120,
              'reviewers': {r: {'completed': completion[r], 'drafts_in_export': drafts[r], 'expected':24} for r in TEAM},
              'contrasts': contrasts, 'interpretation': 'Descriptive only. Ratings repeat within items and movies. No independence-based p-values or confidence intervals are computed. Cannot assess is excluded from agreement.'}
    # Explicit completed membership also includes exported drafts as outstanding.
    completed_pairs = set()
    for path in exports:
        d = load(path)
        completed_pairs.update((d['reviewer'], r['item_id']) for r in d['ratings'] if r['completed'])
    result['missing_completed'] = [{'reviewer': r, 'item_id': row['id']} for row in items.values()
                                   for r in row['reviewers'] if (r, row['id']) not in completed_pairs]
    output.parent.mkdir(parents=True, exist_ok=True)
    dump(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--study-id', required=True)
    p.add_argument('--seed', type=int, default=20261123)
    p.add_argument('--membership', type=Path, default=ROOT.parent/'configs/reframed_splits.json')
    s = sub.add_parser('summarize')
    s.add_argument('--key', type=Path, required=True)
    s.add_argument('--exports', type=Path, nargs='+', required=True)
    s.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare(args.input.resolve(), args.output, args.study_id, args.seed, args.membership)
        else:
            result = summarize(args.key, args.exports, args.output)
        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.exit(1, f'Error: {exc}\n')

if __name__ == '__main__':
    main()
