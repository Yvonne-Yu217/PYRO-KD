#!/usr/bin/env python
"""Freeze three professional REFRAMED training AD examples, without test text selection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.reframed import load_split, merge_cues
from src.generation.exemplars import load_exemplars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root, output = Path(args.data_dir), Path(args.output)
    if output.exists():
        raise FileExistsError(f'Exemplar bundle is frozen; refusing overwrite: {output}')
    # Held-out tracks are parsed by the loader but only movie identifiers are used.
    excluded = {c.movie_id for split in ('validation', 'test') for c in load_split(root, split)}
    examples, used = [], set()
    for clip in sorted(load_split(root, 'train'), key=lambda c: c.video_id):
        if clip.movie_id in excluded or clip.movie_id in used:
            continue
        for cue in merge_cues(clip.ad_us):
            text = ' '.join(cue.text.split())
            if 6 <= len(text.split()) <= 25 and text.endswith(('.', '!', '?')) and 0 <= cue.start < cue.end:
                examples.append({'split':'train', 'track':'us', 'movie_id':clip.movie_id,
                                 'video_id':clip.video_id, 'start':cue.start, 'end':cue.end, 'text':text})
                used.add(clip.movie_id)
                break
        if len(examples) == 3:
            break
    if len(examples) != 3:
        raise ValueError('fewer than three eligible training movies')
    bundle = {'schema':'reframed-train-exemplars-v1',
              'source':'https://huggingface.co/datasets/igorsterner/reframed',
              'selection':'video_id ascending; first merged US cue with 6-25 whitespace words and terminal punctuation; one per movie; exclude all validation/test movies',
              'source_sha256':{s:hashlib.sha256((root/f'{s}.jsonl').read_bytes()).hexdigest() for s in ('train','validation','test')},
              'excluded_movie_ids':sorted(excluded), 'examples':examples}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents concurrent preparation from overwriting the frozen choice.
    with output.open('x') as handle:
        handle.write(json.dumps(bundle, indent=2)+'\n')
    _, digest = load_exemplars(output)
    print(json.dumps({'output':str(output),'sha256':digest,'examples':examples},indent=2))


if __name__ == '__main__':
    main()
