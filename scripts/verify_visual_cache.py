"""Audit frozen development visual caches and local/memory ranking invariants."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.utils.visual_features import load_validated_features, load_embeddings, novelty_curve, VisualCueSelector
from src.utils.selection import k_for_duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--split-config', type=Path, default=Path('configs/reframed_splits.json'))
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    splits = json.loads(args.split_config.read_text())
    ids = [r['video_id'] for r in splits['clips']]
    records = []
    for video_id in ids:
        cache = args.cache_dir / video_id
        try:
            features, problem = load_validated_features(cache)
            if problem or features is None:
                raise ValueError(f'invalid current features: {problem}')
            embeddings = load_embeddings(cache, features)
            if embeddings is None:
                raise ValueError('embedding hash/shape/finite check failed')
            if embeddings.shape != (features['num_frames'], 512):
                raise ValueError('unexpected OpenCLIP dimensions')
            if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=.001):
                raise ValueError('embedding normalization mismatch')
            # Stored vectors are float16, whereas original novelty uses float64.
            if not np.allclose(novelty_curve(embeddings), features['novelty'], atol=.001):
                raise ValueError('local novelty disagrees with stored embeddings')
            manifest = json.loads((cache / 'manifest.json').read_text())
            duration = manifest['media_info']['duration']
            if any(t < 0 or t > duration for t in features['shots']):
                raise ValueError('shot boundary outside media')
            k = k_for_duration(duration, splits['k_rule']['rate_per_minute'])
            selector = VisualCueSelector()
            selections = {'local': selector.select(features, k),
                          'memory': selector.select_memory(features, embeddings, k)}
            for name, selected in selections.items():
                times = [c.time for c in selected]
                if len(times) > k or any(b-a < 2 for a,b in zip(times,times[1:])):
                    raise ValueError(f'{name} budget/separation violation')
                if not all(t in features['frame_times'] for t in times):
                    raise ValueError('selected timestamp missing from source frames')
            records.append({'video_id': video_id, 'verified': True, 'frames': len(embeddings),
                            'shots': len(features['shots']), 'k_requested': k,
                            'selected': {name: [c.time for c in selected] for name, selected in selections.items()},
                            'features_sha256': hashlib.sha256((cache/'visual_features.json').read_bytes()).hexdigest()})
        except Exception as exc:
            records.append({'video_id': video_id, 'verified': False, 'error': repr(exc)})
    report = {'complete': bool(records) and all(r['verified'] for r in records),
              'expected': len(ids), 'verified': sum(r['verified'] for r in records), 'videos': records,
              'note': 'Cache and ranking invariants only; no accuracy or gap-placement claim.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='videos'}))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
