"""Evaluate cached tagger windows against supplied human-reviewed annotations."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.evaluation.event_detection import score_event_windows, metrics
from src.utils.audio_features import resolve_audio_path
from src.utils.media import load_manifest, cache_is_current
from src.utils.event_features import EVENT_VERSION


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous evaluation')
    annotation = json.loads(args.annotations.read_text())
    protocol = json.loads(args.protocol.read_text())
    rows = annotation['videos']
    ids = protocol['video_ids']
    if (annotation.get('origin') != 'human' or not rows or not ids
            or len(set(ids)) != len(ids) or len(rows) != len(ids)
            or {r['video_id'] for r in rows} != set(ids)):
        raise ValueError('complete human annotation coverage of explicit protocol IDs required')
    if protocol.get('split') != 'development':
        raise ValueError('detector calibration is development-only')
    records = []
    for row in rows:
        ident = row['video_id']
        if not isinstance(ident, str) or Path(ident).name != ident or ident in ('.', '..'):
            raise ValueError('unsafe video ID')
        if not isinstance(row.get('annotator_id'), str) or not row['annotator_id'].strip():
            raise ValueError('human annotator ID required')
        cache = args.cache_dir / ident
        meta_path, arrays_path = cache/'event_features.json', cache/'event_features.npz'
        meta = json.loads(meta_path.read_text())
        provenance = meta['provenance']
        manifest = load_manifest(cache)
        audio = resolve_audio_path(manifest, cache)
        if (not meta.get('complete') or sha(arrays_path) != meta['arrays_sha256']
                or not cache_is_current(manifest) or sha(audio) != row['audio_sha256']
                or provenance['version'] != EVENT_VERSION
                or meta['config_hash'] != hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
                or row['audio_sha256'] != provenance['audio_sha256']
                or meta['duration'] != row['duration']
                or provenance['models'] != protocol['models']
                or provenance['window_seconds'] != protocol['window_seconds']
                or provenance['hop_seconds'] != protocol['hop_seconds']
                or provenance['edge_padding'] != 'zeros'):
            raise ValueError('annotation/cache/protocol provenance mismatch')
        with np.load(arrays_path, allow_pickle=False) as arrays:
            hop = protocol['hop_seconds']
            if not isinstance(hop, (int, float)) or not np.isfinite(hop) or hop <= 0:
                raise ValueError('positive finite hop required')
            expected_times = np.arange(int(np.ceil(row['duration']/hop))) * hop
            if (arrays['times'].shape != expected_times.shape
                    or not np.allclose(arrays['times'], expected_times, atol=1e-8, rtol=0)
                    or meta['windows'] != len(expected_times)):
                raise ValueError('incomplete tagger time grid')
            result = score_event_windows(arrays['times'], arrays['probabilities'], meta['labels'],
                                         row, protocol['thresholds'], protocol['window_seconds'])
        result.update(video_id=ident, annotator_id=row['annotator_id'],
                      metadata_sha256=sha(meta_path), arrays_sha256=sha(arrays_path))
        records.append(result)
    per_class = {label: metrics(*(sum(r['per_class'][label][key] for r in records)
                                 for key in ('tp', 'fp', 'fn', 'tn')))
                 for label in protocol['thresholds']}
    report = dict(complete=True, videos=records, per_class=per_class,
                  micro=metrics(*(sum(r[key] for r in per_class.values()) for key in ('tp', 'fp', 'fn', 'tn'))),
                  annotations_sha256=sha(args.annotations), protocol_sha256=sha(args.protocol),
                  scope='supplied human annotation coverage; no authentication of annotator identity; window-level presence, not event onset accuracy')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({'videos': len(records), 'micro': report['micro']}))


if __name__ == '__main__':
    main()
