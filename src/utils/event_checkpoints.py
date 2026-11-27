"""Atomic, content-checked CPU event batches for a single cache writer.

Final feature caches remain authoritative. A partial batch is only reusable
when its source/settings digest, indices, shape and payload hash all match.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def validate_batch(probabilities, embeddings, count, classes, dimensions):
    p, e = np.asarray(probabilities), np.asarray(embeddings)
    if (p.shape != (count, classes) or e.shape != (count, dimensions)
            or p.dtype != np.float32 or e.dtype != np.float32
            or not np.isfinite(p).all() or not np.isfinite(e).all()
            or (p < 0).any() or (p > 1).any()
            or (np.linalg.norm(e, axis=1) <= 1e-12).any()):
        raise ValueError('invalid event checkpoint arrays')
    return p, e


class EventCheckpoints:
    def __init__(self, root, config_hash, classes=527, dimensions=512):
        if (len(config_hash) != 64 or any(c not in '0123456789abcdef' for c in config_hash)
                or classes < 1 or dimensions < 1):
            raise ValueError('valid configuration digest and dimensions required')
        self.root = Path(root) / config_hash
        self.digest, self.classes, self.dimensions = config_hash, classes, dimensions

    def paths(self, start, count):
        if (isinstance(start, bool) or not isinstance(start, int) or start < 0
                or isinstance(count, bool) or not isinstance(count, int) or count < 1):
            raise ValueError('nonnegative start and positive batch size required')
        stem = self.root / f'{start:08d}-{start+count:08d}'
        return stem.with_suffix('.npz'), stem.with_suffix('.json')

    def load(self, start, count):
        payload, metadata = self.paths(start, count)
        try:
            record = json.loads(metadata.read_text())
            if (record['config_hash'] != self.digest or record['start'] != start
                    or record['count'] != count
                    or record['sha256'] != hashlib.sha256(payload.read_bytes()).hexdigest()):
                return None
            with np.load(payload, allow_pickle=False) as arrays:
                return validate_batch(arrays['probabilities'], arrays['embeddings'], count,
                                      self.classes, self.dimensions)
        except (OSError, ValueError, KeyError, EOFError):
            return None

    def save(self, start, probabilities, embeddings):
        count = len(probabilities)
        p, e = validate_batch(probabilities, embeddings, count, self.classes, self.dimensions)
        payload, metadata = self.paths(start, count)
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = payload.with_suffix('.npz.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f, probabilities=p, embeddings=e)
        temporary.replace(payload)
        record = dict(config_hash=self.digest, start=start, count=count,
                      sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
        temporary = metadata.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, sort_keys=True)+'\n')
        temporary.replace(metadata)
