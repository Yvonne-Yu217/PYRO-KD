"""Provenance-checked training-only professional AD prompt examples."""
import hashlib
import json
from pathlib import Path


def load_exemplars(path: str | Path) -> tuple[list[str], str]:
    raw = Path(path).read_bytes()
    bundle = json.loads(raw)
    if bundle.get('schema') != 'reframed-train-exemplars-v1':
        raise ValueError('unsupported exemplar bundle schema')
    records = bundle.get('examples', [])
    if len(records) != 3:
        raise ValueError('exactly three exemplars required')
    movies, texts = set(), []
    excluded = set(bundle.get('excluded_movie_ids', []))
    if not excluded or not bundle.get('source_sha256'):
        raise ValueError('exemplar split provenance is missing')
    for row in records:
        if row.get('split') != 'train' or row.get('track') not in ('us', 'uk'):
            raise ValueError('exemplars must be professional REFRAMED training AD')
        movie = row.get('movie_id')
        if not movie or movie in movies or movie in excluded:
            raise ValueError('exemplar movies must be distinct and outside evaluation splits')
        text = row.get('text')
        if not isinstance(text, str) or not text.strip() or not row.get('video_id'):
            raise ValueError('exemplar text and video provenance required')
        if not isinstance(row.get('start'), (int, float)) or not isinstance(row.get('end'), (int, float)) or not 0 <= row['start'] < row['end']:
            raise ValueError('invalid exemplar source interval')
        movies.add(movie)
        texts.append(text)
    return texts, hashlib.sha256(raw).hexdigest()


def check_resume_prior(rows: list[dict], prior: str, exemplar_sha256: str | None) -> None:
    """Never reuse outputs from another prior or a different P1 example bundle."""
    for row in rows:
        if row.get('prior', 'P0') != prior:
            raise ValueError('output contains a different prior; use a separate output file')
        if row.get('exemplar_sha256') != exemplar_sha256:
            raise ValueError('output exemplar hash differs; use a separate output file')
