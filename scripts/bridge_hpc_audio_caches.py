"""Build PR22 cue views from verified HPC caches; no model inference or raw writes.

Development-only: preserves exact cached ASR, CLAP and acoustic tracks. Tagger
onset extraction uses PR22's declared default policy, which remains untuned.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.audio.cues_io import malformed_cues, write_cues, merged_config
from src.audio.salience import class_onsets
from src.utils.audio_features import load_current_features, resolve_audio_path
from src.utils.media import load_manifest, cache_is_current
from src.utils.speech_cues import speech_chunks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bridge(cache, speech_record, config):
    manifest = load_manifest(cache)
    if not cache_is_current(manifest):
        raise ValueError('stale media cache')
    vad, reason = load_current_features(cache, manifest)
    if vad is None:
        raise ValueError(reason)
    wavhash = sha(resolve_audio_path(manifest, cache))
    sources = {name: cache / name for name in ('speech_features.json', 'event_features.json',
               'event_features.npz', 'onset_features.json', 'audio_features.json')}
    speech = json.loads(sources['speech_features.json'].read_text())
    event = json.loads(sources['event_features.json'].read_text())
    onset = json.loads(sources['onset_features.json'].read_text())
    for data in (speech, event, onset):
        if not data['complete'] or data['provenance']['audio_sha256'] != wavhash:
            raise ValueError('incomplete or stale feature cache')
    if sha(sources['event_features.npz']) != event['arrays_sha256']:
        raise ValueError('event array hash mismatch')
    if speech_record['speech_sha256'] != sha(sources['speech_features.json']) or speech_record['vad_config_hash'] != vad['config_hash']:
        raise ValueError('stale speech cue tracks')
    with np.load(sources['event_features.npz'], allow_pickle=False) as arrays:
        tracks = {k: arrays[k] for k in arrays.files}
    times = tracks['times'].tolist()
    if times != onset['times']:
        raise ValueError('onset/event grid mismatch')
    segments = speech_chunks(speech['words'], speech['windows'])
    for segment in segments:
        segment['words'] = [{**w, 'word': w['text']} for w in speech['words']
                            if segment['start'] <= w['start'] <= w['end'] <= segment['end']]
    cues = []
    for row in speech_record['tracks']['speech']:
        for kind, key in [('semantic_shift', 'semantic_shift'), ('deictic', 'deictic')]:
            cues.append({'time': row['time'], 'strength': row[key], 'kind': kind,
                         'text': row['text'], 'timing': 'segment'})
    source_hashes = {str(p.resolve()): sha(p) for p in sources.values()}
    identity = {'sources': source_hashes, 'speech_record': speech_record, 'config': config}
    record = {
        'version': 1, 'duration': event['duration'], 'config': config,
        'config_hash': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        'audio_sha256': wavhash, 'vad_hash': vad['config_hash'],
        'bridge_sources': source_hashes,
        'backends': {'source': 'verified HPC feature reuse', 'event': event['provenance'],
                     'speech': speech['provenance']},
        'streams': {key: 'ok' for key in ('asr', 'tagger', 'clap', 'acoustic', 'semantic')},
        'asr': {'source': 'whisper-vad-gated', 'segments': segments},
        'tagger': {'model': 'pinned AST (see backends)', 'times': times, 'labels': event['labels'],
                   'posteriors': tracks['probabilities'].tolist()},
        'clap': {'model': 'pinned CLAP (see backends)', 'times': times, 'change': tracks['clap_change'].tolist()},
        'acoustic': {'times': times, 'loudness_jump': tracks['loudness_rise_db'].tolist(),
                     'onset_strength': onset['normalized']},
        'events': [e.to_dict() for e in class_onsets(times, tracks['probabilities'], event['labels'], config['nse'])],
        'speech_cues': cues,
    }
    record['streams']['diarization'] = 'missing: optional diarization not computed'
    problem = malformed_cues(record)
    if problem:
        raise ValueError(problem)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root', required=True, type=Path)
    p.add_argument('--speech-cues', required=True, type=Path)
    p.add_argument('--splits', type=Path, default=Path('configs/reframed_splits.json'))
    p.add_argument('--output-root', required=True, type=Path)
    args = p.parse_args()
    split = json.loads(args.splits.read_text())
    speech = json.loads(args.speech_cues.read_text())
    if not speech['complete'] or speech['split_sha256'] != sha(args.splits):
        raise ValueError('speech cue split mismatch')
    by_id = {r['video_id']: r for r in speech['videos']}
    records = []
    for row in split['clips']:
        ident = row['video_id']; cache = args.cache_root / ident
        record = bridge(cache, by_id[ident], merged_config(None))
        record['bridge_sources'][str(args.speech_cues.resolve())] = sha(args.speech_cues)
        record['bridge_sources'][str(args.splits.resolve())] = sha(args.splits)
        target = args.output_root / ident
        target.mkdir(parents=True, exist_ok=True)
        for source in cache.iterdir():
            if source.name.startswith('audio_cues'):
                continue
            link = target / source.name
            if not link.exists():
                link.symlink_to(source.resolve(), target_is_directory=source.is_dir())
        output = target / 'audio_cues.json'
        if output.exists():
            if json.loads(output.read_text()) != record:
                raise ValueError('changed bridge; choose new output-root')
        else:
            write_cues(target, record)
        records.append({'video_id': ident, 'events': len(record['events']), 'speech_cues': len(record['speech_cues'])})
    (args.output_root / 'bridge-report.json').write_text(json.dumps({'complete': True, 'scope': 'development only; untuned onset extraction', 'videos': records}, indent=2)+'\n')
    print(json.dumps(records))

if __name__ == '__main__':
    main()
