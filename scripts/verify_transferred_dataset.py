"""Read-only coverage/size/optional stream audit of transferred evaluation data.

Supports the supplied EgoSchema/REFRAMED download_manifest.jsonl schema.
A complete report requires every successful manifest ID and source snapshot
counts/bytes to agree. Partial and unexpected files never satisfy coverage.
"""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--probe', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    if args.output.resolve().is_relative_to(root):
        p.error('write audit output outside the source dataset directory')
    availability = json.loads((root / 'availability.json').read_text())
    manifest = [json.loads(line) for line in (root / 'download_manifest.jsonl').read_text().splitlines() if line.strip()]
    expected, duplicates = {}, []
    for row in manifest:
        if row['status'] != 'ok':
            continue
        key = row['id']
        if not isinstance(key, str) or Path(key).name != key or key in ('', '.', '..'):
            raise ValueError('invalid video ID')
        if key in expected:
            duplicates.append(key)
        expected[key] = row
    actual = {f.stem: f for f in (root / 'videos').glob('*.mp4') if not f.name.endswith('.partial.mp4')}
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    errors, verified = [], []
    for key in sorted(expected.keys() & actual.keys()):
        f, row = actual[key], expected[key]
        size = f.stat().st_size
        if size != row['bytes']:
            errors.append({'id': key, 'reason': 'size_mismatch', 'actual': size, 'expected': row['bytes']})
            continue
        record = {'id': key, 'bytes': size}
        if args.probe:
            try:
                result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type:format=duration', '-of', 'json', str(f)], capture_output=True, text=True, timeout=30, check=True)
                probe = json.loads(result.stdout)
                types = sorted(s['codec_type'] for s in probe.get('streams', []))
                expected_types = sorted(s['codec_type'] for s in row['streams'])
                duration = float(probe['format']['duration'])
                if types != expected_types or abs(duration - row['duration']) > 0.1:
                    raise ValueError('streams/duration differ from transferred manifest')
                record.update(streams=types, duration=duration)
            except (subprocess.SubprocessError, ValueError, KeyError) as exc:
                errors.append({'id': key, 'reason': 'probe_failure', 'error': str(exc)})
                continue
        verified.append(record)
    snapshot_errors = []
    if len(expected) != availability['available']:
        snapshot_errors.append('manifest count != availability count')
    if sum(r['bytes'] for r in expected.values()) != availability['bytes']:
        snapshot_errors.append('manifest bytes != availability bytes')
    if len(expected) + len(availability['missing_ids']) != availability['requested']:
        snapshot_errors.append('available + source missing != requested')
    if set(expected) & set(availability['missing_ids']):
        snapshot_errors.append('source missing IDs overlap available manifest')
    report = {'root': str(root), 'complete': not (missing or unexpected or errors or duplicates or snapshot_errors),
              'verification': 'size and ffprobe streams/duration' if args.probe else 'size only',
              'not_verified': ['full decode', 'content hashes'], 'expected': len(expected), 'verified': len(verified),
              'missing': missing, 'unexpected': unexpected, 'duplicates': duplicates,
              'source_missing_ids': availability['missing_ids'], 'errors': errors,
              'snapshot_errors': snapshot_errors, 'videos': verified}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('complete', 'expected', 'verified', 'verification')}))
    raise SystemExit(0 if report['complete'] else 1)


if __name__ == '__main__':
    main()
