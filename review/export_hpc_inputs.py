#!/usr/bin/env python3
"""Bind the five frozen local AD conditions and their existing WAVs into review render inputs."""
import argparse
from pathlib import Path
from review_tool import CONTRASTS, ROOT, digest, dump, load


def export(cache_root, tts_root, output, report):
    if output.exists():
        raise ValueError('Output exists; choose a new path.')
    evidence = load(report)
    ids = load(ROOT.parent/'configs/reframed_splits.json')['test']['video_ids']
    clusters = {}
    metrics = ROOT.parent/'docs/results/reframed-selector-ablations/paired-metrics.jsonl'
    import json
    for line in metrics.read_text().splitlines():
        row = json.loads(line)
        if row['example_id'] in clusters and clusters[row['example_id']] != row['cluster_id']:
            raise ValueError('Conflicting movie assignment.')
        clusters[row['example_id']] = row['cluster_id']
    def bound(path, suffix):
        matches = [sha for source, sha in evidence['sources'].items() if source.endswith(suffix)]
        if len(matches) != 1 or not path.is_file() or digest(path) != matches[0]:
            raise ValueError(f'Artifact missing or differs from frozen receipt: {path}')
    clips = {}
    for ident in ids:
        manifest = cache_root/ident/'manifest.json'
        bound(manifest, f'/{ident}/manifest.json')
        source = Path(load(manifest)['source'])
        if not source.is_file():
            raise ValueError(f'Source media missing: {source}')
        clips[ident] = {'clip_id':ident,'movie_id':clusters[ident],'original':str(source.resolve()),'narrations':{}}
    labels = {v for pair in CONTRASTS.values() for v in pair}
    for label in sorted(labels):
        selector, prior = label.rsplit('_',1)
        runs = [r for r in evidence['results'] if r['selector']==selector and r['prior']==prior and r['model']=='qwen' and r['timeline'] is True]
        if len(runs) != 1:
            raise ValueError(f'Expected one frozen condition: {label}')
        run = runs[0]
        rendered = tts_root/run['run_id']/'rendered/rendered.jsonl'
        bound(rendered, f"/{run['run_id']}/rendered/rendered.jsonl")
        rows = [json.loads(line) for line in rendered.read_text().splitlines() if line.strip()]
        if len(rows) != run['coverage']['valid'] or len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Rendered count or ID coverage differs from frozen result.')
        for c in clips.values():
            c['narrations'][label] = []
        for row in rows:
            if row['video_id'] not in clips:
                raise ValueError('Unexpected video in rendered output.')
            wav = Path(row['tts_audio'])
            # A relocated TTS root may retain the original absolute filename in JSONL.
            if not wav.is_file():
                wav = rendered.parent/wav.name
            if not wav.is_file() or digest(wav) != row['tts_audio_sha256']:
                raise ValueError(f'WAV missing or changed: {wav}')
            physical_speed = row.get('tts_playback_speed')
            if type(physical_speed) not in (int,float) or physical_speed <= 0:
                raise ValueError('Explicit original WAV playback speed required.')
            clips[row['video_id']]['narrations'][label].append({
                'time':row['time'],'wav':str(wav.resolve()),'playback_speed':1.2/physical_speed,
                'prediction_id':row['id'],'wav_sha256':row['tts_audio_sha256']})
    output.parent.mkdir(parents=True,exist_ok=True)
    dump(output,{'clips':list(clips.values()),'receipt':{'results_report_sha256':digest(report),
         'membership_sha256':digest(ROOT.parent/'configs/reframed_splits.json'),
         'movie_mapping_sha256':digest(metrics),'target_playback_speed':1.2,
         'note':'All frozen valid renders included. Failed/unprepared descriptions remain absent; no replacement narration.'}})
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root',type=Path,default=Path('/scratch/yy5919/pyro-kd-derived/cache/reframed-test-whisper-gated-v1'))
    p.add_argument('--tts-root',type=Path,default=Path('/scratch/yy5919/pyro-kd-derived/tts/formal-local-v1'))
    p.add_argument('--report',type=Path,default=ROOT.parent/'docs/results/reframed-formal-gap-utilization/measured-gap-utilization.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        print(export(a.cache_root,a.tts_root,a.output,a.report))
    except (ValueError,KeyError,OSError,TypeError) as exc:
        p.exit(1,f'Error: {exc}\n')
