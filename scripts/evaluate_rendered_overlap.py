"""Measure actual 1.2x rendered narration collision against current VAD/Whisper.

Input JSONL: id, video_id, time, output (generated text), tts_audio (1x WAV),
tts_audio_sha256, tts_text_sha256 (UTF-8 text digest), tts_playback_speed=1.0. The TTS producer is
responsible for this binding. This evaluator does not synthesize narration.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.speech_overlap import accelerate_rendered_audio, evaluate_intervals
from src.utils.audio_features import load_current_features, resolve_audio_path
from src.utils.media import load_manifest, cache_is_current


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('no rendered predictions; no result can be claimed')
    ids, by_video, sources = set(), defaultdict(list), {}
    for row in rows:
        ident, video_id = row['id'], row['video_id']
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError('unique nonempty prediction IDs required')
        if not isinstance(video_id, str) or not video_id or Path(video_id).name != video_id or video_id in ('.','..'):
            raise ValueError('invalid video ID')
        ids.add(ident)
        if row.get('tts_playback_speed') != 1.0:
            raise ValueError('explicit unaccelerated TTS playback speed 1.0 required')
        if 'error' in row or not isinstance(row['output'], str) or not row['output'].strip():
            raise ValueError('input must contain successful, nonempty rendered predictions')
        if hashlib.sha256(row['output'].encode()).hexdigest() != row['tts_text_sha256']:
            raise ValueError('TTS text binding mismatch')
        if sha(row['tts_audio']) != row['tts_audio_sha256']:
            raise ValueError('TTS waveform changed')
        by_video[video_id].append(row)
    # Validate every source before creating any result artifacts.
    for video_id in by_video:
        cache = args.cache_dir/video_id
        manifest = load_manifest(cache)
        if not cache_is_current(manifest):
            raise ValueError('stale media cache')
        vad, reason = load_current_features(cache, manifest)
        if vad is None:
            raise ValueError(reason)
        speech = json.loads((cache/'speech_features.json').read_text())
        if (not speech['complete'] or speech['provenance']['audio_sha256'] != sha(resolve_audio_path(manifest, cache))
                or speech['provenance']['vad_config_hash'] != vad['config_hash']
                or speech['provenance']['speech'] != vad['speech']):
            raise ValueError('stale Whisper speech gates or waveform')
        sources[video_id] = (manifest, vad, speech, {
            name: sha(cache/name) for name in ('manifest.json','audio_features.json','speech_features.json')})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    videos = []
    for video_id, predictions in by_video.items():
        manifest, vad, speech, hashes = sources[video_id]
        narrations, artifacts = [], []
        for row in predictions:
            target = args.output_dir/(hashlib.sha256(row['id'].encode()).hexdigest()+'.wav')
            duration = accelerate_rendered_audio(row['tts_audio'], target)
            narrations.append({'id': row['id'], 'start': row['time'], 'duration': duration})
            artifacts.append({'id': row['id'], 'source_sha256': row['tts_audio_sha256'],
                              'text_sha256': row['tts_text_sha256'], 'rendered_path': str(target),
                              'rendered_sha256': sha(target), 'duration': duration})
        metrics = evaluate_intervals(narrations, vad['speech'],
            [(word['start'],word['end']) for word in speech['words']], manifest['media_info']['duration'])
        videos.append({'video_id': video_id, 'source_sha256': hashes, 'artifacts': artifacts, **metrics})
    report = {'complete': True, 'input_sha256': sha(args.input), 'playback_speed': 1.2,
              'implementation_sha256': {str(path): sha(path) for path in (Path(__file__), Path(__file__).resolve().parents[1]/'src/evaluation/speech_overlap.py')},
              'git_sha': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
              'ffmpeg': subprocess.check_output(['ffmpeg','-version'], text=True).splitlines()[0],
              'scope': 'rendered predictions provided only; generation failures and selection shortfalls require separate coverage',
              'videos': videos}
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'videos': len(videos), 'rendered_predictions': len(rows), 'output': str(args.output_dir)}))


if __name__ == '__main__':
    main()
