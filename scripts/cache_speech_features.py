#!/usr/bin/env python
"""Cache English Whisper words from current Silero VAD intervals, offline on CPU."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.audio_features import load_current_features, resolve_audio_path
from src.utils.media import load_manifest
from src.utils.speech_features import SPEECH_FEATURES_VERSION, transcribe_gated


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--model-report',type=Path,required=True)
    p.add_argument('--video-id',action='append',help='restrict to these prepared video IDs')
    args=p.parse_args()
    import soundfile as sf
    from faster_whisper import WhisperModel
    record=next(r for r in json.loads(args.model_report.read_text())['models'] if r['repo_id']=='Systran/faster-whisper-small.en')
    settings={'version':SPEECH_FEATURES_VERSION,'model':record['repo_id'],'revision':record['revision'],
              'compute_type':'int8','device':'cpu','language':'en','max_seconds':28.0,
              'beam_size':5,'temperature':0.0,'condition_on_previous_text':False,
              'faster_whisper':importlib.metadata.version('faster-whisper'),
              'ctranslate2':importlib.metadata.version('ctranslate2')}
    manifests=sorted(args.cache_dir.glob('*/manifest.json'))
    if args.video_id:
        manifests=[p for p in manifests if p.parent.name in args.video_id]
        if {p.parent.name for p in manifests} != set(args.video_id):
            raise ValueError('requested video cache missing')
    if not manifests:
        raise ValueError('no prepared video caches')
    model=None
    report={'settings':settings,'results':[],'complete':False}
    for path in manifests:
        cache=path.parent
        try:
            manifest=load_manifest(cache)
            vad,reason=load_current_features(cache,manifest)
            if vad is None:
                raise ValueError(f'current VAD required: {reason}')
            audio=resolve_audio_path(manifest,cache)
            raw=audio.read_bytes()
            provenance={**settings,'audio_sha256':hashlib.sha256(raw).hexdigest(),
                        'vad_config_hash':vad['config_hash'],'speech':vad['speech']}
            digest=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
            output=cache/'speech_features.json'
            previous=json.loads(output.read_text()) if output.exists() else {}
            if previous.get('complete') and previous.get('config_hash')==digest:
                result={'video_id':cache.name,'status':'current','words':len(previous['words'])}
            else:
                wave,sr=sf.read(io.BytesIO(raw),dtype='float32')
                if sr!=16000 or wave.ndim!=1:
                    raise ValueError('requires prepared 16kHz mono audio')
                if model is None:
                    model=WhisperModel(record['snapshot'],device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
                features=transcribe_gated(wave,vad['speech'],model)
                current,reason=load_current_features(cache,manifest)
                if current is None or current['config_hash']!=vad['config_hash'] or hashlib.sha256(audio.read_bytes()).hexdigest()!=provenance['audio_sha256']:
                    raise ValueError('audio/VAD changed during transcription')
                payload={**features,'provenance':provenance,'config_hash':digest,'complete':True}
                temporary=output.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(payload,indent=2)+'\n')
                temporary.replace(output)
                result={'video_id':cache.name,'status':'computed','words':len(features['words']),
                        'rejected_words':len(features['rejected_words']),'windows':len(features['windows'])}
        except Exception as exc:
            result={'video_id':cache.name,'status':'failed','error':repr(exc)}
        print(json.dumps(result),flush=True)
        report['results'].append(result)
        (args.cache_dir/'speech_features_report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=all(r['status'] in ('computed','current') for r in report['results'])
    (args.cache_dir/'speech_features_report.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['complete'] else 1)


if __name__=='__main__':
    main()
