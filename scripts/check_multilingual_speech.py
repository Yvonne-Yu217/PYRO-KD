"""Offline multilingual Whisper compatibility on a fixed REFRAMED dev VAD slice.

This English development excerpt checks model loading, auto-language metadata,
and word timestamp mechanics; it does not validate non-English ASR accuracy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.media import load_manifest,cache_is_current
from src.utils.audio_features import load_current_features,resolve_audio_path
from src.utils.speech_features import transcribe_gated


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--model-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('use a new report path')
    import soundfile as sf
    from faster_whisper import WhisperModel
    downloaded=json.loads(args.model_report.read_text())
    if not downloaded['complete']: raise ValueError('incomplete model download')
    record=next(r for r in downloaded['models'] if r['repo_id']=='Systran/faster-whisper-small')
    cache=args.cache_dir/'CxWby8RuUwU6'
    manifest=load_manifest(cache)
    if not cache_is_current(manifest): raise ValueError('stale media cache')
    vad,reason=load_current_features(cache,manifest)
    if vad is None: raise ValueError(reason)
    wav=resolve_audio_path(manifest,cache)
    wave,sr=sf.read(wav,dtype='float32')
    if sr!=16000 or wave.ndim!=1: raise ValueError('expected mono16k audio')
    start,end=next((s,e) for s,e in vad['speech'] if e-s>=2)
    end=min(end,start+10)
    model=WhisperModel(record['snapshot'],device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
    if not model.model.is_multilingual: raise ValueError('loaded English-only weights')
    features=transcribe_gated(wave,[(start,end)],model,language=None)
    if not features['words'] or features['rejected_words'] or len(features['windows'])!=1:
        raise ValueError('expected nonempty in-gate words from the development excerpt')
    result={'complete':True,'scope':__doc__,'video_id':cache.name,'interval':[start,end],
            'audio_sha256':sha(wav),'vad_sha256':sha(cache/'audio_features.json'),
            'model':record,'model_files_sha256':{path.name:sha(path) for path in Path(record['snapshot']).iterdir() if path.is_file()},
            'features':features,'device':'cpu','compute_type':'int8','local_files_only':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps({'complete':True,'words':len(features['words']),'windows':features['windows']}))


if __name__=='__main__': main()
