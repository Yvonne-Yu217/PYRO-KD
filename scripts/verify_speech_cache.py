"""Audit cached ASR provenance, VAD sample coverage and word/language metadata.

Does not read QA questions/answers or judge transcription accuracy. Reported
words stay in their detected language; no transcript text is printed.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import wave
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.audio_features import load_current_features,resolve_audio_path,wav_digest
from src.utils.media import load_manifest,cache_is_current
from src.utils.speech_features import speech_windows,SPEECH_FEATURES_VERSION


def check_words(features, speech, samples, language):
    expected=list(speech_windows(speech,samples))
    windows=features['windows']
    if len(windows)!=len(expected): raise ValueError('ASR window coverage mismatch')
    offset=0
    languages=Counter()
    for window,(left,right) in zip(windows,expected):
        start,end=left/16000,right/16000
        if (window['start'],window['end'],window['samples'])!=(start,end,right-left):
            raise ValueError('ASR window differs from VAD sample gate')
        count=window['words']
        if isinstance(count,bool) or not isinstance(count,int) or count<0:
            raise ValueError('invalid per-window word count')
        selected=features['words'][offset:offset+count]
        if len(selected)!=count: raise ValueError('word count exceeds available words')
        for word in selected:
            a,b,p=word['start'],word['end'],word['probability']
            if not all(math.isfinite(x) for x in (a,b,p)) or not start<=a<=b<=end or not 0<=p<=1:
                raise ValueError('invalid or out-of-gate word')
            if not isinstance(word['text'],str): raise ValueError('invalid word text')
        offset+=count
        if language is None:
            code,probability=window['detected_language'],window['language_probability']
            if not isinstance(code,str) or not code or not math.isfinite(probability) or not 0<=probability<=1:
                raise ValueError('invalid language metadata')
            languages[code]+=1
    if offset!=len(features['words']): raise ValueError('unassigned words')
    if sum(right-left for left,right in expected)!=features['speech_samples']:
        raise ValueError('gated sample coverage mismatch')
    gates={(w['start'],w['end']) for w in windows}
    if any(tuple(w['gate']) not in gates for w in features['rejected_words']):
        raise ValueError('rejected word without a known gate')
    return {'words':offset,'windows':len(windows),'speech_samples':features['speech_samples'],
            'rejected_words':len(features['rejected_words']),'detected_language_windows':dict(languages)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--video-id',action='append',required=True)
    p.add_argument('--model-report',type=Path,required=True)
    p.add_argument('--model-id',required=True)
    p.add_argument('--language',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('use a new audit report path')
    if len(set(args.video_id))!=len(args.video_id): raise ValueError('duplicate video IDs')
    report=json.loads(args.model_report.read_text())
    if not report['complete']: raise ValueError('model download not complete')
    model=next(r for r in report['models'] if r['repo_id']==args.model_id)
    language=None if args.language=='auto' else args.language
    records=[]
    for ident in args.video_id:
        if Path(ident).name!=ident or ident in ('.','..'): raise ValueError('invalid video ID')
        cache=args.cache_dir/ident
        manifest=load_manifest(cache)
        if not cache_is_current(manifest): raise ValueError('stale source media')
        vad,reason=load_current_features(cache,manifest)
        if vad is None: raise ValueError(reason)
        path=cache/'speech_features.json'
        features=json.loads(path.read_text()); provenance=features['provenance']
        digest=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
        if not features['complete'] or features['config_hash']!=digest: raise ValueError('invalid ASR feature fingerprint')
        expected={'version':SPEECH_FEATURES_VERSION,'model':args.model_id,'revision':model['revision'],
                  'language':language,'max_seconds':28.,'beam_size':5,'temperature':0.,
                  'condition_on_previous_text':False,'vad_config_hash':vad['config_hash'],'speech':vad['speech']}
        if any(provenance.get(k)!=v for k,v in expected.items()): raise ValueError('ASR protocol or VAD gates changed')
        wav=resolve_audio_path(manifest,cache)
        if provenance['audio_sha256']!=wav_digest(wav): raise ValueError('ASR source audio changed')
        with wave.open(str(wav),'rb') as stream:
            if stream.getframerate()!=16000 or stream.getnchannels()!=1: raise ValueError('expected mono16k')
            samples=stream.getnframes()
        record={'video_id':ident,'verified':True,'feature_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                **check_words(features,vad['speech'],samples,language)}
        records.append(record);print(json.dumps(record),flush=True)
    result={'complete':True,'scope':__doc__,'model':args.model_id,'revision':model['revision'],
            'language':language,'videos':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(result,handle,indent=2,allow_nan=False)


if __name__=='__main__': main()
