"""CPU compatibility checks on two frozen REFRAMED development audio excerpts."""
import argparse
import gc
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
import torch
import librosa
from transformers import AutoFeatureExtractor, ASTForAudioClassification, ClapAudioModelWithProjection
from faster_whisper import WhisperModel
from src.utils.audio_features import load_current_features
from src.utils.media import load_manifest

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cache-dir',type=Path,required=True)
parser.add_argument('--model-report',type=Path,required=True)
parser.add_argument('--report',type=Path,required=True)
args=parser.parse_args()
torch.set_num_threads(4)
cache=args.cache_dir/'l5lLQICPOcnb'
manifest=load_manifest(cache)
vad,reason=load_current_features(cache,manifest)
assert vad is not None, reason
wave,sr=sf.read(manifest['audio_path'],dtype='float32')
assert sr==16000 and wave.ndim==1
paths={r['repo_id']:r['snapshot'] for r in json.loads(args.model_report.read_text())['models']}
report={'scope':'CPU offline model/input compatibility, not accuracy validation','video_id':cache.name,'checks':{}}
output=args.report
output.parent.mkdir(parents=True,exist_ok=True)
def save(): output.write_text(json.dumps(report,indent=2)+'\n')
chunk=wave[:10*sr]
path=paths['MIT/ast-finetuned-audioset-10-10-0.4593']
extractor=AutoFeatureExtractor.from_pretrained(path,local_files_only=True)
model=ASTForAudioClassification.from_pretrained(path,local_files_only=True).eval()
inputs=extractor(chunk,sampling_rate=sr,return_tensors='pt')
with torch.inference_mode(): probs=model(**inputs).logits.sigmoid()
assert probs.shape==(1,527) and torch.isfinite(probs).all()
report['checks']['ast']={'shape':list(probs.shape),'finite':True,'audio_interval':[0.0,10.0]}
save(); print('AST passed',flush=True)
del model,inputs,extractor;gc.collect()
path=paths['laion/clap-htsat-unfused']
extractor=AutoFeatureExtractor.from_pretrained(path,local_files_only=True)
model=ClapAudioModelWithProjection.from_pretrained(path,local_files_only=True).eval()
clap_wave=librosa.resample(chunk,orig_sr=sr,target_sr=48000)
inputs=extractor(clap_wave,sampling_rate=48000,return_tensors='pt')
with torch.inference_mode(): embeddings=model(**inputs).audio_embeds
assert embeddings.ndim==2 and embeddings.shape[0]==1 and torch.isfinite(embeddings).all()
report['checks']['clap']={'shape':list(embeddings.shape),'finite':True,'norm':float(embeddings.norm()),'audio_interval':[0.0,10.0]}
save();print('CLAP passed',flush=True)
del model,inputs,extractor;gc.collect()
# A distinct fixed development clip has VAD-positive intervals long enough for this check.
speech_cache=cache.parent/'CxWby8RuUwU6'
speech_manifest=load_manifest(speech_cache)
vad,reason=load_current_features(speech_cache,speech_manifest)
assert vad is not None, reason
wave,sr=sf.read(speech_manifest['audio_path'],dtype='float32')
assert sr==16000 and wave.ndim==1
start,end=next((s,e) for s,e in vad['speech'] if e-s>=2)
end=min(end,start+10)
segment=wave[int(np.ceil(start*sr)):int(np.floor(end*sr))]
model=WhisperModel(paths['Systran/faster-whisper-small.en'],device='cpu',compute_type='int8',cpu_threads=4,local_files_only=True)
segments,info=model.transcribe(segment,language='en',word_timestamps=True,beam_size=5,temperature=0.0,condition_on_previous_text=False,vad_filter=False)
words=[{'start':start+w.start,'end':start+w.end,'text':w.word} for s in segments for w in (s.words or [])]
assert all(start<=w['start']<=w['end']<=end+1/sr for w in words)
report['checks']['whisper']={'video_id':speech_cache.name,'vad_interval':[start,end],'word_count':len(words),'words':words,'compute_type':'int8'}
report['complete']=len(report['checks'])==3
save();print('Whisper passed',flush=True)
