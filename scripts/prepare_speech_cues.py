"""Prepare separate speech cue tracks on frozen REFRAMED dev clips, CPU only."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.utils.speech_cues import DEICTIC_PHRASES,speech_chunks,cue_tracks
from src.utils.audio_features import load_current_features,resolve_audio_path
from src.utils.media import load_manifest,cache_is_current

MODEL='sentence-transformers/all-mpnet-base-v2'
REVISION='e8c3b32edf5434bc2275fc9bab85f82640a19130'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--split-config',type=Path,default=Path('configs/reframed_splits.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--min-silence',type=float,required=True)
    p.add_argument('--silence-scale',type=float,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('use a new output path to preserve cue records')
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    snapshot=Path(snapshot_download(MODEL,revision=REVISION,local_files_only=True))
    model=SentenceTransformer(str(snapshot),device='cpu',local_files_only=True)
    ids=[r['video_id'] for r in json.loads(args.split_config.read_text())['clips']]
    records=[]
    for video_id in ids:
        cache=args.cache_dir/video_id
        manifest=load_manifest(cache)
        if not cache_is_current(manifest): raise ValueError(f'{video_id}: stale source')
        vad,reason=load_current_features(cache,manifest)
        if vad is None: raise ValueError(f'{video_id}: {reason}')
        speech_path=cache/'speech_features.json'
        raw=speech_path.read_bytes()
        speech=json.loads(raw)
        wav=resolve_audio_path(manifest,cache)
        if not speech['complete']: raise ValueError('incomplete ASR')
        if speech['provenance']['audio_sha256']!=hashlib.sha256(wav.read_bytes()).hexdigest():
            raise ValueError('ASR source WAV changed')
        if speech['provenance']['vad_config_hash']!=vad['config_hash'] or speech['provenance']['speech']!=vad['speech']:
            raise ValueError('ASR VAD gates changed')
        chunks=speech_chunks(speech['words'],speech['windows'])
        embeddings=(model.encode([c['text'] for c in chunks],batch_size=16,normalize_embeddings=True,
                                 show_progress_bar=False) if chunks else np.zeros((0,768)))
        tracks=cue_tracks(chunks,embeddings,[(g['start'],g['end']) for g in vad['gaps']],
                          min_silence=args.min_silence,silence_scale=args.silence_scale)
        if speech_path.read_bytes()!=raw: raise ValueError('ASR changed during cue preparation')
        records.append({'video_id':video_id,'tracks':tracks,'text_embeddings':embeddings.tolist(),
                        'speech_sha256':hashlib.sha256(raw).hexdigest(),'vad_config_hash':vad['config_hash']})
        print(json.dumps({'video_id':video_id,'chunks':len(chunks),'silence_cues':len(tracks['silence'])}),flush=True)
    report={'complete':len(records)==len(ids),'model':MODEL,'revision':REVISION,
            'model_files_sha256':{str(f.relative_to(snapshot)):hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in sorted(snapshot.rglob('*')) if f.is_file()},
            'versions':{name:importlib.metadata.version(name) for name in ('sentence-transformers','torch','transformers')},
            'settings':{'max_words':24,'phrases':DEICTIC_PHRASES,'min_silence':args.min_silence,'silence_scale':args.silence_scale},
            'split_sha256':hashlib.sha256(args.split_config.read_bytes()).hexdigest(),
            'scope':'exploratory automated cues; no diarization, human labels, threshold tuning or accuracy claim',
            'videos':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle: json.dump(report,handle,indent=2,allow_nan=False)


if __name__=='__main__': main()
