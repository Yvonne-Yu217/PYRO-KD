#!/usr/bin/env python
"""Offline CPU AST/CLAP tracks on a one-second grid of ten-second audio windows."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.audio_features import resolve_audio_path
from src.utils.media import load_manifest
from src.utils.event_features import EVENT_VERSION, EXCLUDED_LABELS, centered_window, event_times, event_cues
from src.utils.event_checkpoints import EventCheckpoints


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir',type=Path,required=True)
    parser.add_argument('--model-report',type=Path,required=True)
    parser.add_argument('--video-id',action='append')
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    if args.batch_size<1: raise ValueError('positive batch size required')
    import numpy as np
    import soundfile as sf
    import librosa
    import torch
    from transformers import AutoFeatureExtractor, ASTForAudioClassification, ClapAudioModelWithProjection
    torch.set_num_threads(4)
    models={r['repo_id']:r for r in json.loads(args.model_report.read_text())['models']}
    ast=models['MIT/ast-finetuned-audioset-10-10-0.4593']; clap=models['laion/clap-htsat-unfused']
    settings={'version':EVENT_VERSION,'models':{m['repo_id']:m['revision'] for m in (ast,clap)},
              'hop_seconds':1.0,'window_seconds':10.0,'edge_padding':'zeros','memory_windows':5,
              'device':'cpu','dtype':'float32','batch_size':args.batch_size,
              'excluded_labels':sorted(EXCLUDED_LABELS),
              'libraries':{m:importlib.metadata.version(m) for m in ('torch','transformers','librosa','numpy')}}
    paths=sorted(args.cache_dir.glob('*/manifest.json'))
    if args.video_id:
        paths=[p for p in paths if p.parent.name in args.video_id]
        if {p.parent.name for p in paths}!=set(args.video_id): raise ValueError('requested video missing')
    if not paths: raise ValueError('no prepared caches')
    report={'settings':settings,'results':[],'complete':False}
    ast_model=clap_model=None
    for path in paths:
        cache=path.parent
        try:
            audio=resolve_audio_path(load_manifest(cache),cache)
            if audio is None: raise ValueError('no audio stream; event features unavailable')
            raw=audio.read_bytes(); audio_hash=hashlib.sha256(raw).hexdigest()
            provenance={**settings,'audio_sha256':audio_hash}
            digest=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
            metadata=cache/'event_features.json'; arrays=cache/'event_features.npz'
            try: previous=json.loads(metadata.read_text())
            except (OSError,ValueError): previous={}
            if previous.get('complete') and previous.get('config_hash')==digest and arrays.exists() and hashlib.sha256(arrays.read_bytes()).hexdigest()==previous.get('arrays_sha256'):
                result={'video_id':cache.name,'status':'current','windows':previous['windows']}
            else:
                wave,sr=sf.read(io.BytesIO(raw),dtype='float32')
                if sr!=16000 or wave.ndim!=1: raise ValueError('16kHz mono prepared audio required')
                if ast_model is None:
                    ast_extractor=AutoFeatureExtractor.from_pretrained(ast['snapshot'],local_files_only=True)
                    ast_model=ASTForAudioClassification.from_pretrained(ast['snapshot'],local_files_only=True).eval()
                    clap_extractor=AutoFeatureExtractor.from_pretrained(clap['snapshot'],local_files_only=True)
                    clap_model=ClapAudioModelWithProjection.from_pretrained(clap['snapshot'],local_files_only=True).eval()
                labels=[ast_model.config.id2label[i] for i in range(ast_model.config.num_labels)]
                if not EXCLUDED_LABELS.issubset(labels): raise ValueError('unexpected AudioSet class vocabulary')
                times=event_times(len(wave),sr)
                wave48=librosa.resample(wave,orig_sr=sr,target_sr=48000)
                probabilities,embeddings=[],[]
                checkpoints=EventCheckpoints(cache/'.event-checkpoints',digest)
                reused_batches=0
                for start in range(0,len(times),args.batch_size):
                    centers=times[start:start+args.batch_size]
                    batch=checkpoints.load(start,len(centers))
                    if batch is None:
                        ast_inputs=ast_extractor([centered_window(wave,sr,t) for t in centers],sampling_rate=sr,return_tensors='pt')
                        clap_inputs=clap_extractor([centered_window(wave48,48000,t) for t in centers],sampling_rate=48000,return_tensors='pt')
                        with torch.inference_mode():
                            batch=(ast_model(**ast_inputs).logits.sigmoid().cpu().numpy(),
                                   clap_model(**clap_inputs).audio_embeds.cpu().numpy())
                        checkpoints.save(start,*batch)
                    else:
                        reused_batches+=1
                    probabilities.append(batch[0]);embeddings.append(batch[1])
                    if start==0 or (start//args.batch_size+1)%25==0 or start+len(centers)==len(times):
                        print(json.dumps({'video_id':cache.name,'status':'progress',
                            'windows_done':start+len(centers),'windows_total':len(times),
                            'reused_batches':reused_batches}),flush=True)
                probabilities=np.concatenate(probabilities); embeddings=np.concatenate(embeddings)
                rms=np.asarray([np.sqrt(np.mean(centered_window(wave,sr,t,seconds=1.0)**2)) for t in times])
                cues=event_cues(probabilities,embeddings,rms,labels)
                if hashlib.sha256(audio.read_bytes()).hexdigest()!=audio_hash: raise ValueError('audio changed during extraction')
                temporary=arrays.with_suffix('.npz.tmp')
                with temporary.open('wb') as handle:
                    np.savez_compressed(handle,times=times,probabilities=probabilities,rms=rms,**cues)
                temporary.replace(arrays)
                payload={'complete':True,'config_hash':digest,'provenance':provenance,'labels':labels,
                         'windows':len(times),'duration':len(wave)/sr,
                         'arrays_sha256':hashlib.sha256(arrays.read_bytes()).hexdigest()}
                temporary=metadata.with_suffix('.json.tmp');temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(metadata)
                result={'video_id':cache.name,'status':'computed','windows':len(times)}
        except Exception as exc:
            result={'video_id':cache.name,'status':'failed','error':repr(exc)}
        print(json.dumps(result),flush=True)
        report['results'].append(result)
        (args.cache_dir/'event_features_report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=all(r['status'] in ('computed','current') for r in report['results'])
    (args.cache_dir/'event_features_report.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if report['complete'] else 1)


if __name__=='__main__': main()
