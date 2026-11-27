#!/usr/bin/env python
"""Audit event arrays against model vocabulary, source WAV and recorded configuration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import soundfile as sf
from src.utils.audio_features import resolve_audio_path
from src.utils.event_features import EVENT_VERSION, EXCLUDED_LABELS, centered_window, event_cues, event_times
from src.utils.media import load_manifest, cache_is_current


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--split-config',type=Path,default=Path('configs/reframed_splits.json'))
    p.add_argument('--model-report',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--subset-config',type=Path,help='frozen long-video subset instead of REFRAMED clips')
    p.add_argument('--video-id',action='append',help='audit only these IDs from the frozen subset; reports explicit coverage')
    args=p.parse_args()
    source_config=args.subset_config or args.split_config
    selection=json.loads(source_config.read_text())
    all_ids=[r['video_id'] for r in selection['videos' if args.subset_config else 'clips']]
    if not all_ids or len(set(all_ids))!=len(all_ids): raise ValueError('unique frozen IDs required')
    ids=all_ids
    if args.video_id:
        if len(set(args.video_id))!=len(args.video_id) or not set(args.video_id)<=set(all_ids):
            raise ValueError('requested audit IDs must belong to frozen subset')
        ids=[ident for ident in all_ids if ident in args.video_id]
    models={r['repo_id']:r for r in json.loads(args.model_report.read_text())['models']}
    ast=models['MIT/ast-finetuned-audioset-10-10-0.4593']
    config=json.loads((Path(ast['snapshot'])/'config.json').read_text())
    labels=[config['id2label'][str(i)] for i in range(len(config['id2label']))]
    records=[]
    for video_id in ids:
        cache=args.cache_dir/video_id
        try:
            meta=json.loads((cache/'event_features.json').read_text())
            if not meta.get('complete'): raise ValueError('event metadata incomplete')
            provenance=meta['provenance']
            expected=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
            if meta['config_hash']!=expected: raise ValueError('configuration hash mismatch')
            if provenance['version']!=EVENT_VERSION or provenance['excluded_labels']!=sorted(EXCLUDED_LABELS): raise ValueError('event version/class mask mismatch')
            expected_models={repo:models[repo]['revision'] for repo in ('MIT/ast-finetuned-audioset-10-10-0.4593','laion/clap-htsat-unfused')}
            if provenance['models']!=expected_models: raise ValueError('model revision mismatch')
            if provenance['hop_seconds']!=1.0 or provenance['window_seconds']!=10.0 or provenance['edge_padding']!='zeros': raise ValueError('window protocol mismatch')
            if provenance['memory_windows']!=5: raise ValueError('event memory protocol mismatch')
            if meta['labels']!=labels: raise ValueError('class vocabulary mismatch')
            raw=(cache/'event_features.npz').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=meta['arrays_sha256']: raise ValueError('array hash mismatch')
            manifest=load_manifest(cache)
            if not cache_is_current(manifest): raise ValueError('preprocessed source has changed')
            audio=resolve_audio_path(manifest,cache)
            if hashlib.sha256(audio.read_bytes()).hexdigest()!=provenance['audio_sha256']: raise ValueError('source WAV changed')
            wave,sr=sf.read(audio,dtype='float32')
            if sr!=16000 or wave.ndim!=1 or abs(meta['duration']-len(wave)/sr)>1/sr:
                raise ValueError('audio format or recorded duration mismatch')
            with np.load(cache/'event_features.npz',allow_pickle=False) as arrays:
                tracks={k:arrays[k] for k in arrays.files}
            expected_times=event_times(len(wave),sr,provenance['hop_seconds'])
            n=len(expected_times)
            if tracks['probabilities'].shape!=(n,527) or tracks['clap_embeddings'].shape!=(n,512): raise ValueError('feature dimensions/coverage mismatch')
            if meta['windows']!=n or not np.allclose(tracks['times'],expected_times): raise ValueError('time grid mismatch')
            if not all(np.isfinite(a).all() for a in tracks.values()): raise ValueError('nonfinite features')
            if not np.allclose(np.linalg.norm(tracks['clap_embeddings'],axis=1),1,atol=1e-5): raise ValueError('CLAP embeddings not normalized')
            rms=np.asarray([np.sqrt(np.mean(centered_window(wave,sr,t,seconds=1.0)**2)) for t in expected_times])
            if not np.allclose(tracks['rms'],rms,atol=1e-7): raise ValueError('RMS does not match WAV')
            cues=event_cues(tracks['probabilities'],tracks['clap_embeddings'],rms,labels,provenance['memory_windows'])
            for key,values in cues.items():
                if not np.allclose(tracks[key],values,atol=1e-5): raise ValueError(f'{key} inconsistent with cached base features')
            records.append({'video_id':video_id,'verified':True,'windows':n})
        except Exception as exc:
            records.append({'video_id':video_id,'verified':False,'error':repr(exc)})
    report={'complete':bool(records) and all(r['verified'] for r in records),'expected':len(ids),
            'frozen_subset_size':len(all_ids),'full_subset_coverage':ids==all_ids,
            'selection_config_sha256':hashlib.sha256(source_config.read_bytes()).hexdigest(),
            'verified':sum(r['verified'] for r in records),'videos':records,
            'scope':'cache/source/shape/numerical consistency; not detector accuracy or re-inference'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='videos'}))
    raise SystemExit(0 if report['complete'] else 1)


if __name__=='__main__': main()
