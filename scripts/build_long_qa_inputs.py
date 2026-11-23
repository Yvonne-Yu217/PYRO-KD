"""Prepare frozen long-video U/V QA inputs without reading answer columns.

Retains K=5/min and four frames per moment. Resolution is an inference setting;
no temporal subsampling is hidden in this builder. Timeline compression is fixed
before held-out QA: at most20 speech and20 event entries, evenly across time.
"""
import argparse,hashlib,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.generation.qa import build_qa_prompt,reject_labels
from src.utils.media import load_manifest,cache_is_complete,cache_is_current,frame_window
from src.utils.selection import k_for_duration,uniform_timestamps
from src.utils.visual_features import load_validated_features,load_embeddings,VisualCueSelector

QUESTION_COLUMNS=['question_id','video_id','question','options']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def normalize_options(options):
    if not isinstance(options,list) or len(options)!=4:
        raise ValueError('four ordered options required')
    result=[]
    for i,text in enumerate(options):
        if not isinstance(text,str) or not text.strip():raise ValueError('empty option')
        match=re.match(r'^([A-D])[.)]\s+',text)
        if match:
            if match[1]!=chr(65+i):raise ValueError('option label/order mismatch')
            text=text[match.end():]
        result.append(text)
    return result

def evenly(rows,n):
    if len(rows)<=n:return rows
    return [rows[round(i*(len(rows)-1)/(n-1))] for i in range(n)]

def global_timeline(cache,manifest):
    import numpy as np
    from src.audio.salience import class_onsets
    from src.utils.speech_cues import speech_chunks
    from src.utils.audio_features import load_current_features,resolve_audio_path
    vad,reason=load_current_features(cache,manifest)
    if vad is None:raise ValueError(reason)
    digest=sha(resolve_audio_path(manifest,cache))
    speech=json.loads((cache/'speech_features.json').read_text())
    events=json.loads((cache/'event_features.json').read_text())
    if any(not x['complete'] or x['provenance']['audio_sha256']!=digest for x in (speech,events)):
        raise ValueError('incomplete/stale audio tracks')
    if speech['provenance']['vad_config_hash']!=vad['config_hash']:
        raise ValueError('stale VAD speech gates')
    if sha(cache/'event_features.npz')!=events['arrays_sha256']:raise ValueError('changed event arrays')
    chunks=speech_chunks(speech['words'],speech['windows'])
    speech_items=[{'kind':'speech','start':r['start'],'end':r['end'],'text':r['text']} for r in chunks]
    with np.load(cache/'event_features.npz',allow_pickle=False) as a:
        detected=class_onsets(a['times'].tolist(),a['probabilities'],events['labels'])
    event_items=[{'kind':'event','start':r.time,'end':r.time,'text':r.label} for r in detected]
    entries=sorted(evenly(speech_items,20)+evenly(event_items,20),key=lambda x:(x['start'],x['end'],x['kind']))
    provenance={'policy':'20speech+20event evenly spaced entries; chronological; no question-conditioned selection',
                'speech_entries_total':len(speech_items),'event_entries_total':len(event_items),
                'source_hashes':{name:sha(cache/name) for name in ['speech_features.json','event_features.json','event_features.npz']}}
    return entries,provenance

def audio_selection(cache, cue_cache, manifest, selector_name, budget, selection_config, k):
    """Use validated cue bundles; audio-only A never loads visual features."""
    from src.audio.cues_io import load_cues_for_selection
    from src.utils.audio_features import load_current_features, resolve_audio_path
    from src.selection.registry import SelectionContext, resolve_spec, run_selector, spec_uses_visual
    vad, reason = load_current_features(cache, manifest)
    if vad is None:
        raise ValueError(f"missing/stale VAD: {reason}")
    cues, reason = load_cues_for_selection(cue_cache, resolve_audio_path(manifest, cache), vad)
    if cues is None:
        raise ValueError(f"missing/stale cue bridge: {reason}")
    if (cues.get('asr') or {}).get('source') != 'whisper-vad-gated':
        raise ValueError('long QA requires VAD-gated ASR')
    required = ['asr', 'tagger', 'clap', 'acoustic']
    if any(not str(cues.get('streams', {}).get(x, '')).startswith('ok') for x in required):
        raise ValueError('incomplete required audio streams')
    custom = selection_config.get('selectors', {}).get(selector_name)
    if not isinstance(custom, dict):
        raise ValueError('explicit development-frozen selector configuration required')
    spec = resolve_spec({'name': selector_name, **custom})
    from src.selection.two_stage import merge_config
    if merge_config(spec.get('options'))['min_separation'] != 2.0:
        raise ValueError('long QA holds minimum separation at 2 seconds')
    if selector_name == 'A_cue' and spec_uses_visual(spec):
        raise ValueError('proposal A_cue must not use visual scores')
    spec['budget'] = budget
    if budget == 'hierarchical':
        spec['hierarchy'] = selection_config.get('ablations', {}).get('hierarchy', {})
    visual = embeddings = None
    if spec_uses_visual(spec):
        visual, reason = load_validated_features(cache)
        if visual is None:
            raise ValueError(f'missing/stale visual features: {reason}')
        embeddings = load_embeddings(cache, visual)
        if embeddings is None:
            raise ValueError('running novelty requires validated embeddings')
    context = SelectionContext(cache.name, manifest['media_info']['duration'], vad, cues, visual, embeddings)
    selected = run_selector(spec, context, k)
    if selected.status != 'ok':
        raise ValueError(f'selection did not complete: {selected.status}')
    return selected.times, {'spec': spec, 'status': selected.status, 'diagnostics': selected.diagnostics,
                            'cue_sha256': sha(cue_cache / 'audio_cues.json'),
                            'vad_sha256': sha(cache / 'audio_features.json')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--membership',type=Path,default=Path('configs/video_mme_long_questions.json'))
    p.add_argument('--subset',type=Path,default=Path('configs/video_mme_long_subset.json'))
    p.add_argument('--budget',type=Path,default=Path('configs/reframed_splits.json'))
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--selector',choices=['U','V_local','V_memory','A_cue','AV'],required=True)
    p.add_argument('--audio-context',choices=['none','timeline'],required=True)
    p.add_argument('--budget-allocation', choices=['flat','hierarchical'], default='flat')
    p.add_argument('--selection-config',type=Path,help='development-frozen TOML; required for A_cue/AV')
    p.add_argument('--cue-cache-dir',type=Path,help='validated bridge cache root; defaults to cache-dir')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve prior frozen inputs')
    if args.budget_allocation == 'hierarchical' and args.selector not in ('A_cue','AV'):
        raise ValueError('hierarchical allocation requires A_cue or AV')
    selection_config = None
    if args.selector in ('A_cue','AV'):
        if args.selection_config is None: raise ValueError('explicit --selection-config required')
        from src.experiments.config import load_config
        selection_config = load_config(args.selection_config)
    membership=json.loads(args.membership.read_text());subset=json.loads(args.subset.read_text())
    if sha(args.subset)!=membership['subset_sha256'] or sha(args.annotations)!=membership['annotations_sha256']:
        raise ValueError('frozen membership/source mismatch')
    expected={r['question_id']:r['video_id'] for r in membership['questions']}
    if len(expected)!=membership['expected_questions']:raise ValueError('duplicate question IDs')
    import pyarrow.parquet as pq
    rows=pq.read_table(args.annotations,columns=QUESTION_COLUMNS,filters=[('question_id','in',list(expected))]).to_pylist()
    if len(rows)!=len(expected) or {r['question_id'] for r in rows}!=set(expected):raise ValueError('question coverage mismatch')
    sources={'question_membership':sha(args.membership),'subset':sha(args.subset),'annotations':sha(args.annotations),'budget':sha(args.budget)}
    if args.selection_config is not None: sources['selection_config'] = sha(args.selection_config)
    rate=json.loads(args.budget.read_text())['k_rule']['rate_per_minute']
    contexts={}
    for ident in sorted(set(expected.values())):
        cache=args.cache_dir/ident;manifest=load_manifest(cache)
        if not cache_is_current(manifest) or not cache_is_complete(manifest) or not manifest['audio_path']:
            raise ValueError('verified audio-bearing long media required')
        duration=manifest['media_info']['duration'];k=k_for_duration(duration,rate)
        selection_provenance = {}
        if args.selector=='U':times=uniform_timestamps(duration,k)
        elif args.selector in ('A_cue','AV'):
            times, selection_provenance = audio_selection(cache, (args.cue_cache_dir or args.cache_dir)/ident,
                manifest, args.selector, args.budget_allocation, selection_config, k)
        else:
            features,reason=load_validated_features(cache)
            if features is None:raise ValueError(reason)
            selector=VisualCueSelector()
            if args.selector=='V_memory':
                embeddings=load_embeddings(cache,features)
                if embeddings is None:raise ValueError('missing visual embeddings')
                selected=selector.select_memory(features,embeddings,k)
            else:selected=selector.select(features,k)
            times=[r.time for r in selected]
        frames=[f for t in times for f in frame_window(manifest,t,4,2.)]
        if len(frames)!=4*len(times) or any(b-a<2 for a,b in zip(times,times[1:])):
            raise ValueError('frame budget or temporal separation violation')
        context={'frames':[str(f.path) for f in frames],'frame_times':[f.timestamp for f in frames],
                 'selected_times':times,'k_requested':k,'k_realized':len(times),'k_shortfall':k-len(times),
                 'selection_provenance':selection_provenance,'budget_allocation':args.budget_allocation,
                 'duration':duration,'rate_per_minute':rate,'frames_per_moment':4,'window_seconds':2.}
        if args.audio_context=='timeline':
            context['audio_timeline'],context['audio_timeline_provenance']=global_timeline(cache,manifest)
        contexts[ident]=context
    outputs=[]
    for row in sorted(rows,key=lambda r:r['question_id']):
        if set(row)!=set(QUESTION_COLUMNS) or expected[row['question_id']]!=row['video_id']:
            raise ValueError('unexpected question columns/mapping')
        record={**contexts[row['video_id']],'id':row['question_id'],'video_id':row['video_id'],
                'dataset':'video-mme','split':'long-heldout','selector':args.selector if args.selector=='A_cue' else args.selector.split('_')[0],
                'selection_condition':args.selector + ('_'+args.budget_allocation if args.selector in ('A_cue','AV') else ''),'audio_context':args.audio_context,
                'question':row['question'],'options':normalize_options(row['options']),
                'source_sha256':sources,'protocol_status':'fixed membership; no outcome-based tuning'}
        reject_labels(record);build_qa_prompt(record);outputs.append(record)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        for row in outputs:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps({'questions':len(outputs),'videos':len(contexts),'columns_read':QUESTION_COLUMNS,
                      'frame_counts':{v:len(r['frames']) for v,r in contexts.items()}}))

if __name__=='__main__':main()
