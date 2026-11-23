"""Ablation 3: derive Whisper event tags from verified VAD-gated ASR, no inference.

Removes AST events and CLAP change only. VAD, speech cues, acoustic controls,
placement and selector weights are preserved. Empty tags mean observed zero
Whisper emissions, not missing ASR and not absence of real sound events.
"""
import argparse,copy,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.audio.asr import whisper_tag_events
from src.audio.cues_io import load_cues_for_selection,malformed_cues,write_cues
from src.utils.audio_features import load_current_features,resolve_audio_path
from src.utils.media import load_manifest

POLICY='gated-whisper-tags-v1; replace AST events; remove CLAP; retain acoustic and speech controls'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def derive(record, source_path, source_sha256):
    if (record.get('asr') or {}).get('source')!='whisper-vad-gated' or not str(record.get('streams',{}).get('asr','')).startswith('ok'):
        raise ValueError('verified VAD-gated ASR required')
    result=copy.deepcopy(record)
    result['events']=[event.to_dict() for event in whisper_tag_events(record['asr']['segments'])]
    result['tagger']=None
    result['clap']=None
    result['streams'].update(tagger='disabled: replaced by gated Whisper tags',
                             clap='disabled: Whisper-only event-source ablation',whisper_tags='ok')
    result.setdefault('bridge_sources',{})[str(Path(source_path).resolve())]=source_sha256
    result['ablation3']={'policy':POLICY,'source_sha256':source_sha256,'event_count':len(result['events']),
                        'interpretation':'emitted bracketed tags only; zero tags are not evidence of acoustic silence'}
    result['config_hash']=hashlib.sha256(json.dumps({'source':source_sha256,'policy':POLICY},sort_keys=True).encode()).hexdigest()
    problem=malformed_cues(result)
    if problem:raise ValueError(problem)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root',required=True,type=Path)
    p.add_argument('--output-root',required=True,type=Path)
    p.add_argument('--video-id',action='append',required=True,help='repeat frozen membership IDs explicitly')
    args=p.parse_args()
    if len(args.video_id)!=len(set(args.video_id)):raise ValueError('duplicate membership IDs')
    report=[]
    for ident in args.video_id:
        if Path(ident).name!=ident:raise ValueError('invalid video ID')
        cache=args.cache_root/ident;manifest=load_manifest(cache)
        vad,reason=load_current_features(cache,manifest)
        if vad is None:raise ValueError(reason)
        source=cache/'audio_cues.json'
        cues,reason=load_cues_for_selection(cache,resolve_audio_path(manifest,cache),vad)
        if cues is None:raise ValueError(reason)
        record=derive(cues,source,sha(source))
        dest=args.output_root/ident;dest.mkdir(parents=True,exist_ok=True)
        out=dest/'audio_cues.whisper_tags.json'
        if out.exists() and json.loads(out.read_text())!=record:raise ValueError('refuse stale-output reuse')
        for item in cache.iterdir():
            if item.name=='audio_cues.whisper_tags.json':continue
            link=dest/item.name
            if not link.exists():link.symlink_to(item.resolve(),target_is_directory=item.is_dir())
            elif link.resolve()!=item.resolve():raise ValueError('conflicting output view')
        if not out.exists():write_cues(dest,record,out.name)
        report.append({'video_id':ident,'whisper_tags':len(record['events']),'source_sha256':sha(source)})
    (args.output_root/'whisper_tags_report.json').write_text(json.dumps({'complete':True,'policy':POLICY,'videos':report},indent=2)+'\n')
    print(json.dumps({'complete':True,'videos':len(report),'tags':sum(r['whisper_tags'] for r in report)}))

if __name__=='__main__':main()
