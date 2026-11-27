"""Export exploratory placed selections as explicit event/playback generation inputs.

Requires an explicit frame-center choice. No annotation text enters the output;
attach the cached audio timeline separately for the timeline prompt condition.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.audio_features import load_current_features
from src.utils.visual_features import media_cache_problem
from src.utils.media import load_manifest,frame_window
from src.generation.prompts import WORD_BUDGET_RULE

CONDITIONS = ('U_snapped', 'O_snapped', 'V_local', 'V_memory',
              'A_trigger', 'A_suppress', 'AV_trigger', 'AV_suppress')


def report_selection(report, video, condition):
    """Resolve the explicit report formats without guessing a selector variant."""
    if condition not in CONDITIONS:
        raise ValueError('unsupported selection condition')
    if condition.startswith('V_'):
        return video['modes'][condition[2:]], report['settings']
    return video['conditions'][condition], report


def make_row(video_id, condition, selected, selection, manifest, frame_center, report_hash):
    placement=selected['placement']
    event,t,start,end=(placement[k] for k in ('event_time','time','gap_start','gap_end'))
    if not all(math.isfinite(v) for v in (event,t,start,end)) or not 0<=start<=t<end or event<0:
        raise ValueError('invalid placement interval')
    budget=math.floor((end-t)*3)
    if budget<1 or budget!=placement['word_limit']:
        raise ValueError('invalid recorded word capacity')
    if frame_center not in ('event','placement'): raise ValueError('explicit frame center required')
    center=event if frame_center=='event' else t
    window=frame_window(manifest,center,4,2.0)
    if len(window)!=4 or not all(Path(f.path).is_file() for f in window):
        raise ValueError('four existing cached frames required')
    return {'id':f'{video_id}_{condition}_{selected["candidate_index"]:05d}',
            'video_id':video_id,'dataset':'reframed','selector':condition.split('_')[0],
            'selection_condition':condition,'selection_rank':selected['rank'],
            'selection_score':selected['score'],'selection_report_sha256':report_hash,
            'protocol_status':'exploratory','time':t,'event_time':event,
            'placement_lag_seconds':t-event,'frame_center_policy':frame_center,
            'frame_center_time':center,'frames':[f.path for f in window],
            'frame_times':[f.timestamp for f in window],'transcript':'',
            'k_requested':selection['requested'],'k_shortfall':selection['shortfall'],
            'gap_start':start,'gap_end':end,'gap_seconds':end-t,
            'word_limit':budget,'word_budget_rule':WORD_BUDGET_RULE}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection-report',type=Path,required=True)
    p.add_argument('--condition',choices=CONDITIONS,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--frame-center',choices=('event','placement'),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('use a new path to preserve generation inputs')
    raw=args.selection_report.read_bytes(); report=json.loads(raw)
    digest=hashlib.sha256(raw).hexdigest()
    rows=[]; coverage=[]
    for video in report['videos']:
        video_id=video['video_id']; cache=args.cache_dir/video_id
        for name,expected in video['source_sha256'].items():
            if Path(name).name!=name: raise ValueError('source filename must be cache-local')
            if hashlib.sha256((cache/name).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'{video_id}: selection source {name} changed')
        manifest=load_manifest(cache)
        problem=media_cache_problem(cache,manifest)
        if problem: raise ValueError(f'{video_id}: {problem}')
        audio,reason=load_current_features(cache,manifest)
        if audio is None: raise ValueError(f'{video_id}: {reason}')
        selection,constraints=report_selection(report,video,args.condition)
        if selection['shortfall']!=selection['requested']-len(selection['selected']):
            raise ValueError('invalid selection budget accounting')
        times=[s['placement']['time'] for s in selection['selected']]
        if any(b-a<2 for a,b in zip(times,times[1:])): raise ValueError('placement separation violation')
        for selected in selection['selected']:
            a=selected['placement']
            if not any(a['gap_start']==g['start'] and a['gap_end']==g['end'] for g in audio['gaps']):
                raise ValueError('placement gap not in current VAD cache')
            if not -constraints['max_advance']<=a['time']-a['event_time']<=constraints['max_lag']:
                raise ValueError('placement exceeds declared bounds')
            rows.append(make_row(video_id,args.condition,selected,selection,manifest,args.frame_center,digest))
        coverage.append({'video_id':video_id,'requested':selection['requested'],
                         'written':len(selection['selected']),'shortfall':selection['shortfall']})
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('duplicate generation IDs')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        for row in rows: handle.write(json.dumps(row,allow_nan=False)+'\n')
    args.output.with_suffix('.report.json').write_text(json.dumps({'selection_report_sha256':digest,
        'condition':args.condition,'frame_center':args.frame_center,'scope':'exploratory generation inputs',
        'videos':coverage},indent=2)+'\n')
    print(json.dumps({'condition':args.condition,'rows':len(rows),'videos':len(coverage),
                      'requested':sum(r['requested'] for r in coverage),'shortfall':sum(r['shortfall'] for r in coverage)}))


if __name__=='__main__': main()
