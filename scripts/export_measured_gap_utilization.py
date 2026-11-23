"""Measured 1.2x narration use of all speech-free gaps, including unused gaps.

Primary ratio is unique narration time inside gaps / total gap time. It is
bounded by one and includes zero-output videos. Layered intersections and nominal
start-assigned durations are separate quantities that may exceed one.
"""
import argparse,csv,json,math,os,sys,wave
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.speech_overlap import union,intersection_seconds,evaluate_intervals
from scripts.export_core_temporal_metrics import sha,strict_rows,same_number
from src.generation.outputs import final_rows,is_success


def gap_metrics(narrations,speech,duration):
    speech=union(speech)
    if duration<=0 or not math.isfinite(duration) or any(e>duration for _,e in speech):raise ValueError('valid bounded speech/video required')
    gaps=[];cursor=0.
    for s,e in speech:
        if s>cursor:gaps.append((cursor,s))
        cursor=e
    if cursor<duration:gaps.append((cursor,duration))
    intervals=[(r['start'],r['start']+r['duration']) for r in narrations]
    merged=union(intervals);rows=[];per_caption=[]
    for s,e in gaps:
        used=intersection_seconds(merged,[(s,e)])
        layered=sum(intersection_seconds([x],[(s,e)]) for x in intervals)
        starts=[r for r in narrations if s<=r['start']<e]
        nominal=sum(r['duration'] for r in starts)
        per_caption += [{'id':r['id'],'ratio':r['duration']/(e-r['start'])} for r in starts]
        rows.append({'start':s,'end':e,'gap_seconds':e-s,'unique_narration_seconds':used,
                     'utilization':used/(e-s),'layered_intersection_seconds':layered,
                     'layered_intersection_ratio':layered/(e-s),'nominal_start_duration_seconds':nominal,
                     'nominal_start_duration_ratio':nominal/(e-s),'starts':len(starts)})
    total=sum(r['gap_seconds'] for r in rows);used=sum(r['unique_narration_seconds'] for r in rows)
    layered=sum(r['layered_intersection_seconds'] for r in rows)
    return {'gaps':rows,'total_gap_seconds':total,'unique_narration_in_gaps_seconds':used,
            'pooled_utilization':used/total if total else None,
            'mean_gap_utilization':sum(r['utilization'] for r in rows)/len(rows) if rows else None,
            'unused_gaps':sum(r['unique_narration_seconds']==0 for r in rows),
            'layered_in_gap_seconds':layered,'excess_layers_in_gaps_seconds':max(0.,layered-used),
            'per_caption_remaining_gap_ratios':per_caption,
            'captions_without_start_gap':len(narrations)-len(per_caption),
            'mean_caption_remaining_gap_ratio':sum(r['ratio'] for r in per_caption)/len(per_caption) if per_caption else None}


def summarize_video(narrations,vad,words,duration):
    measured=evaluate_intervals(narrations,vad,words,duration)
    tracks={'vad':union(vad),'whisper':union(words),'union':union(list(vad)+list(words))}
    return {'rendered':len(narrations),'original_collision_metrics':measured,
            'tracks':{name:gap_metrics(narrations,speech,duration) for name,speech in tracks.items()}}


def aggregate(videos):
    result={'expected_videos':len(videos),'videos_with_renders':sum(v['rendered']>0 for v in videos),
            'zero_render_videos':sum(v['rendered']==0 for v in videos),'rendered':sum(v['rendered'] for v in videos),'tracks':{}}
    for name in ('vad','whisper','union'):
        tracks=[v['tracks'][name] for v in videos];eligible=[t['pooled_utilization'] for t in tracks if t['pooled_utilization'] is not None]
        denominator=sum(t['total_gap_seconds'] for t in tracks);numerator=sum(t['unique_narration_in_gaps_seconds'] for t in tracks)
        gaps=[g for t in tracks for g in t['gaps']]
        ratios=[r['ratio'] for t in tracks for r in t['per_caption_remaining_gap_ratios']]
        result['tracks'][name]={'total_gap_seconds':denominator,'unique_narration_in_gaps_seconds':numerator,
          'pooled_utilization':numerator/denominator if denominator else None,
          'macro_video_utilization':sum(eligible)/len(eligible) if eligible else None,'videos_with_gaps':len(eligible),
          'captions_with_start_gap':len(ratios),'captions_without_start_gap':sum(t['captions_without_start_gap'] for t in tracks),
          'mean_caption_remaining_gap_ratio':sum(ratios)/len(ratios) if ratios else None,
          'gaps':len(gaps),'unused_gaps':sum(g['unique_narration_seconds']==0 for g in gaps),
          'mean_gap_utilization':sum(g['utilization'] for g in gaps)/len(gaps) if gaps else None,
          'layered_in_gap_seconds':sum(t['layered_in_gap_seconds'] for t in tracks),
          'excess_layers_in_gaps_seconds':sum(t['excess_layers_in_gaps_seconds'] for t in tracks)}
    result['speech_collision_seconds']=sum(v['original_collision_metrics']['speech_overlap_seconds_timeline']['union'] for v in videos)
    result['out_of_video_seconds_sum']=sum(v['original_collision_metrics']['out_of_video_seconds_sum'] for v in videos)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--tts-root',required=True,type=Path)
    p.add_argument('--cache-root',required=True,type=Path)
    p.add_argument('--matrix',choices=['core','ablation5'],default='core')
    p.add_argument('--output-dir',required=True,type=Path)
    a=p.parse_args();config=a.config.resolve();tts=a.tts_root.resolve();cache_root=a.cache_root.resolve();out=a.output_dir.resolve()
    if out.exists():raise FileExistsError('preserve prior derived metrics')
    os.chdir(config.parents[2])
    from src.experiments.config import load_config
    from src.experiments.data import _subset_ids
    from src.experiments.matrix import core_generation_runs,ablation5_runs
    from src.experiments.generation_stage import generations_path,moments_path
    from src.utils.media import load_manifest,cache_is_current
    from src.utils.audio_features import load_current_features,resolve_audio_path
    cfg=load_config(config);ds=cfg['dataset'];ids=_subset_ids(ds,ds['subset'])
    expected=10 if ds['subset']=='dev' else 30 if ds['subset']=='test_sample' else None
    if ds['name']!='reframed' or expected is None or len(ids)!=expected or len(set(ids))!=expected:raise ValueError('frozen10dev or30test membership required')
    exp=Path(cfg['paths']['output_root'])/cfg['experiment']['name'];sources={str(config):sha(config),str(Path(ds['splits_config']).resolve()):sha(ds['splits_config'])}
    cache={}
    for ident in ids:
        root=cache_root/ident;m=load_manifest(root);vad,reason=load_current_features(root,m)
        if not cache_is_current(m) or vad is None:raise ValueError('stale media/VAD '+str(reason))
        speech=json.loads((root/'speech_features.json').read_text())
        if not speech['complete'] or speech['provenance']['audio_sha256']!=sha(resolve_audio_path(m,root)) or speech['provenance']['vad_config_hash']!=vad['config_hash'] or speech['provenance']['speech']!=vad['speech']:raise ValueError('stale speech cache')
        hashes={name:sha(root/name) for name in ('manifest.json','audio_features.json','speech_features.json')}
        sources.update({str(root/name):h for name,h in hashes.items()})
        cache[ident]=(m['media_info']['duration'],vad['speech'],[(w['start'],w['end']) for w in speech['words']],hashes)
    runs=core_generation_runs(cfg) if a.matrix=='core' else ablation5_runs(cfg)
    runs=[r for r in runs if r.model_label in ('qwen','omni')]
    results=[]
    for run in runs:
        gp=generations_path(exp,run);mp=moments_path(exp,run.selection)
        rows=final_rows(strict_rows(gp));pred={r['id']:r for r in rows};moments=strict_rows(mp)
        if len(pred)!=len(rows) or set(pred)!={r['id'] for r in moments} or any(r['video_id'] not in ids for r in rows):raise ValueError('complete unique generation attempts required')
        valid={r['id']:r for r in rows if is_success(r) and r.get('output','').strip()};sources.update({str(gp):sha(gp),str(mp):sha(mp)})
        rp=tts/run.run_id/'rendered/report.json';rendered=rp.parent/'rendered.jsonl';op=tts/run.run_id/'overlap/report.json'
        by_video={};bound={}
        if valid:
            render_meta=json.loads(rp.read_text());overlap=json.loads(op.read_text())
            if not render_meta['complete'] or render_meta['input_sha256']!=sha(gp) or not overlap['complete'] or overlap['playback_speed']!=1.2 or overlap['input_sha256']!=sha(rendered):raise ValueError('TTS source binding mismatch')
            rendered_rows=strict_rows(rendered);bound={r['id']:r for r in rendered_rows}
            if len(bound)!=len(rendered_rows) or set(bound)!=set(valid):raise ValueError('TTS valid coverage mismatch')
            by_video={v['video_id']:v for v in overlap['videos']}
            if len(by_video)!=len(overlap['videos']) or not set(by_video)<=set(ids):raise ValueError('TTS video membership mismatch')
            sources.update({str(p):sha(p) for p in (rp,rendered,op)})
        videos=[];seen=set()
        for ident in ids:
            duration,vad,words,hashes=cache[ident];narrations=[];previous=by_video.get(ident)
            if previous:
                if previous['source_sha256']!=hashes:raise ValueError('speech sources differ from overlap report')
                arts={r['id']:r for r in previous['artifacts']}
                for row in previous['rows']:
                    key=row['id'];item=bound[key];art=arts[key]
                    if key in seen or item['video_id']!=ident or item['output']!=valid[key]['output'] or not same_number(row['start'],valid[key]['time']):raise ValueError('rendered prediction mismatch')
                    seen.add(key)
                    import hashlib
                    text_hash=hashlib.sha256(item['output'].encode()).hexdigest()
                    if text_hash!=item['tts_text_sha256'] or text_hash!=art['text_sha256'] or sha(item['tts_audio'])!=art['source_sha256'] or sha(art['rendered_path'])!=art['rendered_sha256']:raise ValueError('waveform/text binding mismatch')
                    with wave.open(art['rendered_path'],'rb') as wav:measured=wav.getnframes()/wav.getframerate()
                    if not same_number(measured,row['duration']) or not same_number(measured,art['duration']):raise ValueError('rendered duration mismatch')
                    narrations.append({'id':key,'start':row['start'],'duration':measured})
            measured=summarize_video(narrations,vad,words,duration)
            if previous:
                for key in ('speech_overlap_seconds_timeline','out_of_video_seconds_sum','narration_excess_layer_seconds'):
                    if measured['original_collision_metrics'][key]!=previous[key]:raise ValueError('existing collision report mismatch')
            videos.append({'video_id':ident,**measured})
        if seen!=set(valid):raise ValueError('missing measured valid waveform')
        results.append({'run_id':run.run_id,'selector':run.selection.name,'prior':run.prior,'model':run.model_label,'timeline':run.timeline,
                        'coverage':{'prepared':len(moments),'attempted':len(rows),'valid':len(valid),'failed_or_empty':len(rows)-len(valid)},
                        'aggregate':aggregate(videos),'videos':videos})
    for path,h in sources.items():
        if sha(path)!=h:raise ValueError('source changed during export')
    out.mkdir(parents=True)
    payload={'scope':ds['subset'],'definition':'union of measured1.2x narration intersected with all speech-free gaps / total duration of those gaps; include unused gaps and zero-output videos',
             'primary_speech_track':'union of VAD and VAD-gated Whisper word intervals','gap_boundaries':'complement within[0,video_duration]; no safety-margin trimming',
             'nominal_definition':'sum of full narration durations whose starts lie in a gap / full gap duration; may exceed1 or include later speech/out-of-video portions',
             'layered_definition':'sum of per-narration intersections with a gap / gap duration; counts overlapping narration layers and may exceed1',
             'per_caption_definition':'measured duration / remaining gap from narration start; undefined for starts in speech or outsidevideo',
             'sources':sources,'exporter_sha256':sha(__file__),'results':results}
    (out/'measured-gap-utilization.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    summary=[]
    for r in results:
        ag=r['aggregate'];t=ag['tracks']['union']
        summary.append({'selector':r['selector'],'prior':r['prior'],'model':r['model'],'timeline':r['timeline'],'videos':ag['expected_videos'],'zero_render_videos':ag['zero_render_videos'],'rendered':ag['rendered'],**t,'speech_collision_seconds':ag['speech_collision_seconds'],'out_of_video_seconds_sum':ag['out_of_video_seconds_sum']})
    with (out/'measured-gap-utilization.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    lines=['# Measured gap utilization','',f"Scope: {ds['subset']}; {expected} frozen videos in every cell. Unique measured narration occupancy of all VAD/Whisper-union gaps; unused gaps and zero-output videos remain in denominators. Failed captions contribute no narration. This is a machine-defined speech-free measure, not human-validated silence.",'','| Selector | Prior | Model | Timeline | Renders | Zero-render videos | Unique gap use | All-gap seconds | Excess in-gap layers (s) |','|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['selector']} | {r['prior']} | {r['model']} | {r['timeline']} | {r['rendered']} | {r['zero_render_videos']} | {format(r['pooled_utilization'], '.4f') if r['pooled_utilization'] is not None else 'NA'} | {r['total_gap_seconds']:.3f} | {r['excess_layers_in_gaps_seconds']:.3f} |")
    (out/'measured-gap-utilization.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'complete':True,'cells':len(results),'videos_per_cell':expected,'output':str(out)}))

if __name__=='__main__':main()
