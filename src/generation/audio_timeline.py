"""Compact, explicitly estimated speech/event context shared by caption prompts."""
import hashlib
import json
import math


def timeline_hash(entries):
    return hashlib.sha256(json.dumps(entries,sort_keys=True,allow_nan=False).encode()).hexdigest()


def render_timeline(entries):
    if not isinstance(entries,list):
        raise ValueError('timeline must be an explicit list; missing audio is not an empty timeline')
    lines=[]
    for item in sorted(entries,key=lambda x:(x['start'],x['end'],x['kind'],x['text'])):
        start,end=item['start'],item['end']
        if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in (start,end)) or not 0<=start<=end:
            raise ValueError('invalid timeline timestamps')
        if item['kind'] not in ('speech','event') or not isinstance(item['text'],str) or not item['text'].strip():
            raise ValueError('invalid timeline entry')
        minutes,seconds=divmod(start,60)
        text=' '.join(item['text'].split())
        # JSON quotes keep arbitrary decoded words distinct from prompt instructions.
        lines.append(f'[{int(minutes)}:{seconds:04.1f} {item["kind"]}: {json.dumps(text,ensure_ascii=False)}]')
    return ('Automatic audio estimates (may contain errors):\n' + '\n'.join(lines) if lines else
            'Automatic audio estimates: no speech or selected events detected in this context.')


def contextual_timeline(words,events,center,window=15.0,max_entries=40):
    if not math.isfinite(center) or center<0 or window<=0 or max_entries<1:
        raise ValueError('invalid timeline context settings')
    left,right=max(0,center-window),center+window
    nearby=[w for w in words if left<=w['start'] and w['end']<=right]
    entries=[]
    group=[]
    def flush():
        if group:
            entries.append({'kind':'speech','start':group[0]['start'],'end':group[-1]['end'],
                            'text':''.join(w['text'] for w in group).strip()})
    for word in nearby:
        if group and (len(group)>=12 or word['start']-group[-1]['end']>0.7):
            flush();group=[]
        group.append(word)
    flush()
    entries.extend(e for e in events if left<=e['start']<=right)
    entries.sort(key=lambda x:(x['start'],x['end'],x['kind'],x['text']))
    # Keep nearest entries when the context is crowded, then restore chronology.
    chosen=sorted(entries,key=lambda x:(abs(x['start']-center),x['start'],x['kind'],x['text']))[:max_entries]
    chosen.sort(key=lambda x:(x['start'],x['end'],x['kind'],x['text']))
    render_timeline(chosen)  # validates the serialized prompt data
    return chosen,len(entries)-len(chosen)


def check_audio_resume(previous,rows,mode):
    current={r['id']:r for r in rows}
    for old in previous:
        if old.get('audio_context','legacy')!=mode:
            raise ValueError('output audio context differs; use a separate output file')
        if mode=='timeline' and old['id'] in current:
            expected=timeline_hash(current[old['id']]['audio_timeline'])
            if old.get('audio_timeline_sha256')!=expected:
                raise ValueError('output timeline differs; use a separate output file')
