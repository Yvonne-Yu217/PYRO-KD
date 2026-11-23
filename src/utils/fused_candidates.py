"""Exploratory A/AV candidates with inspectable, independently weighted cues.

No reference AD is read. Event windows are coarse estimates; weights/scales must
be tuned on development data before any held-out experiment. Placement is a
separate operation in placed_selection.
"""
import math
import numpy as np
from src.utils.event_features import EXCLUDED_LABELS
from src.utils.placed_selection import Candidate
from src.utils.visual_features import VisualCueSelector


def nearest_indices(source, query):
    source=np.asarray(source,dtype=float)
    query=np.asarray(query,dtype=float)
    if source.ndim!=1 or not len(source) or not np.isfinite(source).all() or np.any(np.diff(source)<=0):
        raise ValueError('source timestamps must be finite and strictly increasing')
    right=np.minimum(np.searchsorted(source,query),len(source)-1)
    left=np.maximum(right-1,0)
    return np.where(np.abs(query-source[left])<=np.abs(query-source[right]),left,right)


def event_score(onsets, labels, clap_change, loudness_rise_db, *, sign, class_weights=None, onset_strength=None):
    """Signed event role; optional per-label overrides, speech/silence always zero.

    Class aggregation is strongest positive minus strongest negative rise, so
    duplicating a label cannot amplify the score. CLAP distance is divided by2;
    positive loudness rise saturates at20dB. These are explicit exploratory scales.
    """
    p=np.asarray(onsets,dtype=float)
    clap=np.asarray(clap_change,dtype=float)
    loud=np.asarray(loudness_rise_db,dtype=float)
    if p.ndim!=2 or p.shape[1]!=len(labels) or len(set(labels))!=len(labels):
        raise ValueError('onsets require a unique aligned vocabulary')
    if clap.shape!=(len(p),) or loud.shape!=(len(p),): raise ValueError('unaligned event tracks')
    if not all(np.isfinite(a).all() for a in (p,clap,loud)) or np.any(p<0) or np.any(p>1) or np.any(clap<0) or np.any(clap>2.00001) or np.any(loud<0):
        raise ValueError('invalid event cues')
    if sign not in (-1,1): raise ValueError('event sign must be -1 or 1')
    overrides=class_weights or {}
    if set(overrides)-set(labels) or any(not math.isfinite(v) for v in overrides.values()):
        raise ValueError('unknown or nonfinite class weight')
    weights=np.array([0 if name in EXCLUDED_LABELS else overrides.get(name,sign) for name in labels])
    weighted=p*weights
    classes=np.maximum(weighted,0).max(axis=1)-np.maximum(-weighted,0).max(axis=1)
    strength=np.zeros(len(p)) if onset_strength is None else np.asarray(onset_strength,dtype=float)
    if strength.shape!=(len(p),) or not np.isfinite(strength).all() or np.any(strength<0) or np.any(strength>1):
        raise ValueError('normalized onset strength must be aligned in [0,1]')
    return classes+sign*(clap/2+np.minimum(loud/20,1)+strength)


def make_candidates(mode, event, labels, speech_tracks, *, sign, weights,
                    visual=None, visual_embeddings=None, class_weights=None):
    """A uses only audio-derived times and scores; AV adds cached visual frames.

    weights keys: event, semantic, deictic, silence, shot, novelty. Novelty is
    returned separately for dynamic memory ranking. No implicit dev tuning.
    """
    if mode not in ('A','AV'): raise ValueError('expected A or AV')
    if set(weights)!={'event','semantic','deictic','silence','shot','novelty'} or any(not math.isfinite(v) or v<0 for v in weights.values()):
        raise ValueError('explicit nonnegative component weights required')
    event_times=np.asarray(event['times'],dtype=float)
    nse=event_score(event['class_onsets'],labels,event['clap_change'],event['loudness_rise_db'],
                    sign=sign,class_weights=class_weights,onset_strength=event.get("onset_strength"))
    if len(event_times)!=len(nse): raise ValueError('event timestamps mismatch')
    times=set(event_times.tolist())
    times.update(c['time'] for kind in ('speech','silence') for c in speech_tracks[kind])
    if mode=='AV':
        if visual is None or visual_embeddings is None: raise ValueError('AV requires visual caches')
        times.update(visual['frame_times'])
    times=sorted(times)
    if any(not math.isfinite(t) or t<0 for t in times): raise ValueError('invalid candidate time')
    indices=nearest_indices(event_times,times)
    components=[]
    for t,index in zip(times,indices):
        semantic=max((c['semantic_shift'] for c in speech_tracks['speech'] if abs(c['time']-t)<=.5),default=0)
        deictic=max((c['deictic'] for c in speech_tracks['speech'] if c['start']<=t<=c['end']),default=0)
        silence=max((c['long_silence'] for c in speech_tracks['silence'] if c['start']<=t<=c['end']),default=0)
        components.append({'event':float(nse[index]),'semantic':semantic,'deictic':deictic,
                           'silence':silence,'shot':0.0,'novelty':0.0})
    embeddings=None
    if mode=='AV':
        base=VisualCueSelector().candidates(visual)
        v_indices=nearest_indices(visual['frame_times'],times)
        embeddings=np.asarray(visual_embeddings)[v_indices]
        peak=max(visual['novelty']) or 1
        for c,i in zip(components,v_indices):
            c['shot']=float(base[i].shot_boundary)
            c['novelty']=base[i].novelty/peak
    candidates=[Candidate(t,sum(weights[k]*c[k] for k in weights if k!='novelty'),c['novelty'])
                for t,c in zip(times,components)]
    return candidates,embeddings,components
