"""Descriptive event-window association with AD starts, not detector accuracy."""
import math
import numpy as np


def alignment(times, probabilities, rises, ad_starts, *, probability=.5, rise=.1, tolerance=2.):
    t=np.asarray(times,dtype=float);p=np.asarray(probabilities,dtype=float);r=np.asarray(rises,dtype=float)
    truth=np.asarray(ad_starts,dtype=float)
    if (t.ndim!=1 or not len(t) or p.ndim!=2 or p.shape!=r.shape or len(p)!=len(t)
            or truth.ndim!=1 or not all(np.isfinite(x).all() for x in (t,p,r,truth))
            or (t<0).any() or (truth<0).any() or (np.diff(t)<=0).any()
            or (p<0).any() or (p>1).any() or (r<0).any() or (r>1).any()):
        raise ValueError('invalid event or AD arrays')
    if (not all(math.isfinite(x) for x in (probability,rise,tolerance))
            or not 0<=probability<=1 or not 0<rise<=1 or tolerance<0):
        raise ValueError('invalid explicit alignment thresholds')
    near=(np.abs(t[:,None]-truth[None,:])<=tolerance).any(axis=1)
    detected=(p>=probability)&(r>=rise)
    output=[]
    for active in detected.T:
        both=int(np.sum(active&near));event_only=int(np.sum(active&~near))
        ad_only=int(np.sum(~active&near));neither=int(np.sum(~active&~near))
        event_count=both+event_only;no_event=ad_only+neither
        denominator=math.sqrt(event_count*no_event*(both+ad_only)*(event_only+neither))
        output.append(dict(windows=len(t),event_windows=event_count,near_ad_windows=int(near.sum()),
            both=both,event_only=event_only,ad_only=ad_only,neither=neither,
            near_ad_given_event=both/event_count if event_count else None,
            near_ad_without_event=ad_only/no_event if no_event else None,
            phi=(both*neither-event_only*ad_only)/denominator if denominator else None))
    return output
