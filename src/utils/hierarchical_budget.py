"""Explicit segmentation and capacity-aware fixed-budget allocation for long video.

No labels, model inference or implicit K reduction. These mechanics require
caller-chosen boundary sources and development-frozen settings before evaluation.
"""
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
import math


def segment_video(duration, boundary_times, *, min_seconds, max_seconds):
    """Retain separated scene/silence/music boundaries and subdivide long spans.

    Boundary candidates are sorted and deduplicated. A boundary is kept if both
    it and the final endpoint leave at least min_seconds. Spans over max_seconds
    are divided equally, avoiding tiny artificial tails. A video shorter than
    min_seconds remains one segment. No content boundaries are inferred here.
    """
    if not all(math.isfinite(x) and x>0 for x in (duration,min_seconds,max_seconds)) or max_seconds<2*min_seconds:
        raise ValueError('positive durations and max_seconds >= 2*min_seconds required')
    if any(not math.isfinite(t) or not 0<=t<=duration for t in boundary_times):
        raise ValueError('boundary outside video')
    kept=[0.0]
    for t in sorted(set(boundary_times)):
        if t-kept[-1]>=min_seconds and duration-t>=min_seconds:
            kept.append(float(t))
    kept.append(float(duration))
    result=[]
    for start,end in zip(kept,kept[1:]):
        count=math.ceil((end-start)/max_seconds)
        points=[start+(end-start)*i/count for i in range(count)]+[end]
        result.extend(zip(points,points[1:]))
    return result


def event_densities(segments, event_times, event_weights=None):
    """Unsigned event mass per second; events on a boundary belong to the right."""
    if not segments or segments[0][0]!=0 or any(not (math.isfinite(a) and math.isfinite(b) and a<b) for a,b in segments):
        raise ValueError('ordered contiguous segments starting at zero required')
    if any(a[1]!=b[0] for a,b in zip(segments,segments[1:])):
        raise ValueError('segments must be contiguous')
    weights=[1.0]*len(event_times) if event_weights is None else event_weights
    if len(weights)!=len(event_times): raise ValueError('event weights must align')
    mass=[0.0]*len(segments)
    starts=[s[0] for s in segments]
    for time,weight in zip(event_times,weights):
        if not math.isfinite(time) or not 0<=time<=segments[-1][1] or not math.isfinite(weight) or weight<0:
            raise ValueError('events must be in bounds with finite nonnegative mass')
        mass[min(bisect_right(starts,time)-1,len(segments)-1)]+=weight
    return [m/(b-a) for m,(a,b) in zip(mass,segments)]


@dataclass(frozen=True)
class BudgetAllocation:
    requested: int
    quotas: list[int]
    capacity_shortfall: int
    floor_shortfall_segments: list[int]


def allocate_budget(densities,k,*,floor,capacities=None):
    """Quiet-segment floor then capped proportional largest-remainder allocation.

    K too small for the stated floor fails explicitly. A segment's physical
    capacity can prevent its floor; such segments and total capacity shortfall
    are reported. Residual budget is redistributed across uncapped segments.
    If all eligible densities are zero, distribute evenly. Exact fractions make
    remainder ties deterministic (earlier segment first), conserving integers.
    """
    if not densities or any(not math.isfinite(w) or w<0 for w in densities):
        raise ValueError('finite nonnegative segment densities required')
    if any(isinstance(x,bool) or not isinstance(x,int) or x<0 for x in (k,floor)):
        raise ValueError('k and floor must be nonnegative integers')
    n=len(densities)
    if k<n*floor: raise ValueError('requested K cannot cover the quiet-segment floor; merge segments or revise protocol')
    caps=[k]*n if capacities is None else list(capacities)
    if len(caps)!=n or any(isinstance(c,bool) or not isinstance(c,int) or c<0 for c in caps):
        raise ValueError('capacities must be aligned nonnegative integers')
    quotas=[min(floor,c) for c in caps]
    target=min(k,sum(caps));remaining=target-sum(quotas)
    weights=[Fraction(str(w)) for w in densities]
    while remaining:
        eligible=[i for i in range(n) if quotas[i]<caps[i]]
        total=sum(weights[i] for i in eligible)
        current={i:weights[i] if total else Fraction(1) for i in eligible}
        total=sum(current.values())
        ideal={i:remaining*current[i]/total for i in eligible}
        saturated=[i for i in eligible if ideal[i]>=caps[i]-quotas[i]]
        if saturated:
            for i in saturated:
                assigned=caps[i]-quotas[i];quotas[i]+=assigned;remaining-=assigned
            continue
        assigned={i:int(ideal[i]) for i in eligible}
        leftovers=remaining-sum(assigned.values())
        order=sorted(eligible,key=lambda i:(-(ideal[i]-assigned[i]),i))
        for i in order[:leftovers]: assigned[i]+=1
        for i,value in assigned.items(): quotas[i]+=value
        remaining=0
    return BudgetAllocation(k,quotas,k-target,[i for i,c in enumerate(caps) if c<floor])
