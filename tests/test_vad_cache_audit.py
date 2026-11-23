import copy
import pytest
from scripts.verify_vad_cache import validate_intervals


def features():
    return {'duration':10.,'speech':[[2.,4.],[6.,8.]],'speech_seconds':4.,
            'config':{'edge_margin_seconds':.25,'min_gap_seconds':1.},
            'gaps':[{'start':a,'end':b,'rms':.1,'onset':0.} for a,b in [(0.,1.75),(4.25,5.75),(8.25,10.)]]}


def test_independent_gap_accounting():
    r=validate_intervals(features(),10.)
    assert r['usable_gaps']==3 and r['usable_gap_seconds']==5.


@pytest.mark.parametrize('corruption',['boundary','missing','overlap','total','nan'])
def test_inconsistent_cache_rejected(corruption):
    f=copy.deepcopy(features())
    if corruption=='boundary': f['gaps'][0]['end']=2.
    if corruption=='missing': f['gaps'].pop()
    if corruption=='overlap': f['speech'][1][0]=3.
    if corruption=='total': f['speech_seconds']=5.
    if corruption=='nan': f['gaps'][0]['rms']=float('nan')
    with pytest.raises(ValueError): validate_intervals(f,10.)
