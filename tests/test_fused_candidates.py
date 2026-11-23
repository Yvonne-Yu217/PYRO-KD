import numpy as np
import pytest
from src.utils.fused_candidates import event_score,make_candidates,nearest_indices


def test_speech_mask_sign_and_class_override():
    labels=['Speech','Music','Clapping']
    values=[[1,.4,.2]]
    assert event_score(values,labels,[0],[0],sign=1)[0]==.4
    assert event_score(values,labels,[0],[0],sign=-1)[0]==-.4
    assert event_score(values,labels,[0],[0],sign=1,class_weights={'Music':-1})[0]==pytest.approx(-.2)
    assert event_score(values,labels,[0],[0],sign=1,class_weights={'Speech':100})[0]==.4


def test_nearest_ties_choose_earlier():
    assert list(nearest_indices([0,2,4],[1,3,5]))==[0,1,2]


def test_audio_mode_is_independent_of_visual_inputs():
    event={'times':[0,1],'class_onsets':[[0],[.5]],'clap_change':[0,0],'loudness_rise_db':[0,0]}
    tracks={'speech':[],'silence':[]}
    weights={k:1 for k in ('event','semantic','deictic','silence','shot','novelty')}
    a=make_candidates('A',event,['Music'],tracks,sign=1,weights=weights)
    b=make_candidates('A',event,['Music'],tracks,sign=1,weights=weights,visual={'invalid':True},visual_embeddings='bad')
    assert a==b
    assert a[1] is None
    assert a[0][1].base_score==.5


@pytest.mark.parametrize('kwargs',[{'sign':0},{'sign':1,'class_weights':{'unknown':1}},{'sign':1,'class_weights':{'Music':float('nan')}}])
def test_bad_sign_or_class_map_rejected(kwargs):
    with pytest.raises(ValueError): event_score([[.1]],['Music'],[0],[0],**kwargs)
