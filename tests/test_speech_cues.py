import numpy as np
import pytest
from src.utils.speech_cues import speech_chunks,cue_tracks


def test_chunks_never_cross_vad_windows():
    words=[{'start':0,'end':.5,'text':' Look'},{'start':1,'end':1.5,'text':' at this.'},
           {'start':3,'end':3.5,'text':' Elsewhere'}]
    chunks=speech_chunks(words,[{'start':0,'end':2},{'start':3,'end':4}])
    assert [c['gate'] for c in chunks]==[0,1]
    result=cue_tracks(chunks,[[1,0],[0,1]],[(2,3)],min_silence=1,silence_scale=2)
    assert result['speech'][0]['deictic']==1
    assert result['speech'][0]['semantic_shift']==0
    assert result['speech'][1]['semantic_shift']==.5
    assert result['silence'][0]['long_silence']==.5


def test_phrase_boundaries_do_not_match_substrings():
    chunks=[{'start':0,'end':1,'text':'Look at thisness over thereabouts','gate':0}]
    assert cue_tracks(chunks,[[1,0]],[],min_silence=1,silence_scale=2)['speech'][0]['deictic']==0


def test_empty_asr_does_not_imply_silence():
    result=cue_tracks([],np.zeros((0,2)),[],min_silence=1,silence_scale=2)
    assert result=={'speech':[],'silence':[]}


def test_gate_violation_rejected():
    with pytest.raises(ValueError,match='outside'):
        speech_chunks([{'start':1,'end':3,'text':'x'}],[{'start':0,'end':2}])


def test_word_limit_and_zero_duration_words():
    words=[{'start':i,'end':i,'text':'x'} for i in range(3)]
    chunks=speech_chunks(words,[{'start':0,'end':2}],max_words=2)
    assert [c['text'] for c in chunks]==['x x','x']


@pytest.mark.parametrize('embedding',[[[0,0]],[[float('nan'),0]],[[1,0],[1,0]]])
def test_invalid_embeddings(embedding):
    with pytest.raises(ValueError):
        cue_tracks([{'text':'x','start':0}],embedding,[],min_silence=1,silence_scale=2)
