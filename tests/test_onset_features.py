import numpy as np
import pytest
from src.utils.onset_features import bin_onsets,compute_onsets
from src.utils.fused_candidates import event_score


def test_half_open_bins_and_silence():
    raw,scaled=bin_onsets([2,4,1],[.9,1,1.9],[0,1])
    assert list(raw)==[2,4] and list(scaled)==[.5,1]
    assert bin_onsets([0],[0],[0])[1][0]==0


def test_real_silent_wave_has_no_onsets():
    result=compute_onsets(np.zeros(32000,dtype=np.float32),16000)
    assert result['normalized']==[0,0]


def test_onset_role_tracks_global_sign():
    assert event_score([[0]],['Music'],[0],[0],sign=1,onset_strength=[.7])[0]==.7
    assert event_score([[0]],['Music'],[0],[0],sign=-1,onset_strength=[.7])[0]==-.7
    with pytest.raises(ValueError): event_score([[0]],['Music'],[0],[0],sign=1,onset_strength=[2])
