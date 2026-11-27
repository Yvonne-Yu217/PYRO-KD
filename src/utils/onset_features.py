"""Cheap full-audio spectral-flux cue on the event stack's one-second grid."""
import numpy as np
from src.utils.event_features import event_times

SETTINGS={'version':1,'hop_length':160,'n_fft':512,'n_mels':64,'center':False,
          'aggregation':'maximum within each half-open one-second bin','normalization':'divide by video maximum; silent track remains zero'}


def bin_onsets(values,times,grid):
    values=np.asarray(values,dtype=float);times=np.asarray(times,dtype=float);grid=np.asarray(grid,dtype=float)
    if values.shape!=times.shape or values.ndim!=1 or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('invalid onset envelope')
    if not np.isfinite(times).all() or not len(grid) or grid[0]!=0 or not np.allclose(np.diff(grid),1):
        raise ValueError('invalid one-second time grid')
    result=np.zeros(len(grid))
    for value,t in zip(values,times):
        if 0<=t<len(grid): result[int(t)]=max(result[int(t)],value)
    normalized=result/(result.max() or 1)
    return result,normalized


def compute_onsets(wave,sr):
    import librosa
    if sr!=16000 or wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError('finite nonempty16kHz mono audio required')
    envelope=librosa.onset.onset_strength(y=wave,sr=sr,hop_length=160,n_fft=512,n_mels=64,center=False)
    times=librosa.frames_to_time(np.arange(len(envelope)),sr=sr,hop_length=160)
    grid=event_times(len(wave),sr)
    raw,normalized=bin_onsets(envelope,times,grid)
    return {'times':grid.tolist(),'strength':raw.tolist(),'normalized':normalized.tolist()}
