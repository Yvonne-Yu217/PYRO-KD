"""Deterministic audio-event windows and non-speech cue tracks."""
import math
import numpy as np

EVENT_VERSION = 1
# AudioSet Speech descendants, plus Whispering and Silence. Keep laughter/music.
EXCLUDED_LABELS = frozenset(('Speech','Male speech, man speaking','Female speech, woman speaking',
    'Child speech, kid speaking','Conversation','Narration, monologue','Babbling',
    'Speech synthesizer','Whispering','Silence'))


def event_times(num_samples, sample_rate, hop=1.0):
    if num_samples <= 0 or sample_rate <= 0 or not math.isfinite(hop) or hop <= 0:
        raise ValueError('positive audio length, sampling rate and hop required')
    return np.arange(math.ceil(num_samples/sample_rate/hop),dtype=np.float64)*hop


def centered_window(wave, sample_rate, center, seconds=10.0):
    """Fixed-length window, zero-padded only outside the actual audio."""
    if wave.ndim != 1 or not math.isfinite(center) or center < 0 or seconds <= 0:
        raise ValueError('invalid waveform/window')
    length=round(seconds*sample_rate)
    if length <= 0:
        raise ValueError('empty window')
    left=round(center*sample_rate)-length//2
    right=left+length
    out=np.zeros(length,dtype=np.float32)
    a,b=max(0,left),min(len(wave),right)
    if a<b:
        out[a-left:b-left]=wave[a:b]
    return out


def event_cues(probabilities, embeddings, rms, labels, memory=5):
    """Positive class rises, prior-window cosine change and positive dB deltas."""
    p=np.asarray(probabilities,dtype=np.float32)
    e=np.asarray(embeddings,dtype=np.float32)
    rms=np.asarray(rms,dtype=np.float32)
    if p.ndim!=2 or e.ndim!=2 or len(p)!=len(e) or p.shape[1]!=len(labels) or rms.shape!=(len(p),) or memory<1:
        raise ValueError('incompatible feature tracks')
    if not all(np.isfinite(x).all() for x in (p,e,rms)) or (p<0).any() or (p>1).any() or (rms<0).any():
        raise ValueError('invalid feature values')
    norms=np.linalg.norm(e,axis=1,keepdims=True)
    if (norms<=1e-12).any():
        raise ValueError('zero CLAP embedding')
    normalized=e/norms
    onset=np.maximum(np.diff(p,axis=0,prepend=p[:1]),0)
    onset[:,[i for i,label in enumerate(labels) if label in EXCLUDED_LABELS]]=0
    novelty=np.zeros(len(p),dtype=np.float32)
    for i in range(1,len(p)):
        previous=normalized[max(0,i-memory):i].mean(axis=0)
        norm=np.linalg.norm(previous)
        novelty[i]=1-np.clip(np.dot(normalized[i],previous/norm),-1,1) if norm>1e-12 else 1
    db=20*np.log10(np.maximum(rms,1e-8))
    return {'class_onsets':onset,'clap_embeddings':normalized,'clap_change':novelty,
            'loudness_rise_db':np.maximum(np.diff(db,prepend=db[:1]),0)}
