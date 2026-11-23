"""Explicit waveform context for native-audio AD; no transcript substitution."""
import hashlib
import math
from pathlib import Path


def span_for_center(center, frames, sample_rate, context_seconds=15.):
    if (not math.isfinite(center) or center<0 or not math.isfinite(context_seconds) or context_seconds<=0
            or isinstance(frames,bool) or not isinstance(frames,int) or frames<=0
            or sample_rate!=16000 or center>frames/sample_rate):
        raise ValueError('valid center and nonempty mono16k audio required')
    start=max(0,math.floor((center-context_seconds)*sample_rate))
    end=min(frames,math.ceil((center+context_seconds)*sample_rate))
    if end<=start:raise ValueError('empty native audio span')
    return start,end


def load_native_audio(record):
    """Read the exact recorded samples after verifying the entire source WAV."""
    import numpy as np
    import soundfile as sf
    path=Path(record['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['source_sha256']:
        raise ValueError('native audio source hash mismatch')
    with sf.SoundFile(path) as f:
        if f.samplerate!=16000 or record['sample_rate']!=f.samplerate or f.channels!=1 or len(f)!=record['source_samples']:
            raise ValueError('native audio format or duration changed')
        start,end=span_for_center(record['frame_center_time'],len(f),f.samplerate,record['context_seconds'])
        if (start,end)!=(record['start_sample'],record['end_sample']):
            raise ValueError('native audio boundaries differ from declared context')
        if (record['start_seconds'],record['end_seconds'])!=(start/f.samplerate,end/f.samplerate):
            raise ValueError('native audio seconds differ from sample boundaries')
        f.seek(start);wave=f.read(end-start,dtype='float32')
    if len(wave)!=end-start or not np.isfinite(wave).all():raise ValueError('invalid native audio samples')
    if hashlib.sha256(wave.tobytes()).hexdigest()!=record['samples_sha256']:
        raise ValueError('native audio slice hash mismatch')
    return wave
