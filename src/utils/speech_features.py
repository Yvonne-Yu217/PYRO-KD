"""Whisper transcription restricted to VAD-positive samples, with absolute word times."""
import math
from .audio_features import merge_segments

SPEECH_FEATURES_VERSION = 1


def speech_windows(speech, num_samples, sample_rate=16000, max_seconds=28.0):
    """Disjoint half-open sample windows; never bridge a non-speech interval."""
    if sample_rate <= 0 or num_samples < 0 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('invalid audio/window dimensions')
    duration = num_samples / sample_rate
    for start, end in speech:
        if not all(math.isfinite(t) for t in (start,end)) or not 0 <= start <= end <= duration + 1/sample_rate:
            raise ValueError('VAD interval outside audio')
    width = int(max_seconds * sample_rate)
    if width < 1:
        raise ValueError('window must contain at least one sample')
    for start,end in merge_segments(speech):
        left = math.ceil(start * sample_rate)
        right = min(num_samples, math.floor(end * sample_rate))
        for offset in range(left,right,width):
            yield offset,min(right,offset+width)


def transcribe_gated(wave, speech, model, sample_rate=16000, max_seconds=28.0):
    """Transcribe separate VAD slices; retain out-of-gate predictions only as diagnostics."""
    if getattr(wave,'ndim',1) != 1:
        raise ValueError('expected mono waveform')
    words, rejected, windows = [], [], []
    for left,right in speech_windows(speech,len(wave),sample_rate,max_seconds):
        start,end = left/sample_rate,right/sample_rate
        segments,_ = model.transcribe(wave[left:right],language='en',word_timestamps=True,
            beam_size=5,temperature=0.0,condition_on_previous_text=False,vad_filter=False)
        count = 0
        for segment in segments:
            for word in segment.words or []:
                begin,finish = start+float(word.start),start+float(word.end)
                record = {'start':begin,'end':finish,'text':word.word,'probability':float(word.probability)}
                # Do not silently clamp or move hallucinated/padded word timestamps.
                if not all(math.isfinite(v) for v in (begin,finish,record['probability'])) or not start <= begin <= finish <= end:
                    rejected.append({**record,'gate':[start,end],'reason':'invalid_or_outside_vad_window'})
                    continue
                words.append(record); count += 1
        windows.append({'start':start,'end':end,'samples':right-left,'words':count})
    return {'words':words,'rejected_words':rejected,'windows':windows,
            'speech_samples':sum(w['samples'] for w in windows)}
