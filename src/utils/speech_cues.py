"""Auditable ASR-derived speech cues; no AD references or visual inputs."""
import math
import re

DEICTIC_PHRASES = ('look at this', 'look at that', 'over there', 'watch this', 'watch that')


def speech_chunks(words, windows, max_words=24):
    """Group words within their own VAD transcription window, ending at punctuation."""
    if isinstance(max_words,bool) or not isinstance(max_words,int) or max_words < 1:
        raise ValueError('max_words must be a positive integer')
    chunks=[]
    previous=-1.0
    for window in windows:
        a,b=window['start'],window['end']
        if not all(math.isfinite(v) for v in (a,b)) or not 0<=a<b or a<previous:
            raise ValueError('windows must be finite, ordered and disjoint')
        previous=b
    groups=[[] for _ in windows]
    previous=-1.0
    for word in words:
        a,b=word['start'],word['end']
        if not all(math.isfinite(v) for v in (a,b)) or not 0<=a<=b or a<previous:
            raise ValueError('words must have finite ordered timestamps')
        previous=a
        if not isinstance(word['text'],str) or not word['text'].strip():
            raise ValueError('words require nonempty text')
        gate=next((i for i,w in enumerate(windows) if w['start']<=a<=b<=w['end']),None)
        if gate is None: raise ValueError('word lies outside VAD transcription windows')
        groups[gate].append(word)
    for gate,group in enumerate(groups):
        chunk=[]
        def flush():
            if chunk:
                chunks.append({'start':chunk[0]['start'],'end':max(w['end'] for w in chunk),
                               'text':' '.join(w['text'].strip() for w in chunk),'gate':gate})
        for word in group:
            chunk.append(word)
            if len(chunk)>=max_words or re.search(r'[.!?]["\u201d\u2019\']?$',word['text'].strip()):
                flush();chunk=[]
        flush()
    return chunks


def cue_tracks(chunks, embeddings, gaps, *, min_silence, silence_scale):
    """Keep cue components separate for dev tuning and speech-only ablations.

    Semantic shift is adjacent chunk cosine distance / 2; first chunk is zero.
    Phrase detection is lexical and may be figurative; it is not a human label.
    Long silence uses already validated VAD gaps, not absence of ASR words.
    """
    import numpy as np
    if not all(math.isfinite(x) and x>0 for x in (min_silence,silence_scale)):
        raise ValueError('silence settings must be finite and positive')
    array=np.asarray(embeddings,dtype=np.float64)
    if array.ndim!=2 or len(array)!=len(chunks) or array.shape[1]<1 or not np.isfinite(array).all():
        raise ValueError('finite aligned text embeddings required')
    norms=np.linalg.norm(array,axis=1)
    if np.any(norms<=0): raise ValueError('text embeddings must be nonzero')
    unit=array/norms[:,None]
    shifts=np.zeros(len(chunks))
    if len(chunks)>1:
        shifts[1:]=(1-np.clip(np.sum(unit[1:]*unit[:-1],axis=1),-1,1))/2
    speech=[]
    for i,chunk in enumerate(chunks):
        tokens=' '.join(re.findall(r"[a-z]+",chunk['text'].lower()))
        matches=[p for p in DEICTIC_PHRASES if f' {p} ' in f' {tokens} ']
        speech.append({**chunk,'time':chunk['start'],'semantic_shift':float(shifts[i]),
                       'deictic':float(bool(matches)),'matched_phrases':matches})
    silence=[]
    for start,end in gaps:
        if not all(math.isfinite(x) for x in (start,end)) or not 0<=start<end:
            raise ValueError('invalid silence interval')
        if end-start>=min_silence:
            silence.append({'time':(start+end)/2,'start':start,'end':end,
                            'long_silence':min(1.0,(end-start)/silence_scale)})
    return {'speech':speech,'silence':silence}
