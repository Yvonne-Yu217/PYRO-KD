"""Label-free multiple-choice QA prompts and strict answer parsing."""
import math
import re

FORBIDDEN={'answer','answers','label','labels','correct_answer','answer_idx','target'}


def reject_labels(value):
    if isinstance(value,dict):
        if FORBIDDEN & value.keys(): raise ValueError('answer labels must not enter generation inputs')
        for child in value.values(): reject_labels(child)
    elif isinstance(value,list):
        for child in value: reject_labels(child)


def build_qa_prompt(row):
    reject_labels(row)
    question,options=row['question'],row['options']
    if not isinstance(question,str) or not question.strip(): raise ValueError('nonempty question required')
    if not isinstance(options,list) or len(options) not in (4,5) or any(not isinstance(x,str) or not x.strip() for x in options):
        raise ValueError('four or five nonempty options required')
    times=row['frame_times']
    if not times or len(times)!=len(row['frames']) or any(not math.isfinite(t) or t<0 for t in times):
        raise ValueError('finite aligned frame timestamps required')
    if any(b<a for a,b in zip(times,times[1:])): raise ValueError('frames must be chronological')
    letters='ABCDE'[:len(options)]
    prompt=('Answer the multiple-choice question using the supplied video frames. '
            'Frames are in chronological order at these times in seconds: '
            +', '.join(f'{t:.3f}' for t in times)+'.\n')
    if row.get('audio_context','none')=='timeline':
        from src.generation.audio_timeline import render_timeline
        prompt+='Automatic audio timeline:\n'+render_timeline(row['audio_timeline'])+'\n'
    elif row.get('audio_context','none')!='none': raise ValueError('unsupported QA audio context')
    prompt+='Question: '+question+'\n'
    prompt+='\n'.join(f'{letter}. {option}' for letter,option in zip(letters,options))
    return prompt+'\nReturn exactly one letter ('+', '.join(letters)+') and no explanation.'


def parse_answer(text, num_options):
    """Accept only a standalone option letter (optional terminal period/parenthesis).

    Never search free-form reasoning for a favorable option or use the label to
    resolve ambiguity. Invalid answers remain invalid for accuracy accounting.
    """
    if num_options not in (4,5): raise ValueError('expected four or five options')
    if not isinstance(text,str): return None
    match=re.fullmatch(r'\s*([A-Ea-e])[.)]?\s*',text)
    if not match: return None
    index=ord(match[1].upper())-ord('A')
    return index if index<num_options else None
