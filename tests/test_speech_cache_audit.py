import copy
import pytest
from scripts.verify_speech_cache import check_words


def features():
    return {'windows':[{'start':1.,'end':2.,'samples':16000,'words':1,
                        'detected_language':'es','language_probability':.8}],
            'words':[{'start':1.1,'end':1.5,'text':' hola','probability':.9}],
            'speech_samples':16000,'rejected_words':[]}


def test_counts_and_language_without_transcript_output():
    result=check_words(features(),[(1.,2.)],48000,None)
    assert result['detected_language_windows']=={'es':1} and result['words']==1
    assert 'hola' not in str(result)


@pytest.mark.parametrize('kind',['window','word','count','samples','language'])
def test_bad_coverage_fails(kind):
    f=copy.deepcopy(features())
    if kind=='window': f['windows'][0]['start']=0.
    if kind=='word': f['words'][0]['end']=2.1
    if kind=='count': f['windows'][0]['words']=0
    if kind=='samples': f['speech_samples']=15000
    if kind=='language': f['windows'][0]['language_probability']=float('nan')
    with pytest.raises(ValueError): check_words(f,[(1.,2.)],48000,None)
