import copy
import pytest
from scripts.derive_gated_whisper_cues import derive
from src.selection.cues import build_cue_bundle

def record(text):
    return {'version':1,'duration':20.,'config':{},'config_hash':'source','audio_sha256':'wav','vad_hash':'vad',
            'streams':{'asr':'ok','tagger':'ok','clap':'ok','acoustic':'ok','semantic':'ok'},
            'asr':{'source':'whisper-vad-gated','segments':[{'start':1.,'end':3.,'text':text}]},
            'events':[{'time':4.,'label':'Music','superclass':'music','strength':.9}],
            'tagger':None,'clap':{'times':[1.],'change':[.7]},
            'acoustic':{'times':[1.],'loudness_jump':[2.],'onset_strength':[.1]},'speech_cues':[]}

def test_only_event_source_changes_and_parent_is_immutable(tmp_path):
    original=record('[applause]');before=copy.deepcopy(original)
    result=derive(original,tmp_path/'parent.json','digest')
    assert original==before and result['asr']==before['asr']
    assert result['acoustic']==before['acoustic'] and result['speech_cues']==before['speech_cues']
    assert result['clap'] is None and result['tagger'] is None
    assert [e['label'] for e in result['events']]==['Applause']
    assert result['bridge_sources'][str(tmp_path/'parent.json')]=='digest'

def test_zero_emitted_tags_is_valid_zero_track_not_missing(tmp_path):
    result=derive(record('normal spoken words'),tmp_path/'parent.json','digest')
    bundle=build_cue_bundle(20.,{'speech':[],'gaps':[]},result)
    assert result['events']==[] and result['ablation3']['event_count']==0
    assert bundle.nse['music']==[0.]*len(bundle.grid)
    assert 'tagger' not in bundle.missing and 'clap_change' not in bundle.nse

def test_ungated_source_refused(tmp_path):
    bad=record('[music]');bad['asr']['source']='whisper-ungated'
    with pytest.raises(ValueError,match='VAD-gated'):derive(bad,tmp_path/'parent.json','digest')
