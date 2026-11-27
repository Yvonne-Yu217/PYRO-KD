import pytest
from scripts.build_long_qa_inputs import normalize_options,evenly,QUESTION_COLUMNS

def test_option_order_is_preserved_and_validated():
    assert normalize_options(['A. first','B. second','C. third','D. fourth'])==['first','second','third','fourth']
    with pytest.raises(ValueError,match='order'):
        normalize_options(['B. first','A. second','C. third','D. fourth'])
    assert 'answer' not in QUESTION_COLUMNS and 'question' in QUESTION_COLUMNS

def test_timeline_budget_covers_both_ends():
    rows=list(range(101));chosen=evenly(rows,20)
    assert len(chosen)==len(set(chosen))==20 and chosen[0]==0 and chosen[-1]==100
    assert evenly(rows[:3],20)==rows[:3]


def test_audio_only_builder_does_not_read_visuals(monkeypatch, tmp_path):
    import scripts.build_long_qa_inputs as b
    import src.audio.cues_io as cio
    import src.utils.audio_features as af
    import src.selection.registry as registry
    import types
    vad = {'speech': [], 'gaps': []}
    cues = {'asr': {'source': 'whisper-vad-gated'}, 'streams': {x:'ok' for x in ['asr','tagger','clap','acoustic']}}
    monkeypatch.setattr(af, 'load_current_features', lambda *a: (vad,None))
    monkeypatch.setattr(af, 'resolve_audio_path', lambda *a: tmp_path/'audio.wav')
    monkeypatch.setattr(cio, 'load_cues_for_selection', lambda *a: (cues,None))
    monkeypatch.setattr(b, 'sha', lambda *a: 'verified-test-digest')
    monkeypatch.setattr(b, 'load_validated_features', lambda *a: pytest.fail('A_cue read visual input'))
    captured = {}
    def select(spec, context, k):
        captured.update(spec=spec,context=context,k=k)
        return types.SimpleNamespace(status='ok', times=[2.,5.], diagnostics={'shortfall':1})
    monkeypatch.setattr(registry, 'run_selector', select)
    cfg={'selectors':{'A_cue':{'base':'A_cue','options':{'weights':{'nse':.5,'sc':.5,'gf':.5}}}}}
    times, provenance=b.audio_selection(tmp_path,tmp_path,{'media_info':{'duration':60}},'A_cue','hierarchical',cfg,3)
    assert times==[2.,5.] and provenance['diagnostics']['shortfall']==1
    assert captured['context'].visual_features is None and captured['spec']['budget']=='hierarchical'
    assert captured['spec']['options']['weights']['nse']==.5
    cues['asr']['source']='whisper-ungated'
    with pytest.raises(ValueError,match='VAD-gated'):
        b.audio_selection(tmp_path,tmp_path,{'media_info':{'duration':60}},'A_cue','flat',cfg,3)
