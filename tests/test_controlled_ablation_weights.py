from src.experiments.matrix import ablation_selections


def config():
    return {'selectors': {'A_cue': {'base':'A_cue','options':{'weights':{'nse':.7,'sc':.8,'gf':.9}}},
                          'AV': {'base':'AV','options':{'weights':{'nse':.7,'sc':.8,'gf':.9},'memory_blend':.25}}}}


def test_event_source_ablation_preserves_all_frozen_weights():
    runs=ablation_selections('3',config())
    assert runs[0].spec['options']==runs[1].spec['options']
    assert runs[0].spec['options']['weights']['gf']==.9
    assert runs[0].spec['cue_set']!=runs[1].spec['cue_set']


def test_hierarchy_changes_budget_not_score_weights():
    flat,hier=ablation_selections('7',config())
    assert flat.spec['options']==hier.spec['options']
    assert flat.spec['options']['weights']['sc']==.8
    assert flat.spec['budget']=='flat' and hier.spec['budget']=='hierarchical'


def test_component_ablation_overrides_only_target_weight():
    speech,events,both=ablation_selections('1',config())
    assert speech.spec['options']['weights']['nse']==0
    assert speech.spec['options']['weights']['sc']==.8
    assert events.spec['options']['weights']['sc']==0
    assert events.spec['options']['weights']['nse']==.7
    assert all(r.spec['options']['weights']['gf']==.9 for r in (speech,events,both))
    assert config()['selectors']['A_cue']['options']['weights']['nse']==.7
