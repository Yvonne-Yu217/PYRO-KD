"""Frozen preparation must neither mix splits nor silently drop required IDs."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from src.datasets.frozen_membership import frozen_video_ids, egoschema_ids
from scripts.bridge_hpc_audio_caches import validate_speech_membership


def reframed():
    return {'dataset':'igorsterner/reframed','clips':[{'video_id':'dev'}],
            'test':{'video_ids':['test']}}


def ego():
    return {'pilot_q_uids':[f'p{i}' for i in range(10)],
            'heldout_q_uids':['h0','h1'],'num_heldout':2}


def test_explicit_frozen_memberships_and_wrong_schema():
    assert frozen_video_ids(reframed()) == ['dev']
    assert frozen_video_ids(reframed(), 'test') == ['test']
    long = {'dataset':'lmms-eval/Video-MME','official_split':'test','videos':[{'video_id':'604'}]}
    assert frozen_video_ids(long, 'long') == ['604']
    for config, subset in [(long,'dev'),(reframed(),'long')]:
        with pytest.raises(ValueError): frozen_video_ids(config,subset)


@pytest.mark.parametrize('bad', [['test','test'], ['dev'], ['../escape'], []])
def test_invalid_or_overlapping_test_membership_rejected(bad):
    config=reframed();config['test']['video_ids']=bad
    with pytest.raises(ValueError): frozen_video_ids(config,'test')


def test_bridge_requires_exact_speech_subset_and_coverage():
    old_dev={'complete':True,'split_sha256':'hash','videos':[{'video_id':'dev'}]}
    assert set(validate_speech_membership(old_dev,['dev'],'dev','hash'))=={'dev'}
    with pytest.raises(ValueError,match='subset'): validate_speech_membership(old_dev,['dev'],'test','hash')
    test={**old_dev,'subset':'test','videos':[{'video_id':'test'}]}
    assert set(validate_speech_membership(test,['test'],'test','hash'))=={'test'}
    for rows in [[],[{'video_id':'test'},{'video_id':'test'}],[{'video_id':'wrong'}]]:
        with pytest.raises(ValueError,match='membership'):
            validate_speech_membership({**test,'videos':rows},['test'],'test','hash')
    with pytest.raises(ValueError,match='split'): validate_speech_membership(test,['test'],'test','changed')


def test_egoschema_split_is_explicit_and_disjoint():
    cfg=ego()
    assert egoschema_ids(cfg)==cfg['pilot_q_uids']
    assert egoschema_ids(cfg,'heldout')==['h0','h1']
    cfg['heldout_q_uids'][0]='p0'
    with pytest.raises(ValueError,match='overlap'):egoschema_ids(cfg,'heldout')
    cfg=ego();cfg['heldout_q_uids']=['h0','h0']
    with pytest.raises(ValueError,match='duplicate'):egoschema_ids(cfg,'heldout')


def test_heldout_cli_emits_exact_label_free_membership(tmp_path):
    from scripts import build_egoschema_qa_inputs as cli
    config=tmp_path/'membership.json';config.write_text(json.dumps(ego()))
    budget=tmp_path/'budget.json';budget.write_text(json.dumps({'k_rule':{'rate_per_minute':5.}}))
    questions=tmp_path/'questions.json'
    questions.write_text(json.dumps([{'q_uid':ident,'question':'What is visible?',
        **{f'option {i}':f'choice {i}' for i in range(5)},'answer':'DO_NOT_COPY'}
        for ident in ['p0','h0','h1']]))
    output=tmp_path/'heldout.jsonl'
    manifest={'audio_path':None,'media_info':{'duration':12.}}
    frames=[SimpleNamespace(path=f'frame{i}.jpg',timestamp=float(i)) for i in range(4)]
    argv=['build','--questions',str(questions),'--pilot-config',str(config),
          '--budget-config',str(budget),'--cache-dir',str(tmp_path/'cache'),
          '--output',str(output),'--split','heldout','--selector','U']
    with patch('sys.argv',argv),patch.object(cli,'load_manifest',return_value=manifest), \
         patch.object(cli,'cache_is_current',return_value=True), \
         patch.object(cli,'cache_is_complete',return_value=True), \
         patch.object(cli,'frame_window',return_value=frames):
        cli.main()
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    assert [r['id'] for r in rows]==['h0','h1']
    assert all(r['split']=='heldout' and len(r['frames'])==4 for r in rows)
    assert 'DO_NOT_COPY' not in output.read_text()
    assert all('answer' not in r for r in rows)
