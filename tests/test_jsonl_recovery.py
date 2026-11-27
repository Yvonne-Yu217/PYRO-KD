import json
import pytest
from src.generation.jsonl_recovery import read_resume_rows,apply_tail_recovery


@pytest.mark.parametrize('tail',[b'{"id":"b',b'{"text":"\xe4\xb8'])
def test_interrupted_tail_preserved_and_next_append_readable(tmp_path,tail):
    path=tmp_path/'predictions.jsonl'
    original=b'{"id":"a"}\n'+tail;path.write_bytes(original)
    rows,recovery=read_resume_rows(path)
    assert rows==[{'id':'a'}] and path.read_bytes()==original
    event=apply_tail_recovery(path,recovery)
    assert __import__('pathlib').Path(event['backup']).read_bytes()==original
    with path.open('a') as handle: handle.write('{"id":"b"}\n')
    assert read_resume_rows(path)==([{'id':'a'},{'id':'b'}],None)


def test_valid_last_record_is_retained_and_delimited(tmp_path):
    path=tmp_path/'p.jsonl';path.write_bytes(b'{"id":"a"}')
    rows,recovery=read_resume_rows(path)
    assert rows==[{'id':'a'}]
    apply_tail_recovery(path,recovery)
    assert path.read_bytes()==b'{"id":"a"}\n'
    assert read_resume_rows(path)[1] is None


@pytest.mark.parametrize('raw',[b'{broken}\n',b'{broken}\n{"id":"b"}',b'[]\n'])
def test_complete_corruption_not_silently_removed(tmp_path,raw):
    path=tmp_path/'p.jsonl';path.write_bytes(raw)
    with pytest.raises(ValueError): read_resume_rows(path)
    assert path.read_bytes()==raw and len(list(tmp_path.iterdir()))==1


def test_changed_output_refuses_repair(tmp_path):
    path=tmp_path/'p.jsonl';path.write_bytes(b'{')
    _,recovery=read_resume_rows(path)
    path.write_bytes(b'{"id":"new"}\n')
    with pytest.raises(ValueError,match='changed'): apply_tail_recovery(path,recovery)
