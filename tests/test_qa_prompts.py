import pytest
from src.generation.qa import build_qa_prompt,parse_answer


def row():
    return {'question':'What happens?','options':['One','Two','Three','Four','Five'],
            'frames':['a.jpg','b.jpg'],'frame_times':[0.,2.],'audio_context':'none'}


def test_prompt_and_parser_keep_fixed_option_order():
    prompt=build_qa_prompt(row())
    assert 'A. One' in prompt and 'E. Five' in prompt and '0.000, 2.000' in prompt
    assert parse_answer(' B. ',5)==1
    assert parse_answer('e',4) is None


@pytest.mark.parametrize('text',['A or B','The answer is C','B because it fits','',None,'0','AB'])
def test_ambiguous_or_freeform_answer_not_salvaged(text):
    assert parse_answer(text,5) is None


def test_nested_labels_rejected():
    data=row();data['metadata']={'answer':2}
    with pytest.raises(ValueError,match='labels'): build_qa_prompt(data)


def test_timestamp_alignment_required():
    data=row();data['frame_times']=[2,0]
    with pytest.raises(ValueError,match='chronological'): build_qa_prompt(data)


def test_runner_dry_run_never_loads_model_or_writes_predictions(tmp_path,monkeypatch):
    import json,sys
    from PIL import Image
    import scripts.run_qa_vlm as runner
    image=tmp_path/'frame.jpg';Image.new('RGB',(8,8)).save(image)
    data={**row(),'id':'q','frames':[str(image)],'frame_times':[0.]}
    source=tmp_path/'input.jsonl';source.write_text(json.dumps(data)+'\n')
    output=tmp_path/'predictions.jsonl'
    monkeypatch.setattr(sys,'argv',['run_qa_vlm','--input',str(source),'--output',str(output),'--dry-run'])
    def forbidden(*args,**kwargs): raise AssertionError('dry-run loaded weights')
    monkeypatch.setattr(runner,'load_model',forbidden)
    runner.main()
    assert not output.exists()


def test_runner_records_choice_and_resumes_without_loading(tmp_path,monkeypatch):
    import json,sys
    from PIL import Image
    import scripts.run_qa_vlm as runner
    image=tmp_path/'frame.jpg';Image.new('RGB',(8,8)).save(image)
    data={**row(),'id':'q','frames':[str(image)],'frame_times':[0.]}
    source=tmp_path/'input.jsonl';source.write_text(json.dumps(data)+'\n')
    output=tmp_path/'predictions.jsonl'
    monkeypatch.setattr(sys,'argv',['run_qa_vlm','--input',str(source),'--output',str(output)])
    import torch
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(runner,'load_model',lambda *args:(None,None,'unit-test-double'))
    monkeypatch.setattr(runner,'generate_batch',lambda *args,**kwargs:(['B'],['B'],[1]))
    with pytest.raises(SystemExit) as exc: runner.main()
    assert exc.value.code==0
    prediction=json.loads(output.read_text())
    assert prediction['predicted_index']==1 and 'error' not in prediction
    def forbidden(*args): raise AssertionError('completed resume loaded weights')
    monkeypatch.setattr(runner,'load_model',forbidden)
    runner.main()
    assert len(output.read_text().splitlines())==1


def test_runner_recovers_partial_tail_before_resuming(tmp_path,monkeypatch):
    import json,sys
    from PIL import Image
    import scripts.run_qa_vlm as runner
    image=tmp_path/'frame.jpg';Image.new('RGB',(8,8)).save(image)
    inputs=[{**row(),'id':ident,'frames':[str(image)],'frame_times':[0.]} for ident in ('first','second')]
    source=tmp_path/'input.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in inputs))
    output=tmp_path/'predictions.jsonl'
    argv=['run_qa_vlm','--input',str(source),'--output',str(output)]
    monkeypatch.setattr(sys,'argv',argv+['--limit','1'])
    import torch
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 0)
    monkeypatch.setattr(runner,'load_model',lambda *args:(None,None,'unit-test-double'))
    monkeypatch.setattr(runner,'generate_batch',lambda *args,**kwargs:(['B'],['B'],[1]))
    with pytest.raises(SystemExit) as exc: runner.main()
    assert exc.value.code==0
    with output.open('ab') as handle: handle.write(b'{"id":"second","output":"\xe4\xb8')
    interrupted=output.read_bytes()
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(SystemExit) as exc: runner.main()
    assert exc.value.code==0
    assert [r['id'] for r in map(json.loads,output.read_text().splitlines())]==['first','second']
    backups=list(tmp_path.glob('*.before-recovery-*.bin'))
    assert len(backups)==1 and backups[0].read_bytes()==interrupted
