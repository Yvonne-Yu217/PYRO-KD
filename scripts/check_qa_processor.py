"""CPU-only QA prompt encoding with the pinned local processor, no VLM weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_qa_vlm import prepare
from src.generation.hf_vlm import build_conversation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--limit',type=int,default=1)
    p.add_argument('--max-side',type=int,default=768)
    p.add_argument('--max-new-tokens',type=int,default=16)
    args=p.parse_args()
    if min(args.limit,args.max_side,args.max_new_tokens)<1:
        raise ValueError('positive resource bounds required')
    if args.report.exists():raise FileExistsError('preserve previous processor audit')
    if not args.model_path.is_dir():raise ValueError('pinned local snapshot required')
    raw=args.input.read_bytes()
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('unique nonempty QA inputs required')
    config_path=args.model_path/'config.json'
    config=json.loads(config_path.read_text())
    context=config.get('text_config',config).get('max_position_embeddings')
    if isinstance(context,bool) or not isinstance(context,int) or context<1:
        raise ValueError('model config must declare max_position_embeddings')
    import torch
    from transformers import AutoProcessor
    torch.set_num_threads(4)
    processor=AutoProcessor.from_pretrained(str(args.model_path),local_files_only=True)
    records=[]
    for row in rows[:args.limit]:
        images,prompt=prepare(row,args.max_side)
        encoded=processor.apply_chat_template([build_conversation(images,prompt)],
            tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,padding=True)
        tokens=int(encoded['input_ids'].shape[-1])
        records.append(dict(id=row['id'],frames=len(images),input_tokens=tokens,
            requested_new_tokens=args.max_new_tokens,configured_context=context,
            within_configured_context=tokens+args.max_new_tokens<=context,
            encoded_tensors={key:{'shape':list(value.shape),'dtype':str(value.dtype),
                                  'bytes':value.numel()*value.element_size()}
                             for key,value in encoded.items() if hasattr(value,'shape')},
            frame_sha256={path:hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in row['frames']}))
        print(json.dumps({k:v for k,v in records[-1].items() if k not in ('frame_sha256','encoded_tensors')}),flush=True)
        del encoded,images
    report=dict(complete=all(r['within_configured_context'] for r in records),
        questions_checked=len(records),input_questions=len(rows),full_input_coverage=len(records)==len(rows),
        model_path=str(args.model_path.resolve()),model_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(raw).hexdigest(),max_side=args.max_side,model_weights_loaded=False,
        scope='processor encoding and declared context only; not GPU memory feasibility or QA accuracy',questions=records)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    raise SystemExit(0 if report['complete'] else 1)


if __name__=='__main__':main()
