"""Run frozen visual multiple-choice QA with label-free inputs and resume guards.

--dry-run validates every image/prompt without loading model weights or creating
predictions. Real inference retains invalid answers and generation failures.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.generation.qa import build_qa_prompt,parse_answer
from src.generation.hf_vlm import load_model,generate_batch,resolve_model
from src.generation.jsonl_recovery import read_resume_rows,apply_tail_recovery
from src.generation.provenance import capture_provenance,check_resume_fingerprint
from src.generation.outputs import settled_ids
from scripts.run_ad_vlm import resize_max_side,finalize_generation


def prepare(row,max_side):
    from PIL import Image
    prompt=build_qa_prompt(row)
    images=[]
    for path in row['frames']:
        with Image.open(path) as image:
            images.append(resize_max_side(image.convert('RGB'),max_side))
    return images,prompt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='Qwen/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--quant',choices=('none','4bit','8bit'),default='none')
    p.add_argument('--attn-impl',default='sdpa')
    p.add_argument('--max-side',type=int,default=768)
    p.add_argument('--max-new-tokens',type=int,default=16)
    p.add_argument('--limit',type=int)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    if args.max_new_tokens<1 or args.max_side<1 or (args.limit is not None and args.limit<1):
        raise ValueError('positive resource limits required')
    rows=[json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows})!=len(rows): raise ValueError('nonempty unique questions required')
    for row in rows: build_qa_prompt(row)
    model_id=resolve_model(args.model)
    settings={'task':'multiple_choice_qa','model_id':model_id,'quant':args.quant,
              'attn_impl':args.attn_impl,'max_side':args.max_side,'temperature':0.,'top_p':.9,
              'parser':'strict_standalone_letter_v1'}
    provenance=capture_provenance(Path(__file__).resolve().parents[1],args.input,rows,settings)
    prior,recovery=read_resume_rows(args.output)
    check_resume_fingerprint(prior,provenance['run_fingerprint'])
    if args.limit: rows=rows[:args.limit]
    done,_=settled_ids(prior,args.max_new_tokens,0.)
    todo=[r for r in rows if r['id'] not in done]
    if args.dry_run:
        for row in todo:
            images,prompt=prepare(row,args.max_side)
            print(json.dumps({'id':row['id'],'frames':len(images),'options':len(row['options']),
                              'prompt_characters':len(prompt),'dry_run':True}))
        print(json.dumps({'complete':True,'questions_checked':len(todo),'model_loaded':False}))
        return
    if recovery is not None:
        if not prior:
            sidecar=json.loads(args.output.with_suffix(args.output.suffix+'.provenance.json').read_text())
            if sidecar['run_fingerprint']!=provenance['run_fingerprint']:
                raise ValueError('interrupted output provenance mismatch')
        print(json.dumps({'output_recovery':apply_tail_recovery(args.output,recovery)}),flush=True)
    if not todo:
        print('No pending questions; no model loaded');return
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix(args.output.suffix+'.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    import torch
    import transformers
    started=time.monotonic()
    processor,model,model_class=load_model(model_id,args.attn_impl,args.quant)
    load_seconds=time.monotonic()-started
    for device in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(device)
    errors=0
    with args.output.open('a') as handle:
        for row in todo:
            base={k:row.get(k) for k in ('id','video_id','dataset','split','selector','audio_context',
                  'k_requested','k_realized','k_shortfall','selected_times','frame_times','protocol_status','source_sha256',
                  'selection_condition','visual_features_sha256')}
            base.update(model_id=model_id,temperature=0.,run_fingerprint=provenance['run_fingerprint'])
            try:
                images,prompt=prepare(row,args.max_side)
                started=time.monotonic()
                texts,raws,tokens=generate_batch(processor,model,[images],[prompt],
                    max_new_tokens=args.max_new_tokens,temperature=0.,top_p=.9)
                result=finalize_generation(texts[0],raws[0],tokens[0],1,time.monotonic()-started,args.max_new_tokens)
                choice=parse_answer(texts[0],len(row['options'])) if 'error' not in result else None
                if choice is None and 'error' not in result: result['error']='invalid_answer: expected a single option letter'
                result.update(predicted_index=choice,prompt=prompt,num_options=len(row['options']))
            except Exception as exc:
                result={'error':repr(exc),'predicted_index':None}
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            errors+='error' in result
            handle.write(json.dumps({**base,**result},allow_nan=False)+'\n');handle.flush()
            print(json.dumps({'id':row['id'],'success':'error' not in result}),flush=True)
    meta={'model_class':model_class,'load_seconds':load_seconds,'questions_attempted':len(todo),'errors':errors,
          'torch':torch.__version__,'transformers':transformers.__version__,'provenance':provenance,
          'peak_gpu_memory_bytes':sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))}
    with args.output.with_suffix(args.output.suffix+'.attempts.jsonl').open('a') as handle:
        handle.write(json.dumps(meta,allow_nan=False)+'\n')
    raise SystemExit(1 if errors else 0)


if __name__=='__main__':main()
