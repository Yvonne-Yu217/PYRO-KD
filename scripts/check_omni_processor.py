"""Encode real development audio plus frames with pinned Omni assets on CPU."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.media import load_manifest,cache_is_current,frame_window
from src.utils.audio_features import resolve_audio_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--video-id',default='CxWby8RuUwU6')
    p.add_argument('--model-report',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if args.report.exists():raise FileExistsError('preserve previous check')
    dev=json.loads(Path('configs/reframed_splits.json').read_text())
    if args.video_id not in {r['video_id'] for r in dev['clips']}:
        raise ValueError('development clip required')
    pin=json.loads(Path('configs/qwen25_omni_processor.json').read_text())['models'][0]
    downloaded=json.loads(args.model_report.read_text())
    model=downloaded['models'][0]
    if (not downloaded['complete'] or model['repo_id']!=pin['repo_id'] or model['revision']!=pin['revision']):
        raise ValueError('pinned processor download mismatch')
    root=Path(model['snapshot'])
    for item in pin['files']:
        if (root/item['name']).stat().st_size!=item['bytes']:raise ValueError('processor file changed')
    manifest=load_manifest(args.cache_dir/args.video_id)
    if not cache_is_current(manifest):raise ValueError('stale media')
    import soundfile as sf
    import torch
    from PIL import Image
    from transformers import Qwen2_5OmniProcessor,Qwen2_5OmniForConditionalGeneration
    torch.set_num_threads(4)
    audio_path=resolve_audio_path(manifest,args.cache_dir/args.video_id)
    audio,sr=sf.read(audio_path,dtype='float32',frames=30*16000)
    if sr!=16000 or audio.ndim!=1 or len(audio)!=30*16000:
        raise ValueError('real 30s mono16k development audio required')
    frames=frame_window(manifest,15.,4,2.)
    if len(frames)!=4:raise ValueError('four aligned visual frames required')
    images=[]
    for frame in frames:
        with Image.open(frame.path) as image:images.append(image.convert('RGB'))
    processor=Qwen2_5OmniProcessor.from_pretrained(str(root),local_files_only=True)
    conversation=[{'role':'user','content':[
        *[{'type':'image','image':image} for image in images],
        {'type':'audio','audio':audio},
        {'type':'text','text':'The frames show the scene around 15 seconds. The audio covers 0 to 30 seconds. Describe what is visible using audio as context.'}]}]
    text=processor.apply_chat_template(conversation,add_generation_prompt=True,tokenize=False)
    encoded=processor(text=[text],audio=[audio],images=images,return_tensors='pt',padding=True,sampling_rate=sr)
    if 'input_features' not in encoded or 'pixel_values' not in encoded:
        raise ValueError('processor did not encode both audio and images')
    report=dict(complete=True,video_id=args.video_id,model_id=pin['repo_id'],revision=pin['revision'],
        processor_class=type(processor).__name__,model_class_importable=Qwen2_5OmniForConditionalGeneration.__name__,
        generate_declares_return_audio='return_audio' in inspect.signature(Qwen2_5OmniForConditionalGeneration.generate).parameters,
        generate_implementation=Qwen2_5OmniForConditionalGeneration.generate.__qualname__,
        model_weights_loaded=False,audio_seconds=30,sample_rate=sr,frame_times=[f.timestamp for f in frames],
        audio_slice_sha256=hashlib.sha256(audio.tobytes()).hexdigest(),
        audio_file_sha256=hashlib.sha256(Path(audio_path).read_bytes()).hexdigest(),
        frame_sha256={f.path:hashlib.sha256(Path(f.path).read_bytes()).hexdigest() for f in frames},
        processor_files_sha256={f['name']:hashlib.sha256((root/f['name']).read_bytes()).hexdigest() for f in pin['files']},
        input_tokens=int(encoded['input_ids'].shape[-1]),
        tensors={key:{'shape':list(value.shape),'dtype':str(value.dtype)} for key,value in encoded.items()},
        scope='real development multimodal processor encoding only; not inference, accuracy or GPU memory validation')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k not in ('frame_sha256','processor_files_sha256','tensors')}))


if __name__=='__main__':main()
