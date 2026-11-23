"""CPU preflight of the actual native AD runner prompt and processor path."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_ad_vlm import prepare
from src.generation.native_audio import load_native_audio
from src.generation.omni import encode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--model-report', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    if args.report.exists():
        raise FileExistsError('preserve existing evidence')
    pin = json.loads(Path('configs/qwen25_omni_model.json').read_text())['models'][0]
    download = json.loads(args.model_report.read_text())
    model = download['models'][0]
    if not download['complete'] or (model['repo_id'], model['revision']) != (pin['repo_id'], pin['revision']):
        raise ValueError('pinned download mismatch')
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError('empty input')
    row = rows[0]
    dev = json.loads(Path('configs/reframed_splits.json').read_text())
    if row['video_id'] not in {item['video_id'] for item in dev['clips']}:
        raise ValueError('development input required')
    import torch
    from transformers import Qwen2_5OmniProcessor
    torch.set_num_threads(4)
    processor = Qwen2_5OmniProcessor.from_pretrained(model['snapshot'], local_files_only=True)
    images, prompt, budget = prepare(row, SimpleNamespace(
        max_side=768, audio_context='native', prior='P0', span=2., exemplar_texts=[]))
    waveform = load_native_audio(row['native_audio'])
    inputs = encode(processor, images, prompt, waveform)
    report = dict(scope='first development moment only; no model weights loaded or inference',
                  id=row['id'], input_moments=len(rows), checked_moments=1,
                  model_id=model['repo_id'], revision=model['revision'],
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  frame_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in row['frames']},
                  native_audio=row['native_audio'], prompt=prompt, word_limit=budget,
                  input_tokens=int(inputs['input_ids'].shape[-1]),
                  tensors={k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in inputs.items()})
    with args.report.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({'id': row['id'], 'input_tokens': report['input_tokens'], 'checked_moments': 1}))


if __name__ == '__main__':
    main()
