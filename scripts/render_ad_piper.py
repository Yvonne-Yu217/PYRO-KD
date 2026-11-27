"""Render valid AD predictions with pinned local Piper; retain failed-input coverage."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import wave
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.generation.outputs import split_rows,is_failure


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--voice-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/piper_tts.json'))
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError('preserve previous TTS artifacts')
    config=json.loads(args.config.read_text())
    if importlib.metadata.version('piper-tts')!=config['engine_version']:raise ValueError('engine version differs from pin')
    for item in config['files']:
        path=args.voice_dir/item['name']
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:raise ValueError('voice asset changed')
    rows,history=split_rows([json.loads(l) for l in args.input.read_text().splitlines() if l.strip()])
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('unique nonempty predictions required')
    failed=[r for r in rows if is_failure(r)]
    valid=[r for r in rows if not is_failure(r)]
    if not valid or any(not r['output'].strip() for r in valid):raise ValueError('valid nonempty descriptions required')
    import onnxruntime as ort
    import numpy as np
    from piper import PiperVoice,SynthesisConfig
    from piper.config import PiperConfig
    model=args.voice_dir/(config['voice']+'.onnx')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    voice=PiperVoice(session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider']),
                     config=PiperConfig.from_dict(json.loads(Path(str(model)+'.json').read_text())))
    settings=SynthesisConfig(**config['synthesis'])
    args.output_dir.mkdir(parents=True)
    output=[]
    for row in valid:
        path=args.output_dir/(hashlib.sha256(row['id'].encode()).hexdigest()+'.wav')
        with wave.open(str(path),'wb') as wav:voice.synthesize_wav(row['output'],wav,syn_config=settings)
        with wave.open(str(path),'rb') as wav:
            count=wav.getnframes();rate=wav.getframerate()
            if wav.getnchannels()!=1 or wav.getsampwidth()!=2 or rate!=config['sample_rate'] or count<=0:raise ValueError('invalid rendered WAV')
            pcm=np.frombuffer(wav.readframes(count),dtype='<i2')
            if len(pcm)!=count or not np.any(pcm):raise ValueError('empty or incomplete rendered speech')
        output.append({**row,'tts_audio':str(path.resolve()),'tts_audio_sha256':sha(path),
                       'tts_text_sha256':hashlib.sha256(row['output'].encode()).hexdigest(),
                       'tts_playback_speed':1.0,'tts_duration_seconds':count/rate,
                       'tts_config_sha256':sha(args.config)})
    target=args.output_dir/'rendered.jsonl'
    with target.open('x') as f:
        for row in output:f.write(json.dumps(row,allow_nan=False)+'\n')
    report=dict(complete=True,input_sha256=sha(args.input),config_sha256=sha(args.config),
                source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                implementation_sha256=sha(__file__),input_rows=len(rows),rendered=len(output),
                failed_ids=[r['id'] for r in failed],superseded_errors=len(history),
                packages={name:importlib.metadata.version(name) for name in ('piper-tts','onnxruntime','numpy')},
                scope='actual unaccelerated synthesized WAVs; duration measurement, not speech-quality validation')
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
