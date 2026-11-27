"""Attach exact waveform spans to existing development AD selections, no new selections."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import soundfile as sf
from src.generation.native_audio import span_for_center,load_native_audio
from src.generation.prompts import CONTEXT_SECONDS
from src.utils.audio_features import resolve_audio_path
from src.utils.media import load_manifest,cache_is_current


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split-config',type=Path,default=Path('configs/reframed_splits.json'))
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve earlier native inputs')
    split=json.loads(args.split_config.read_text())
    if split['split']!='validation':raise ValueError('development adaptation only')
    ids={r['video_id'] for r in split['clips']}
    rows=[json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('unique nonempty moments required')
    sources={};output=[]
    for row in rows:
        ident=row['video_id']
        if ident not in ids or row.get('audio_timeline') or row.get('transcript') or row.get('native_audio'):
            raise ValueError('use frozen development moments without text audio context or prior native spans')
        if ident not in sources:
            cache=args.cache_dir/ident;manifest=load_manifest(cache)
            if not cache_is_current(manifest):raise ValueError('stale media')
            path=resolve_audio_path(manifest,cache)
            if path is None:raise ValueError('native audio unavailable; never fabricate a silent replacement')
            info=sf.info(path)
            if info.channels!=1 or info.samplerate!=16000:raise ValueError('mono16k prepared audio required')
            sources[ident]=(Path(path),info.frames,hashlib.sha256(Path(path).read_bytes()).hexdigest())
        path,samples,digest=sources[ident]
        center=row.get('frame_center_time',row['time'])
        start,end=span_for_center(center,samples,16000,CONTEXT_SECONDS)
        with sf.SoundFile(path) as f:
            f.seek(start);wave=f.read(end-start,dtype='float32')
        audio=dict(path=str(path.resolve()),source_sha256=digest,source_samples=samples,sample_rate=16000,
                   frame_center_time=center,context_seconds=CONTEXT_SECONDS,start_sample=start,end_sample=end,
                   start_seconds=start/16000,end_seconds=end/16000,samples_sha256=hashlib.sha256(wave.tobytes()).hexdigest())
        # Independent loader validates the same span contract used by inference.
        load_native_audio(audio)
        output.append({**row,'audio_context':'native','native_audio':audio})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        for row in output:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps({'moments':len(output),'videos':len(sources),'context_seconds':CONTEXT_SECONDS,
                      'scope':'development waveform input adaptation only; no model inference or selector choice'}))


if __name__=='__main__':main()
