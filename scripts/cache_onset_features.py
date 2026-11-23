"""Cache inexpensive onset cues separately from pinned AST/CLAP outputs."""
import argparse,hashlib,importlib.metadata,io,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.audio_features import resolve_audio_path
from src.utils.media import load_manifest,cache_is_current
from src.utils.onset_features import SETTINGS,compute_onsets

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-dir',type=Path,required=True);args=p.parse_args()
import soundfile as sf
records=[]
for manifest_path in sorted(args.cache_dir.glob('*/manifest.json')):
 cache=manifest_path.parent;manifest=load_manifest(cache)
 if not cache_is_current(manifest): raise ValueError('stale media cache')
 audio=resolve_audio_path(manifest,cache);raw=audio.read_bytes()
 provenance={'settings':SETTINGS,'audio_sha256':hashlib.sha256(raw).hexdigest(),
             'libraries':{name:importlib.metadata.version(name) for name in ('librosa','numpy','soundfile')}}
 digest=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
 output=cache/'onset_features.json'
 old=json.loads(output.read_text()) if output.exists() else {}
 if old.get('config_hash')==digest and old.get('complete'): status='current';payload=old
 else:
  wave,sr=sf.read(io.BytesIO(raw),dtype='float32');tracks=compute_onsets(wave,sr)
  if hashlib.sha256(audio.read_bytes()).hexdigest()!=provenance['audio_sha256']: raise ValueError('WAV changed')
  payload={**tracks,'provenance':provenance,'config_hash':digest,'complete':True}
  temporary=output.with_suffix('.json.tmp');temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(output);status='computed'
 record={'video_id':cache.name,'status':status,'windows':len(payload['times'])};records.append(record);print(json.dumps(record),flush=True)
if not records: raise ValueError('no caches')
(args.cache_dir/'onset_features_report.json').write_text(json.dumps({'complete':True,'results':records},indent=2)+'\n')
