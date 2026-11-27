"""CPU-only real development pilot: pinned OpenCLIP extraction and memory ranking."""
import argparse
import sys,json,hashlib,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.cache_visual_features import process_video
from src.utils.visual_features import clip_embedder,load_embeddings,VisualCueSelector,DEFAULT_CONFIG,load_current_features
from huggingface_hub import hf_hub_download
import torch
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cache',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
torch.set_num_threads(4)
cache=args.cache
t=time.time()
status,features=process_video(cache,lambda:clip_embedder(batch_size=8,device='cpu'))
features=load_current_features(cache)
assert features is not None
emb=load_embeddings(cache,features)
assert emb is not None
selector=VisualCueSelector()
local=selector.select(features,6)
memory=selector.select_memory(features,emb,6)
weights=Path(hf_hub_download('laion/CLIP-ViT-B-32-laion2B-s34B-b79K','open_clip_model.safetensors',revision=DEFAULT_CONFIG['clip_revision'],local_files_only=True))
record={'video_id':cache.name,'status':status,'frames':features['num_frames'],'embedding_shape':list(emb.shape),'shots':features['shots'],'local_times':[x.time for x in local],'memory_times':[x.time for x in memory],'elapsed_seconds':time.time()-t,'config':DEFAULT_CONFIG,'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'note':'K=6 exploratory ranking check, not frozen placement or evaluation'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
