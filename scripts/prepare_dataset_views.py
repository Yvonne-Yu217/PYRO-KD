"""Create loader-compatible views without changing the transferred raw datasets."""
import argparse
from pathlib import Path
import zipfile
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--view-root', type=Path, required=True)
args=parser.parse_args()
base=args.source_root.resolve()
views=args.view_root.resolve()
if views.is_relative_to(base):
 parser.error('view-root must be outside the raw source directory')
views.mkdir(parents=True,exist_ok=True)
def link(dst,src):
 if dst.is_symlink() and dst.resolve()==src.resolve(): return
 if dst.exists() or dst.is_symlink(): raise RuntimeError(f'refusing to replace {dst}')
 dst.symlink_to(src)
link(views/'mtg-ad',base/'mtg-ad')
link(views/'egoschema',base/'Evaluation-datasets/egoschema')
root=base/'Evaluation-datasets/reframed'
view=views/'reframed'
view.mkdir(exist_ok=True)
for name in ['train.jsonl','validation.jsonl','test.jsonl']:
 link(view/name,root/'source'/name)
link(view/'videos',root/'videos')
with zipfile.ZipFile(root/'source/keys.zip') as z:
 for info in z.infolist():
  if info.is_dir(): continue
  if info.filename not in ['keys/private_key.pem','keys/skey.key']: raise ValueError('unexpected archive member')
  dst=view/info.filename
  dst.parent.mkdir(exist_ok=True)
  data=z.read(info)
  if dst.exists():
   if dst.read_bytes()!=data: raise RuntimeError('existing key differs')
  else: dst.write_bytes(data)
print(views)
