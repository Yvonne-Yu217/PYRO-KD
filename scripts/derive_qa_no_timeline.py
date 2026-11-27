"""Make paired no-timeline QA inputs without changing frames or questions."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.generation.qa import build_qa_prompt, reject_labels

def derive(row):
    if row.get('audio_context')!='timeline' or 'audio_timeline' not in row:raise ValueError('timeline input required')
    reject_labels(row)
    out=dict(row);out.pop('audio_timeline');out.pop('audio_timeline_provenance',None);out['audio_context']='none'
    build_qa_prompt(out)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve frozen input')
    raw=a.input.read_bytes();rows=[json.loads(x) for x in raw.splitlines()];assert len(rows)==len({r['id'] for r in rows})==36
    derived=[derive(r) for r in rows]
    a.output.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in derived))
    receipt=dict(source=str(a.input.resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),questions=len(rows),change='Remove prompt timeline only; frames, selected times, questions and options unchanged')
    a.output.with_suffix(a.output.suffix+'.derivation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
