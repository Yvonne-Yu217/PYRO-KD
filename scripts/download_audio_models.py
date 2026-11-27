#!/usr/bin/env python
"""Cache pinned speech/event model files for later offline feature extraction."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/audio_models.json'))
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    report={'complete':False,'models':[]}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    for model in json.loads(args.config.read_text())['models']:
        root=Path(snapshot_download(model['repo_id'],revision=model['revision'],
                  cache_dir=str(args.cache_dir),allow_patterns=[f['name'] for f in model['files']],max_workers=4))
        errors=[f['name'] for f in model['files'] if not (root/f['name']).is_file() or (root/f['name']).stat().st_size!=f['bytes']]
        record={'repo_id':model['repo_id'],'revision':model['revision'],'snapshot':str(root),'size_errors':errors}
        report['models'].append(record)
        args.report.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(record),flush=True)
        if errors:
            raise RuntimeError(f'Incomplete model files: {errors}')
    report['complete']=True
    args.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
