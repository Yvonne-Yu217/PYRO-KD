"""Stream-verify a model snapshot against pinned upstream file identities.

Safetensors headers are inspected without loading tensor data or a model.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--download-report',type=Path,required=True)
    p.add_argument('--upstream-identities',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('preserve previous verification')
    config=json.loads(args.config.read_text())
    downloaded=json.loads(args.download_report.read_text())
    upstream=json.loads(args.upstream_identities.read_text())
    if len(config['models'])!=1 or len(downloaded['models'])!=1 or not downloaded['complete']:
        raise ValueError('one complete pinned snapshot required')
    model=config['models'][0];local=downloaded['models'][0]
    if (model['repo_id']!=local['repo_id'] or model['revision']!=local['revision']
            or model['revision']!=upstream['revision']):
        raise ValueError('snapshot revision mismatch')
    root=Path(local['snapshot']);identities={r['name']:r for r in upstream['files']}
    records=[]
    for item in model['files']:
        name=item['name'];path=root/name;identity=identities[name]
        size=path.stat().st_size
        if size!=item['bytes'] or size!=identity['bytes']:raise ValueError('size mismatch: '+name)
        digest=hashlib.sha256();git=hashlib.sha1(f'blob {size}\0'.encode())
        with path.open('rb') as f:
            while chunk:=f.read(8*1024*1024):
                digest.update(chunk)
                if not identity['lfs']:git.update(chunk)
        if identity['lfs']:
            if digest.hexdigest()!=identity['lfs']['sha256']:raise ValueError('LFS SHA256 mismatch: '+name)
            check='upstream LFS SHA256'
        else:
            if git.hexdigest()!=identity['blob_id']:raise ValueError('Git blob identity mismatch: '+name)
            check='upstream Git blob SHA1'
        records.append(dict(name=name,bytes=size,sha256=digest.hexdigest(),verified_against=check))
        print(json.dumps({'file':name,'verified':True}),flush=True)
    index=json.loads((root/'model.safetensors.index.json').read_text())
    mapping=index['weight_map']
    shards={r['name'] for r in records if r['name'].endswith('.safetensors')}
    if set(mapping.values())!=shards:raise ValueError('index/shard inventory mismatch')
    from safetensors import safe_open
    count=0
    for shard in sorted(shards):
        with safe_open(str(root/shard),framework='pt',device='cpu') as f:
            keys=set(f.keys())
        if keys!={key for key,value in mapping.items() if value==shard}:
            raise ValueError('header keys differ from pinned index: '+shard)
        count+=len(keys)
    report=dict(complete=True,repo_id=model['repo_id'],revision=model['revision'],snapshot=str(root),
        files=records,bytes=sum(r['bytes'] for r in records),shards=len(shards),tensors=count,
        model_weights_loaded=False,config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        identities_sha256=hashlib.sha256(args.upstream_identities.read_bytes()).hexdigest(),
        scope='complete file identity and safetensors header/index coverage; no model load, GPU feasibility or inference validation')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='files'}))


if __name__=='__main__':main()
