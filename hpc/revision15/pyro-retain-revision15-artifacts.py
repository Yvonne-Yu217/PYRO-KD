#!/usr/bin/env python3
"""Archive finalized small Revision1.5 artifacts outside Git after completion."""
import argparse, hashlib, json, os
from pathlib import Path
import shutil, subprocess, sys, tarfile, tempfile

REPO=Path('/scratch/yy5919/pyro-revision15-records')
sys.path.insert(0,str(REPO))
from scripts.reproduce_revision15_tables import completion_gate
RESULTS=Path('/scratch/yy5919/pyro-resume-results')
DERIVED=Path('/scratch/yy5919/pyro-kd-derived')
FORMAL=Path('/scratch/yy5919/pyro-revision15-formal')
ALLOWED={'.json','.jsonl','.toml','.csv','.md','.txt','.svg','.png','.pdf','.tex'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def permitted(p):
    scoped_driver = p.is_relative_to(REPO/'hpc/revision15') and p.suffix in {'.py','.sbatch','.sh'}
    return p.is_file() and not p.is_symlink() and (p.suffix in ALLOWED or scoped_driver) and not any(s in p.name.lower() for s in ('secret','credential','cookie','ledger','tokenizer','safetensor','pytorch_model'))

def selected(manifest_path,manifest):
    roots=[RESULTS/'experiments/reframed_test_local_v1',RESULTS/'long-qa-frozen',RESULTS/'egoschema-heldout',RESULTS/'formal-core-temporal',RESULTS/'formal-ablation5-temporal',REPO/'configs',FORMAL/'configs',REPO/'docs/results',REPO/'docs/verification',REPO/'docs/report',REPO/'docs/slides',RESULTS/'reproduced-tables-v1',REPO/'hpc/revision15']
    files={manifest_path.resolve()}
    for root in roots:
        if not root.is_dir():raise ValueError('required root absent: '+str(root))
        files.update(p.resolve() for p in root.rglob('*') if permitted(p))
    for name in ['AGENT_LOG.md','EXPERIMENT_STATUS.md','RESULTS_REPRODUCTION.md','PROPOSAL_REVISION_1_5.txt','REVISION_1_5_REQUIREMENTS.md','REVISION15_DELIVERY.md']:
        files.add(REPO/'docs'/name)
    files.add(REPO/'README.md')
    files.add(REPO/'AGENTS.md')
    reproduction=RESULTS/'reproduced-tables-v1/reproduction.json'
    if not json.loads(reproduction.read_text()).get('complete'):
        raise ValueError('finalized successful table reproduction required')
    # Canonical QA/moment inputs, TTS metadata and bound evaluation sources only.
    # No image/audio/feature contents are included, even when paths occur in JSON.
    for name in manifest['artifact_sha256']:
        p=Path(name).resolve()
        if p.is_relative_to(DERIVED/'moments') or p.is_relative_to(DERIVED/'tts'):
            if permitted(p):files.add(p)
    for p in files:
        if not permitted(p):raise ValueError('unsafe or missing selected file: '+str(p))
        if p.stat().st_size>128*1024*1024:raise ValueError('unexpected large artifact: '+str(p))
    return sorted(files)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion-manifest',type=Path,default=RESULTS/'revision15-completion.json')
    parser.add_argument('--destination',type=Path,default=Path('/home/yy5919/pyro-kd-artifacts/revision15-20261123'))
    parser.add_argument('--execute',action='store_true',help='Required to copy; otherwise print finalized selection after gate')
    a=parser.parse_args()
    if a.destination.exists():raise ValueError('preserve existing retention directory')
    manifest=completion_gate(a.completion_manifest)
    files=selected(a.completion_manifest,manifest)
    size=sum(p.stat().st_size for p in files)
    if not a.execute:
        print(json.dumps({'files':[str(p) for p in files],'bytes':size,'destination':str(a.destination)},indent=2));return
    a.destination.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.destination.parent).free < size*3+100*1024*1024:raise ValueError('insufficient free space for safe staging')
    stage=Path(tempfile.mkdtemp(prefix='.revision15-staging-',dir=a.destination.parent))
    try:
        inventory={};payload=stage/'payload';payload.mkdir()
        for p in files:
            digest=sha(p);relative=Path('sources')/p.relative_to('/')
            target=payload/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
            if sha(target)!=digest or sha(p)!=digest:raise ValueError('source changed during copy: '+str(p))
            inventory[str(relative)]={'source':str(p),'sha256':digest,'bytes':target.stat().st_size}
        source_shas={str(p):subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip() for p in [REPO,FORMAL,Path('/scratch/yy5919/pyro-revision15-long-prep'),Path('/scratch/yy5919/pyro-revision15-next')]}
        record={'scope':'Finalized local predictions, failures, per-example metrics and provenance; no large media/models/caches; paid/human requirements deferred','completion_manifest':str(a.completion_manifest),'completion_manifest_sha256':sha(a.completion_manifest),'source_shas':source_shas,'files':inventory,'retention_scope_note':'This is a frozen archive snapshot. Retention verification records added to the repository afterward are not retroactively covered by these hashes.', 'uncompressed_bytes':size,'driver_sha256':sha(__file__)}
        (payload/'inventory.json').write_text(json.dumps(record,indent=2)+'\n')
        archive=stage/'artifacts.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            tar.add(payload/'inventory.json',arcname='inventory.json')
            for name in inventory:tar.add(payload/name,arcname=name,recursive=False)
        with tarfile.open(archive,'r:gz') as tar:
            members=tar.getmembers()
            if len(members)!=len(inventory)+1 or {m.name for m in members}!=set(inventory)|{'inventory.json'}:raise ValueError('tar inventory mismatch')
            for member in members:
                if not member.isfile():raise ValueError('non-file archive member')
                expected=sha(payload/'inventory.json') if member.name=='inventory.json' else inventory[member.name]['sha256']
                if hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=expected:raise ValueError('tar content verification failed')
        completion_gate(a.completion_manifest)
        for item in inventory.values():
            if sha(item['source'])!=item['sha256']:raise ValueError('source changed before publication')
        shutil.copyfile(payload/'inventory.json',stage/'inventory.json')
        (stage/'verification.json').write_text(json.dumps({'verified':True,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'files':len(inventory),'completion_manifest_sha256':record['completion_manifest_sha256'],'source_shas':source_shas},indent=2)+'\n')
        shutil.rmtree(payload)
        os.rename(stage,a.destination)
        print(json.dumps({'verified':True,'destination':str(a.destination),'files':len(inventory),'archive_sha256':sha(a.destination/'artifacts.tar.gz')}))
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True)
        raise

if __name__=='__main__':main()
