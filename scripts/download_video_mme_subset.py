"""Resumable ranged extraction of frozen Video-MME ZIP members, with CRC/SHA/probe."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import zlib
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.inventory_video_mme import HTTPRangeFile


def checksum(path):
    digest=hashlib.sha256();crc=0;size=0
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            digest.update(block);crc=zlib.crc32(block,crc);size+=len(block)
    return size,crc & 0xffffffff,digest.hexdigest()


def unpack(payload,target,member):
    temporary=target.with_suffix('.mp4.part')
    decoder=zlib.decompressobj(-15) if member['compression']==8 else None
    if member['compression'] not in (0,8): raise ValueError('unsupported ZIP compression')
    total=0
    with payload.open('rb') as source, temporary.open('wb') as dest:
        for block in iter(lambda:source.read(1024*1024),b''):
            decoded=decoder.decompress(block) if decoder else block
            total+=len(decoded)
            if total>member['size']: raise ValueError('decompression exceeds declared size')
            dest.write(decoded)
        if decoder:
            dest.write(decoder.flush())
            if not decoder.eof or decoder.unused_data: raise ValueError('invalid deflate stream')
    size,crc,digest=checksum(temporary)
    if size!=member['size'] or crc!=member['crc']: raise ValueError('ZIP member size/CRC mismatch')
    temporary.replace(target)
    return digest


def download(release,row,root):
    member=row['member']
    if member is None: raise ValueError('selected ID not yet located in archive inventory')
    video_id=row['video_id']
    if not video_id.isdigit(): raise ValueError('expected numeric frozen video ID')
    root.mkdir(parents=True,exist_ok=True)
    target=root/(video_id+'.mp4')
    identity={'repo':release['repo'],'revision':release['revision'],'video_id':video_id,'member':member}
    identity_path=root/(video_id+'.source.json')
    if identity_path.exists():
        if json.loads(identity_path.read_text())!=identity: raise ValueError('resume source identity changed')
    else:
        identity_path.write_text(json.dumps(identity,indent=2)+'\n')
    if target.exists():
        size,crc,digest=checksum(target)
        if size!=member['size'] or crc!=member['crc']: raise ValueError('existing final video failed CRC/size')
    else:
        url=f"https://huggingface.co/datasets/{release['repo']}/resolve/{release['revision']}/{member['archive']}"
        remote=HTTPRangeFile(url,member['archive_bytes'])
        remote.seek(member['header_offset']);header=struct.unpack('<4s5H3I2H',remote.read(30))
        if header[0]!=b'PK\x03\x04' or header[2]&1 or header[3]!=member['compression']:
            raise ValueError('invalid/encrypted/local ZIP header mismatch')
        name=remote.read(header[9]).decode('utf-8' if header[2]&0x800 else 'cp437')
        if name!=member['name']: raise ValueError('local ZIP filename mismatch')
        begin=member['header_offset']+30+header[9]+header[10]
        if begin+member['compressed_size']>member['archive_bytes']: raise ValueError('member outside archive')
        payload=root/(video_id+'.compressed.part')
        written=payload.stat().st_size if payload.exists() else 0
        if written>member['compressed_size']: raise ValueError('oversized partial payload')
        remote.seek(begin+written)
        with payload.open('ab') as handle:
            while written<member['compressed_size']:
                block=remote.read(min(8*1024*1024,member['compressed_size']-written))
                if not block: raise ValueError('unexpected end of member')
                handle.write(block);handle.flush();written+=len(block)
                print(json.dumps({'video_id':video_id,'compressed_bytes':written,'total':member['compressed_size']}),flush=True)
        digest=unpack(payload,target,member)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(target)],text=True))
    record={'video_id':video_id,'path':str(target),'sha256':digest,'zip_crc32':member['crc'],
            'bytes':member['size'],'duration_seconds':float(probe['format']['duration']),
            'has_audio':any(s['codec_type']=='audio' for s in probe['streams']),
            'has_video':any(s['codec_type']=='video' for s in probe['streams']),
            'streams':[{k:s.get(k) for k in ('codec_type','codec_name','sample_rate','channels','width','height','duration')} for s in probe['streams']]}
    (root/(video_id+'.verified.json')).write_text(json.dumps(record,indent=2)+'\n')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--members',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--video-id',required=True,help='frozen numeric ID, or all for the full frozen subset')
    args=p.parse_args()
    release=json.loads(args.release.read_text());members=json.loads(args.members.read_text())
    if release['revision']!=members['revision']: raise ValueError('release/member revision mismatch')
    if args.video_id!='all':
        row=next(r for r in members['videos'] if r['video_id']==args.video_id)
        print(json.dumps(download(release,row,args.output_dir)),flush=True)
        return
    from concurrent.futures import ThreadPoolExecutor,as_completed
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(download,release,row,args.output_dir):row for row in members['videos']}
        for future in as_completed(futures):
            row=futures[future]
            try:
                result={'status':'verified',**future.result()}
            except Exception as exc:
                result={'video_id':row['video_id'],'status':'failed','error_type':type(exc).__name__,'error':str(exc)}
            results.append(result)
            print(json.dumps(result),flush=True)
            args.output_dir.mkdir(parents=True,exist_ok=True)
            report={'complete':len(results)==len(members['videos']) and all(r['status']=='verified' for r in results),
                    'expected':len(members['videos']),'results':sorted(results,key=lambda r:r['video_id'])}
            temporary=args.output_dir/'download-report.json.tmp'
            temporary.write_text(json.dumps(report,indent=2)+'\n')
            temporary.replace(args.output_dir/'download-report.json')
    if not report['complete']: raise SystemExit(1)


if __name__=='__main__': main()
