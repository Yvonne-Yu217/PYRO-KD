"""Inventory pinned release ZIP members using bounded verified HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
from pathlib import Path
import urllib.request
import urllib.error
import urllib.parse
import zipfile


class HTTPRangeFile(io.RawIOBase):
    def __init__(self,url,size):
        self.url,self.size,self.position=url,size,0
        self.buffer=b'';self.buffer_start=0;self.transferred=0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position
    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.size+offset
        if position<0: raise ValueError('negative seek')
        self.position=position;return position
    def read(self,n=-1):
        if n<0: n=self.size-self.position
        n=min(n,max(0,self.size-self.position))
        if n==0: return b''
        if n>16*1024*1024: raise ValueError('refusing unbounded remote read')
        start=self.position
        if not self.buffer_start<=start or start+n>self.buffer_start+len(self.buffer):
            end=min(self.size,start+max(n,65536))-1
            separator='&' if urllib.parse.urlsplit(self.url).query else '?'
            # Refresh resolver redirects for this byte range; stale CDN redirects can time out.
            url=self.url+separator+f'download=true&pyro_range={start}-{end}'
            request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request,timeout=20) as response:
                        expected=f'bytes {start}-{end}/{self.size}'
                        if response.status!=206 or response.headers.get('Content-Range')!=expected:
                            raise ValueError('server did not honor exact range; refusing whole-archive download')
                        data=response.read(end-start+2)
                    break
                except (urllib.error.URLError, TimeoutError) as exc:
                    if attempt==2:
                        # ZipFile treats OSError during end-record reads as bad ZIP.
                        raise RuntimeError("range transport failed after three attempts") from exc
            if len(data)!=end-start+1: raise ValueError('range length mismatch')
            self.buffer_start,self.buffer=start,data;self.transferred+=len(data)
        begin=start-self.buffer_start
        self.position+=n
        return self.buffer[begin:begin+n]


def inventory(release,item):
    url=f"https://huggingface.co/datasets/{release['repo']}/resolve/{release['revision']}/{item['path']}"
    stream=HTTPRangeFile(url,item['size'])
    with zipfile.ZipFile(stream) as archive:
        members=[{'name':info.filename,'size':info.file_size,'compressed_size':info.compress_size,
                  'crc':info.CRC,'compression':info.compress_type,'header_offset':info.header_offset}
                 for info in archive.infolist() if not info.is_dir()]
    return {'archive':item['path'],'archive_bytes':item['size'],'inventory_transfer_bytes':stream.transferred,'members':members}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('preserve existing inventory; use another output path')
    release=json.loads(args.release.read_text())
    archives=[r for r in release['files'] if r['path'].startswith('videos_') and r['path'].endswith('.zip')]
    parts=args.output.parent/'archive-index-parts'
    parts.mkdir(parents=True,exist_ok=True)
    def checkpoint(item):
        path=parts/(item['path']+'.json')
        if path.exists():
            cached=json.loads(path.read_text())
            if cached['revision']==release['revision'] and cached['record']['archive_bytes']==item['size']:
                return cached['record']
            raise ValueError('checkpoint identity mismatch')
        record=inventory(release,item)
        temporary=path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'revision':release['revision'],'record':record})+'\n')
        temporary.replace(path)
        return record
    records=[];failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(checkpoint,item):item for item in archives}
        for future in as_completed(futures):
            item=futures[future]
            try:
                record=future.result();records.append(record)
                print(json.dumps({'archive':item['path'],'members':len(record['members'])}),flush=True)
            except Exception as exc:
                failures.append({'archive':item['path'],'error_type':type(exc).__name__})
                print(json.dumps(failures[-1]),flush=True)
    records.sort(key=lambda r:r['archive'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'repo':release['repo'],'revision':release['revision'],
        'complete':not failures and len(records)==len(archives),'failures':failures,'archives':records},indent=2)+'\n')
    print(json.dumps({'archives':len(records),'members':sum(len(r['members']) for r in records),
                      'inventory_transfer_bytes':sum(r['inventory_transfer_bytes'] for r in records)}))
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
