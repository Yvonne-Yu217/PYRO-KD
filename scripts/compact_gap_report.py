"""Keep checked per-video totals in Git while binding detailed gap artifacts."""
import argparse,hashlib,json,shutil
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 if a.output_dir.exists():raise FileExistsError('preserve prior compact report')
 source=a.input_dir/'measured-gap-utilization.json';raw=source.read_bytes();data=json.loads(raw)
 for result in data['results']:
  for video in result['videos']:
   for track in video['tracks'].values():
    track['gap_count']=len(track.pop('gaps'));track['caption_ratio_count']=len(track.pop('per_caption_remaining_gap_ratios'))
 data['full_details']={'path':str(source.resolve()),'sha256':hashlib.sha256(raw).hexdigest()}
 a.output_dir.mkdir(parents=True)
 (a.output_dir/source.name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
 for extension in ['md','csv']:shutil.copyfile(a.input_dir/f'measured-gap-utilization.{extension}',a.output_dir/f'measured-gap-utilization.{extension}')
 print(json.dumps({'cells':len(data['results']),'full_details':data['full_details']}))
if __name__=='__main__':main()
