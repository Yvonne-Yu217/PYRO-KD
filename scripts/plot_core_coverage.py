"""Plot requested-to-rendered AD coverage from export_core_results JSON."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
 p.add_argument('--model',default='qwen');p.add_argument('--title',required=True)
 a=p.parse_args();data=json.loads(a.input.read_text());rows=[r for r in data['rows'] if r['model']==a.model and not r['status'].startswith('deferred')]
 if not rows:raise ValueError('no evaluated model rows')
 fields=['requested_moments','selected_moments','expected','valid','tts_fit'];labels=['Requested','Selected','Prepared','Valid text','Speech-fitting*']
 if any(r[k] is None for r in rows for k in fields):raise ValueError('complete coverage metrics required')
 counts=[sum(r[k] for r in rows) for k in fields]
 if not counts[0] or any(a<b for a,b in zip(counts,counts[1:])):raise ValueError('invalid coverage funnel')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4));bars=ax.bar(labels,counts,color=['#adb5bd','#7a9eaf','#4e8098','#26739a','#12577a'])
 for bar,n in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,n+counts[0]*.018,f'{n:,}\n{n/counts[0]:.1%}',ha='center',va='bottom',fontsize=10)
 ax.set_ylim(0,counts[0]*1.18);ax.set_ylabel(f'Moments across {len(rows)} cells');ax.set_title(a.title);ax.spines[['top','right']].set_visible(False);fig.text(.5,.015,'*Individual descriptions; narration self-overlap is reported separately.',ha='center',fontsize=8);fig.tight_layout(rect=(0,.06,1,1))
 a.output.parent.mkdir(parents=True,exist_ok=True)
 for suffix in ['.png','.pdf']:fig.savefig(a.output.with_suffix(suffix),dpi=180)
 print(json.dumps({'counts':dict(zip(fields,counts)),'output':str(a.output)}))

if __name__=='__main__':main()
