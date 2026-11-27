"""Publish every comparison in a completed statistical family without filtering."""
import argparse,csv,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--title',required=True);p.add_argument('--scope',required=True);a=p.parse_args()
 raw=a.input.read_bytes();data=json.loads(raw);results=data.get('results',data.get('comparisons'));assert isinstance(results,list) and results
 rows=[]
 for r in results:
  b=r['bootstrap'];w=r['wilcoxon'];adjusted=w.get('holm_pvalue',w.get('p_holm'))
  if r['excluded'] or adjusted is None:raise ValueError('complete paired family and adjusted p-values required')
  rows.append(dict(a=r['condition_a'],b=r['condition_b'],metric=r['metric'],unit=r['unit'],n=b['n'],mean_diff=b['mean_diff'],ci_low=b['ci_low'],ci_high=b['ci_high'],pvalue=w['pvalue'],holm_pvalue=adjusted,oracle_upper_bound=r.get('oracle_upper_bound',False)))
 a.output_dir.mkdir(parents=True,exist_ok=True)
 with (a.output_dir/'paired-contrasts.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 lines=['# '+a.title,'',a.scope,'','Pointwise95% paired bootstrap intervals; Holm correction applies to p-values across the full declared family, not to these intervals.','', '| A − B | Units | Mean difference | 95% CI | Holm p |','|---|---:|---:|---|---:|']
 for r in rows:lines.append(f"| {r['a']} − {r['b']} | {r['n']} | {r['mean_diff']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['holm_pvalue']:.4f} |")
 (a.output_dir/'paired-contrasts.md').write_text('\n'.join(lines)+'\n')
 (a.output_dir/'paired-contrasts.provenance.json').write_text(json.dumps(dict(source=str(a.input.resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),comparisons=len(rows),scope=a.scope),indent=2)+'\n')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(10,max(3,0.4*len(rows)+1.5)))
 for i,r in enumerate(rows):
  color='#888888' if r['oracle_upper_bound'] else '#176b91';ax.hlines(i,r['ci_low'],r['ci_high'],color=color);ax.plot(r['mean_diff'],i,'o',color=color)
 ax.set_yticks(range(len(rows)),[r['a']+' − '+r['b'] for r in rows]);ax.invert_yaxis();ax.axvline(0,color='gray',linewidth=.7);ax.set_title(a.title);ax.set_xlabel('Paired mean difference (pointwise95% bootstrap CI)');ax.spines[['top','right']].set_visible(False);fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(a.output_dir/f'paired-contrasts.{ext}',dpi=180)
 print(json.dumps({'comparisons':len(rows),'output':str(a.output_dir)}))
if __name__=='__main__':main()
