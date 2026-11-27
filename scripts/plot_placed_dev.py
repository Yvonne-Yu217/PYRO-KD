"""Plot exploratory placement F1 intervals from evaluate_placed_dev output."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
data=json.loads(args.input.read_text())
preferred=['U_snapped','A_trigger','A_suppress','AV_trigger','AV_suppress','V_local','V_memory','O_snapped']
names=[name for name in preferred if name in data['summaries']]+sorted(set(data['summaries'])-set(preferred))
means=[data['summaries'][name]['macro_f1']['mean'] for name in names]
low=[m-data['summaries'][name]['macro_f1']['ci_low'] for name,m in zip(names,means)]
high=[data['summaries'][name]['macro_f1']['ci_high']-m for name,m in zip(names,means)]
fig,ax=plt.subplots(figsize=(8,4.6),layout='constrained')
ax.errorbar(means,range(len(names)),xerr=[low,high],fmt='o',capsize=4,color='#245a81')
ax.set_yticks(range(len(names)),[n.replace('_',' ') for n in names]);ax.invert_yaxis()
ax.set(xlim=(0,1),xlabel='Macro placement F1 (±2 seconds)',title='Exploratory REFRAMED development placement')
ax.grid(axis='x',alpha=.25)
selected=[data['summaries'][name]['selected'] for name in names]
requested=[data['summaries'][name]['requested'] for name in names]
span=lambda values: str(min(values)) if min(values)==max(values) else f'{min(values)}–{max(values)}'
clips=len({r['video_id'] for r in data['rows']});movies=len({r['movie_id'] for r in data['rows']})
fig.suptitle(f'{clips} clips / {movies} movies · 95% bootstrap intervals · untuned selectors\n{span(selected)} selected / {span(requested)} requested per condition; descriptive diagnostics',fontsize=10)
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output,dpi=180)
plt.close(fig)
