import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('/scratch/yy5919/pyro-resume-results/local-dev-report');rows=json.loads((r/'dev-results.json').read_text())['rows'];selectors=['U','V','A_cue','AV','O'];fig,axes=plt.subplots(1,2,figsize=(10,4))
for offset,prior,color in [(-.18,'P0','#2878b5'),(.18,'P1','#df8f44')]:
 vals=[next(x for x in rows if x['selector']==s and x['prior']==prior) for s in selectors]
 axes[0].bar([i+offset for i in range(5)],[v['placement_f1'] for v in vals],.36,label=prior,color=color)
 axes[1].bar([i+offset for i in range(5)],[v['valid']/v['expected'] for v in vals],.36,label=prior,color=color)
for ax,title in zip(axes,['Placement F1 (all 10 dev clips)','Valid outputs / prepared moments']):
 ax.set_xticks(range(5),selectors);ax.set_ylim(0,1);ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('REFRAMED local development pilot — not held-out results');fig.tight_layout();fig.savefig(r/'dev-results.png',dpi=160);fig.savefig(r/'dev-results.pdf')
