"""Export descriptive selector/ablation results with complete fixed membership."""
import argparse,csv,hashlib,json
from pathlib import Path

def export(data):
    expected=set(data['common_video_ids'])
    if len(expected)!=data['n_videos'] or not expected:raise ValueError('invalid membership')
    rows=[]
    for name,run in data['runs'].items():
        videos=run['videos'];ids=[v['video_id'] for v in videos]
        if not run['complete'] or run['missing_video_ids'] or run['unexpected_video_ids'] or set(ids)!=expected or len(ids)!=len(expected):raise ValueError('incomplete selector coverage: '+name)
        summary=run['summary'];f1=summary['macro_f1']
        rows.append(dict(selector=name,ablation=run['ablation'] or 'core',oracle=run['oracle'],videos=len(ids),movies=len({v['group'] for v in videos}),requested=sum(v['k'] for v in videos),selected=sum(v['num_selected'] for v in videos),shortfall_videos=summary['videos_with_shortfall'],picks_in_speech=sum(v['picks_in_speech'] for v in videos),zero_budget_picks=sum(v['zero_budget_picks'] for v in videos),precision=summary['macro_precision']['mean'],recall=summary['macro_recall']['mean'],f1=f1['mean']))
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
    raw=a.input.read_bytes();rows=export(json.loads(raw));a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'selector-results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    notes='Frozen30clip test subset. Descriptive clip-macro temporal metrics at ±2s, before caption preparation/generation. Requested K need not equal realized K. Picks in speech and zero-budget picks remain visible; selector overlap estimates are not actual rendered speech. O uses reference timestamps and is an oracle upper bound. No detector accuracy or human-quality claim.'
    (a.output_dir/'selector-results.json').write_text(json.dumps(dict(rows=rows,notes=notes,source=str(a.input.resolve()),source_sha256=hashlib.sha256(raw).hexdigest()),indent=2)+'\n')
    keys=['selector','ablation','requested','selected','shortfall_videos','picks_in_speech','zero_budget_picks','precision','recall','f1']
    fmt=lambda x:f'{x:.3f}' if isinstance(x,float) else str(x)
    lines=['# REFRAMED selector and ablation results','',notes,'','| '+' | '.join(keys)+' |','|'+'---|'*len(keys)]+['| '+' | '.join(fmt(r[k]) for k in keys)+' |' for r in rows]
    (a.output_dir/'selector-results.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,10));colors=['#999999' if r['oracle'] else '#176b91' if r['ablation']=='core' else '#78a9bb' for r in rows]
    ax.barh([r['selector'] for r in rows],[r['f1'] for r in rows],color=colors);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Mean clip temporal F1 (±2s)');ax.set_title('REFRAMED: 31 frozen selector configurations, 30 clips');ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(a.output_dir/f'selector-results.{suffix}',dpi=180)
    print(json.dumps({'configurations':len(rows),'videos':rows[0]['videos']}))
if __name__=='__main__':main()
