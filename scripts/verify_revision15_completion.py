"""Fail-closed inventory of the frozen local Revision 1.5 experiments.

This does not certify the deferred paid/API or human-validation proposal work.
QA scoring artifacts are opened only after the whole dataset's raw coverage passes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.generation.outputs import split_rows, is_success
from src.experiments.config import load_config
from src.experiments.matrix import core_generation_runs, ablation5_runs
from src.experiments.generation_stage import generations_path, moments_path
from scripts.export_qa_paired_metrics import export_reports

FORMAL_SHA = '09abb6c0512e5e97549ad4e66d3bae05a68c42c8'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_rows(path):
    data = Path(path).read_bytes()
    if data and not data.endswith(b'\n'):
        raise ValueError('unterminated JSONL tail')
    rows = [json.loads(line) for line in data.splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('JSONL objects required')
    return rows

def coverage(rows, expected):
    final, superseded = split_rows(rows)
    counts = Counter(row.get('id') for row in final)
    actual = set(counts)
    return dict(expected=len(expected), attempted=len(final), valid=sum(is_success(r) for r in final),
                failed_ids=sorted(r['id'] for r in final if r.get('id') and not is_success(r)),
                missing_ids=sorted(set(expected)-actual), unexpected_ids=sorted(str(x) for x in actual-set(expected)),
                duplicate_ids=sorted(str(k) for k,v in counts.items() if v > 1),
                superseded_error_attempts=len(superseded))

def exact(info):
    return not any(info[k] for k in ('missing_ids','unexpected_ids','duplicate_ids'))

class Audit:
    def __init__(self):
        self.errors = []
        self.artifacts = {}
    def bind(self, path):
        path = Path(path)
        self.artifacts[str(path)] = sha(path)
        return path
    def obj(self, path):
        return json.loads(self.bind(path).read_text())
    def rows(self, path):
        return read_rows(self.bind(path))
    def check(self, condition, message):
        if not condition:
            raise ValueError(message)
    def attempt(self, name, function):
        try:
            return function()
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
            self.errors.append(f'{name}: {exc}')
            return None
    def evidence(self, path):
        obj = self.obj(path)
        # Only small experiment records; never revisit waveform, feature or model files.
        for source, digest in obj.get('sources', {}).items():
            p = Path(source)
            if '/generations/' in source or '/eval/' in source or '/tts/' in source and p.suffix in ('.json','.jsonl'):
                self.check(sha(p)==digest, f'stale exporter evidence: {p}')
        return obj

def audit_ad(a, args):
    cfgpath=args.formal_repo/'configs/experiments/reframed_test_local.toml'
    a.bind(cfgpath); cfg=load_config(cfgpath)
    root=args.results_root/'experiments/reframed_test_local_v1'
    ident=a.obj(root/'config.identity.json')
    for path,digest in ident['source_and_input_hashes'].items():
        a.check(sha(path)==digest, f'frozen source/input changed: {path}')
    a.bind(root/'config.resolved.json')
    prep=a.obj(root/'preparation_counts.json')
    a.check(prep['source_commit']==FORMAL_SHA and prep['local_core_requests']==3398,'formal source/count mismatch')
    runs=[r for r in core_generation_runs(cfg) if r.model_label=='qwen']
    runs += [r for r in ablation5_runs(cfg) if not r.timeline]
    a.check(len(runs)==12 and len({r.run_id for r in runs})==12,'expected twelve local AD cells')
    result={}
    for run in runs:
        def cell():
            expected_rows=a.rows(moments_path(root,run.selection)); expected={r['id'] for r in expected_rows}
            a.check(len(expected)==len(expected_rows),'duplicate moment IDs')
            required={'U':283,'V':305,'A_cue':384,'AV':384,'O':343}[run.selection.name]
            a.check(len(expected)==required,'frozen moment count mismatch')
            gp=generations_path(root,run); raw=a.rows(gp); info=coverage(raw,expected); result[run.run_id]=info
            a.check(exact(info),'incomplete/duplicate/unexpected generation IDs')
            final,_=split_rows(raw); valid={r['id'] for r in final if is_success(r)}
            a.check(all(r['run_id']==run.run_id for r in final),'generation run ID mismatch')
            e=a.obj(root/'eval'/f'generation_{run.run_id}.json')
            a.check(e['expected_moments']==len(expected) and not e['missing_moment_ids'] and e['expected_videos']==30,'metric coverage mismatch')
            a.check(sum(v['n_ok'] for v in e['videos'])==len(valid) and sum(v['n_error'] for v in e['videos'])==len(expected)-len(valid),'stale metric counts')
            for key in ('bertscore','sentence_cosine','clip_text'):
                a.check(e['text_metrics'][key]['status']=='ok','text metric incomplete: '+key)
            tr=args.tts_root/run.run_id
            render=a.obj(tr/'rendered/report.json'); overlap=a.obj(tr/'overlap/report.json')
            rp=tr/'rendered/rendered.jsonl'; rendered=a.rows(rp)
            a.check(render['complete'] and render['input_sha256']==sha(gp) and render['input_rows']==len(expected),'stale/incomplete TTS render')
            a.check(set(render['failed_ids'])==set(info['failed_ids']),'TTS failure coverage mismatch')
            a.check(overlap['complete'] and overlap['input_sha256']==sha(rp) and overlap['playback_speed']==1.2,'stale/incomplete overlap')
            for label,rows in [('rendered',rendered),('overlap',[r for v in overlap['videos'] for r in v['rows']])]:
                a.check(len(rows)==len(valid) and {r['id'] for r in rows}==valid,label+' ID coverage mismatch')
        a.attempt(run.run_id,cell)
    gap_ids=set()
    for path in args.gap_evidence:
        obj=a.attempt(str(path),lambda p=path:a.evidence(p))
        if obj: gap_ids.update(r['run_id'] for r in obj['results'])
    a.attempt('measured-gap coverage', lambda: a.check(gap_ids>={r.run_id for r in runs},'detailed measured-gap exporter evidence required for all twelve cells'))
    return result

def audit_qa(a,args,kind):
    repo=Path(__file__).resolve().parents[1]
    is_ego=kind=='egoschema'
    planpath=repo/'configs'/('egoschema_qa_comparisons.json' if is_ego else 'long_qa_comparisons.json')
    plan=a.obj(planpath)
    for filename in (['egoschema_qa_frozen.json'] if is_ego else ['long_visual_qa_frozen.json','long_audio_qa_frozen.json','long_no_timeline_qa_frozen.json']):
        a.bind(repo/'configs'/filename)
    conditions=['U','V_local','V_memory'] if is_ego else plan['conditions']
    member=a.obj(args.formal_repo/'configs'/('egoschema_pilot.json' if is_ego else 'video_mme_long_questions.json'))
    expected={x:x for x in member['heldout_q_uids']} if is_ego else {x['question_id']:x['video_id'] for x in member['questions']}
    a.check(len(expected)==(490 if is_ego else 36),'frozen membership size mismatch')
    root=args.results_root/('egoschema-heldout' if is_ego else 'long-qa-frozen')
    result={}; ready=True; predictions={}
    for condition in conditions:
        def raw_cell():
            p=root/(condition+('.jsonl' if is_ego else '-224.jsonl'))
            raw=a.rows(p); info=coverage(raw,expected);result[condition]=info
            a.check(exact(info),'raw coverage incomplete')
            final,_=split_rows(raw)
            a.check(all(str(r['video_id'])==expected[r['id']] for r in final),'question/video mapping mismatch')
            prov=a.obj(str(p)+'.provenance.json')
            a.check(all(r['run_fingerprint']==prov['run_fingerprint'] for r in final),'generation fingerprint mismatch')
            a.check(all(r['max_new_tokens']==16 for r in final),'QA token setting mismatch')
            a.check(prov['settings']['max_side']==(768 if is_ego else 224) and prov['settings']['temperature']==0,'QA settings mismatch')
            a.check('cc594898137f460bfe9f0759e9844b3ce807cfb5' in prov['settings']['model_id'],'unpinned QA model')
            input_path=Path('/scratch/yy5919/pyro-kd-derived/moments')/(('egoschema-heldout-' if is_ego else 'video-mme-long-')+condition+'-v1.jsonl')
            a.check(sha(a.bind(input_path))==prov['input_sha256'],'QA input provenance mismatch')
            a.check(prov['source_commit'].startswith(('09abb6c',) if is_ego else ('1f5fb60','a7e0e1a')),'unexpected QA source')
            source_repo=args.formal_repo if is_ego else Path('/scratch/yy5919/pyro-revision15-next' if prov['source_commit'].startswith('1f5fb60') else '/scratch/yy5919/pyro-revision15-long-prep')
            for source,digest in prov['source_file_sha256'].items():
                a.check(sha(source_repo/source)==digest, 'QA implementation source changed: '+source)
            predictions[condition]=(p,prov)
            return True
        if not a.attempt(kind+'/'+condition,raw_cell):ready=False
    if not ready:
        a.errors.append(kind+': scoring artifacts deliberately unopened until all raw conditions pass')
        return result
    reports=[]
    for condition in conditions:
        path=root/'evaluation'/(condition+('.evaluation.json' if is_ego else '.json'))
        report=a.obj(path); p,prov=predictions[condition]
        a.check(report['input_sha256']==sha(p) and report['run_fingerprint']==prov['run_fingerprint'],'stale QA score provenance')
        a.check(report['expected']==len(expected) and report['attempted']==len(expected) and not report['status_counts'].get('missing',0),'QA score coverage mismatch')
        a.check({r['id'] for r in report['rows']}==set(expected),'QA score membership mismatch')
        result[condition]['scored_failure_ids']=[r['id'] for r in report['rows'] if r['status']!='valid']
        reports.append((condition,report))
    expected_paired=export_reports(reports, 'egoschema' if is_ego else 'video-mme')
    stats=a.obj(root/'evaluation/comparisons.json')
    paired=root/'evaluation'/('paired.jsonl' if is_ego else 'paired-metrics.jsonl')
    paired_rows=a.rows(paired)
    a.check(sorted(paired_rows,key=lambda r:(r['example_id'],r['condition']))==expected_paired,'paired scores differ from validated reports')
    if is_ego:
        a.check(stats['input_sha256']==sha(paired),'stale paired analysis input')
    comparisons=stats['comparisons']; a.check(len(comparisons)==(3 if is_ego else 13),'paired family incomplete')
    a.check(stats['family_sha256' if is_ego else 'plan_sha256']==sha(planpath),'paired plan hash mismatch')
    a.check(all(('holm_pvalue' if is_ego else 'p_holm') in c['wilcoxon'] for c in comparisons),'Holm correction absent')
    a.check(all(len(c['units'])==(490 if is_ego else 12) and not c['excluded'] for c in comparisons),'paired cluster denominator mismatch')
    return result

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--formal-repo',type=Path,default=Path('/scratch/yy5919/pyro-revision15-formal'))
    p.add_argument('--results-root',type=Path,default=Path('/scratch/yy5919/pyro-resume-results'))
    p.add_argument('--tts-root',type=Path,default=Path('/scratch/yy5919/pyro-kd-derived/tts/formal-local-v1'))
    p.add_argument('--gap-evidence',type=Path,action='append',default=[])
    p.add_argument('--temporal-evidence',type=Path,required=True,help='Detailed temporal exporter JSONL provenance sidecar')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);a=Audit()
    sections={}
    for name,fn in [('ad',lambda:audit_ad(a,args)),('long_qa',lambda:audit_qa(a,args,'long')),('egoschema',lambda:audit_qa(a,args,'egoschema'))]:
        sections[name]=a.attempt(name,fn)
    def temporal():
        obj=a.evidence(args.temporal_evidence)
        a.check(obj['expected_clips']==30 and obj['conditions']==10,'temporal coverage mismatch')
        output=Path(str(args.temporal_evidence).removesuffix('.provenance.json'))
        a.check(sha(a.bind(output))==obj['output_sha256'],'temporal output hash mismatch')
    a.attempt('temporal evidence',temporal)
    manifest=dict(local_experiments_complete=not a.errors,all_proposal_complete=False,
                  expected=dict(ad_core=3398,ad_additional=768,long_qa=360,egoschema=1470),
                  deferred=['Paid API core cells and rubric judging: no authorized budget','Human audio labels and blinded peer ratings: pending user annotations','Cue2Narrate: proposal access gate not met'],
                  limitations=['Long A/AV comparisons have unequal realized frame budgets; no equal-budget claim','Event detector precision/recall and perceptual speech quality are not validated by automated proxies'],
                  source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
                  formal_source_sha=FORMAL_SHA,verifier_sha256=sha(__file__),sections=sections,errors=a.errors,artifact_sha256=a.artifacts)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'local_experiments_complete':not a.errors,'errors':len(a.errors),'manifest':str(args.output)}))
    return int(bool(a.errors))

if __name__=='__main__':
    sys.exit(main())
