import subprocess,json,hashlib,datetime,csv,io,re
from pathlib import Path
root=Path('/scratch/yy5919/pyro-revision15-records');base=Path('/scratch/yy5919/pyro-resume-results');sources={}
def snapshot(p):
 p=Path(p);b=p.read_bytes();sources[str(p)]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)};return b
def jsonfile(p):return json.loads(snapshot(p))
def rows(p):
 b=snapshot(p);parts=b.splitlines(keepends=True);partial=bool(parts and not parts[-1].endswith(b'\n'))
 if partial:parts.pop()
 return [json.loads(x) for x in parts],partial
sacct=subprocess.check_output(['sacct','-u','yy5919','-S','2026-11-23T00:00:00','--allocations','--format=JobID,JobName,State,ExitCode,Start,End,ElapsedRaw,AllocTRES','-P'],text=True)
jobs=list(csv.DictReader(io.StringIO(sacct),delimiter='|'))
for j in jobs:
 m=re.search(r'(?:^|,)gres/gpu=(\d+)',j['AllocTRES']);j['allocated_gpu_count']=int(m[1]) if m else 0;j['elapsed_gpu_hours']=int(j['ElapsedRaw'])*j['allocated_gpu_count']/3600
stages=jsonfile(base/'secondary-gpu-41197.json');prep=jsonfile(base/'experiments/reframed_test_local_v1/preparation_counts.json')
cells=[]
for p in sorted((base/'experiments/reframed_test_local_v1/generations').glob('*.jsonl')):
 rr,partial=rows(p);final={r['id']:r for r in rr};key=p.stem
 expected=next(n for k,n in prep['moments_by_selector'].items() if key.startswith('ad-'+k.split('-')[0]+'-'))
 cells.append({'condition':key,'task':'REFRAMED AD','expected':expected,'rows':len(rr),'unique':len(final),'partial_tail_excluded':partial,'latency_field':'latency_s','recorded_model_call_seconds':sum(r.get('latency_s',0) for r in rr),'rows_with_latency':sum('latency_s' in r for r in rr),'complete_unique_coverage':len(final)==expected,'model_load_seconds':None})
for directory,task,expected in [('long-qa-frozen','Video-MME long QA',36),('egoschema-heldout','EgoSchema QA',490)]:
 for p in sorted((base/directory).glob('*.jsonl')):
  if p.name.endswith('.attempts.jsonl'):continue
  rr,partial=rows(p);meta=p.with_suffix(p.suffix+'.attempts.jsonl');attempts=rows(meta)[0] if meta.exists() else []
  cells.append({'condition':p.stem,'task':task,'expected':expected,'rows':len(rr),'unique':len({r['id'] for r in rr}),'partial_tail_excluded':partial,'latency_field':'batch_latency_s','recorded_model_call_seconds':sum(r.get('batch_latency_s',0) for r in rr),'rows_with_latency':sum('batch_latency_s' in r for r in rr),'complete_unique_coverage':len(rr)==len({r['id'] for r in rr})==expected,'model_load_seconds':sum(a['load_seconds'] for a in attempts) if attempts else None,'completed_process_attempt_records':len(attempts),'questions_in_completed_process_records':sum(a['questions_attempted'] for a in attempts)})
util=[]
for job in ['40964','41197']:
 p=Path('/scratch/yy5919')/('pyro-primary-util-40964.csv' if job=='40964' else 'pyro-secondary-util-41197.csv')
 if p.exists():
  rr=list(csv.reader(snapshot(p).decode().splitlines()));valid=[r for r in rr if len(r)==4 and r[0]!='timestamp'];values=[float(r[2].strip().rstrip('%')) for r in valid]
  util.append({'job_id':job,'samples':len(values),'first':valid[0][0],'last':valid[-1][0],'gpu':valid[0][1].strip(),'sample_mean_gpu_utilization_percent':sum(values)/len(values),'zero_utilization_samples':values.count(0),'peak_sampled_memory_mib':max(float(r[3].strip().split()[0]) for r in valid)})
process=subprocess.check_output(['ps','-eo','pid,lstart,etimes,args'],text=True)
process=[line for line in process.splitlines() if re.search(r'python -u scripts/run_qa_vlm.py --input .*/egoschema-heldout-',line)]
primary=jsonfile(base/'primary-process-timing.json')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
complete=all(c['complete_unique_coverage'] for c in cells) and all('completion_observed_utc' in p for p in primary['processes'])
notes=[('Completed experiment accounting snapshot; user-owned 40964 remains allocated and its allocation time is measured only to this snapshot.' if complete else 'Partial accounting snapshot; running allocations and EgoSchema processes continue.'),'Allocation elapsed GPU-hours are not GPU-busy hours or model inference time. Job 40964 is user-owned; no release is authorized by this report.','Only this Slurm query window is inventoried; it is not lifetime project accounting. Unattributed user allocation jobs are listed separately and are not charged to these experiments.','Process wall includes interpreter/container startup, model loading, I/O and inference. Primary wall times are process-monitor estimates with 15-second completion-poll uncertainty. File modification spans are not used as substitutes.','AD latency_s uses batch_size=1 and includes image preparation within the model wrapper; QA batch_latency_s uses batch_size=1 and excludes preceding frame preparation. Neither includes all model loads, cache preparation, failed unrecorded retries, scoring or TTS. Do not sum wall, latency and allocation hours as independent costs.','Recorded row latency is summed over preserved raw attempts, not only successful outputs; missing latency is not evidence of zero runtime. QA model-load metadata is appended only on process completion, so running processes may not yet have it.','Utilization snapshots are sampled observations over limited time windows, not integrated GPU-busy duration.','Paid API calls remain unauthorized; API spend is $0. No monetary estimate for local compute.']
payload={'captured_utc':now,'status':'experiments_complete_allocation_snapshot' if complete else 'partial','api_spend_usd':0,'notes':notes,'scheduler_command':'sacct -u yy5919 -S 2026-11-23T00:00:00 --allocations --format=JobID,JobName,State,ExitCode,Start,End,ElapsedRaw,AllocTRES -P','scheduler_allocations':jobs,'tracked_gpu_hours_40964_41197':sum(j['elapsed_gpu_hours'] for j in jobs if j['JobID'] in ['40964','41197']),'secondary_completed_stages':stages,'primary_process_timing':primary,'active_primary_process_snapshot':process,'frozen_counts':{'formal_core':3398,'formal_ablation5_incremental':768,'long_qa':360,'egoschema_qa':1470},'cells':cells,'utilization_samples':util,'sources':sources,'export_driver':{'path':__file__,'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
out=root/'docs/results';(out/'compute-accounting.json').write_text(json.dumps(payload,indent=2)+'\n')
lines=['# Compute accounting','',f"{'Completed experiment' if complete else 'Partial'} snapshot at {now}. API spend: **$0**; no paid API calls authorized. "+('All 25 local condition outputs are complete. The agent-owned secondary job has exited; user-owned 40964 remains allocated.' if complete else 'EgoSchema V_local and V_memory are still running.'),'', 'Allocation time, process wall time and recorded model-call latency measure different things and must not be added together.','', '| Job | State | Allocated GPUs | Elapsed GPU-hours |','|---|---|---:|---:|']
for j in jobs:
 if j['JobID'] in ['40964','41197']:lines.append(f"| {j['JobID']} | {j['State']} | {j['allocated_gpu_count']} | {j['elapsed_gpu_hours']:.3f} |")
lines += ['', 'These two allocations total %.3f GPU-hours so far, including setup, any idle time and other user activity. This is not exact experiment-attributed compute. The JSON inventories the complete scheduler query window; other user allocations are not attributed to these experiments.'%payload['tracked_gpu_hours_40964_41197'],'','| Secondary completed stage | Process wall seconds |','|---|---:|']
for s in stages['stages']:
 if 'seconds' in s:lines.append(f"| {s['name']} | {s['seconds']:.3f} |")
lines+=['','Primary process-monitor wall estimates (completion polling uncertainty 15 seconds):','']
for proc in primary['processes']:
 if 'process_wall_seconds_approx' in proc:lines.append(f"- PID {proc['pid']}: {proc['process_wall_seconds_approx']:.3f} seconds; command recorded in JSON.")
lines+=['',('Primary EgoSchema process completion is recorded in JSON. Process timing excludes queue-driver lock waiting.' if complete else 'The active primary EgoSchema process snapshot is retained in JSON; elapsed process time is a lower bound until exit, not the queue-driver age (which includes lock waiting).'),'','| Task | Frozen attempts | Recorded rows | Recorded model-call seconds | Complete cells |','|---|---:|---:|---:|---:|']
for task,expected in [('REFRAMED AD',4166),('Video-MME long QA',360),('EgoSchema QA',1470)]:
 cc=[c for c in cells if c['task']==task];lines.append(f"| {task} | {expected} | {sum(c['rows'] for c in cc)} | {sum(c['recorded_model_call_seconds'] for c in cc):.3f} | {sum(c['complete_unique_coverage'] for c in cc)}/{len(cc)} |")
lines+=['','The formal AD total is 3,398 core requests plus 768 incremental no-context/native requests; the timeline baseline is reused. Long QA is ten conditions × 36 questions. EgoSchema is three conditions × 490 questions. Failed recorded outputs remain in the latency/count inventory. Per-cell counts, missing-latency counts, completed QA model loads and input snapshot hashes are in the JSON.','','Both model wrappers use batch size one. AD call latency includes image loading inside the wrapper; QA call latency excludes preceding frame preparation. These sums omit startup/loading except separately recorded QA load time, other preparation, CPU metrics/TTS and unrecorded failed calls/retries. They are not end-to-end runtime.','','Utilization samples are partial-window observations only:','']
for u in util:lines.append(f"- Job {u['job_id']}: {u['samples']} samples on {u['gpu']}, {u['first']} through {u['last']}; mean sampled utilization {u['sample_mean_gpu_utilization_percent']:.1f}%, {u['zero_utilization_samples']} zero samples. These do not establish integrated busy hours.")
lines+=['',('No USD estimate is assigned to local compute. All local model runs in this inventory are complete. User-owned 40964 remains allocated; its elapsed GPU-hours are a snapshot, not its eventual allocation total. Controller job 41245 completed without submitting another GPU job. This report does not cancel or modify any job.' if complete else 'No USD estimate is assigned to local compute. Update this snapshot after both EgoSchema runs finish and the agent-owned secondary job exits; user-owned 40964 may remain allocated. This report does not cancel or modify any job.'),'', 'Reproduction: run the scheduler command recorded in JSON and `/scratch/yy5919/pyro-compute-accounting-export.py` (driver SHA recorded there). The exporter reads complete JSONL lines from a byte snapshot; any live partial tail is counted as excluded, never repaired. Raw output files and active drivers remain unchanged.']
(out/'compute-accounting.md').write_text('\n'.join(lines)+'\n');print(now,len(cells))
