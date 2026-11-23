# Compute accounting

Completed experiment snapshot at 2026-11-23T18:43:03.758201+00:00. API spend: **$0**; no paid API calls authorized. All 25 local condition outputs are complete. The agent-owned secondary job has exited; user-owned 40964 remains allocated.

Allocation time, process wall time and recorded model-call latency measure different things and must not be added together.

| Job | State | Allocated GPUs | Elapsed GPU-hours |
|---|---|---:|---:|
| 40964 | RUNNING | 1 | 4.043 |
| 41197 | COMPLETED | 1 | 1.878 |

These two allocations total 5.921 GPU-hours so far, including setup, any idle time and other user activity. This is not exact experiment-attributed compute. The JSON inventories the complete scheduler query window; other user allocations are not attributed to these experiments.

| Secondary completed stage | Process wall seconds |
|---|---:|
| AV-flat-timeline | 486.066 |
| AV-hierarchical-timeline | 335.691 |
| formal-no-context | 632.565 |
| formal-native | 984.640 |
| EgoSchema-U | 2129.478 |
| EgoSchema-V_local | 2157.977 |

Primary process-monitor wall estimates (completion polling uncertainty 15 seconds):

- PID 59758: 5228.711 seconds; command recorded in JSON.
- PID 150806: 355.838 seconds; command recorded in JSON.
- PID 158047: 344.827 seconds; command recorded in JSON.
- PID 159792: 2141.257 seconds; command recorded in JSON.

Primary EgoSchema process completion is recorded in JSON. Process timing excludes queue-driver lock waiting.

| Task | Frozen attempts | Recorded rows | Recorded model-call seconds | Complete cells |
|---|---:|---:|---:|---:|
| REFRAMED AD | 4166 | 4166 | 6481.331 | 12/12 |
| Video-MME long QA | 360 | 360 | 2341.504 | 10/10 |
| EgoSchema QA | 1470 | 1470 | 5777.182 | 3/3 |

The formal AD total is 3,398 core requests plus 768 incremental no-context/native requests; the timeline baseline is reused. Long QA is ten conditions × 36 questions. EgoSchema is three conditions × 490 questions. Failed recorded outputs remain in the latency/count inventory. Per-cell counts, missing-latency counts, completed QA model loads and input snapshot hashes are in the JSON.

Both model wrappers use batch size one. AD call latency includes image loading inside the wrapper; QA call latency excludes preceding frame preparation. These sums omit startup/loading except separately recorded QA load time, other preparation, CPU metrics/TTS and unrecorded failed calls/retries. They are not end-to-end runtime.

Utilization samples are partial-window observations only:

- Job 40964: 230 samples on NVIDIA A100-SXM4-40GB, 2026/11/23 16:48:07.667 through 2026/11/23 18:42:38.538; mean sampled utilization 53.8%, 20 zero samples. These do not establish integrated busy hours.
- Job 41197: 226 samples on NVIDIA A100-SXM4-40GB, 2026/11/23 16:41:02.828 through 2026/11/23 18:33:33.736; mean sampled utilization 51.4%, 33 zero samples. These do not establish integrated busy hours.

No USD estimate is assigned to local compute. All local model runs in this inventory are complete. User-owned 40964 remains allocated; its elapsed GPU-hours are a snapshot, not its eventual allocation total. Controller job 41245 completed without submitting another GPU job. This report does not cancel or modify any job.

Reproduction: run the scheduler command recorded in JSON and `/scratch/yy5919/pyro-compute-accounting-export.py` (driver SHA recorded there). The exporter reads complete JSONL lines from a byte snapshot; any live partial tail is counted as excluded, never repaired. Raw output files and active drivers remain unchanged.
