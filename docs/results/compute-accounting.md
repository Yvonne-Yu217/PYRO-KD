# Compute accounting

Partial snapshot at 2026-11-23T18:16:17.377801+00:00. API spend: **$0**; no paid API calls authorized. EgoSchema V_local and V_memory are still running.

Allocation time, process wall time and recorded model-call latency measure different things and must not be added together.

| Job | State | Allocated GPUs | Elapsed GPU-hours |
|---|---|---:|---:|
| 40964 | RUNNING | 1 | 3.596 |
| 41197 | RUNNING | 1 | 1.588 |

These two allocations total 5.184 GPU-hours so far, including setup, any idle time and other user activity. This is not exact experiment-attributed compute. The JSON inventories the complete scheduler query window; other user allocations are not attributed to these experiments.

| Secondary completed stage | Process wall seconds |
|---|---:|
| AV-flat-timeline | 486.066 |
| AV-hierarchical-timeline | 335.691 |
| formal-no-context | 632.565 |
| formal-native | 984.640 |
| EgoSchema-U | 2129.478 |

Primary process-monitor wall estimates (completion polling uncertainty 15 seconds):

- PID 59758: 5228.711 seconds; command recorded in JSON.
- PID 150806: 355.838 seconds; command recorded in JSON.
- PID 158047: 344.827 seconds; command recorded in JSON.

The active primary EgoSchema process snapshot is retained in JSON; elapsed process time is a lower bound until exit, not the queue-driver age (which includes lock waiting).

| Task | Frozen attempts | Recorded rows | Recorded model-call seconds | Complete cells |
|---|---:|---:|---:|---:|
| REFRAMED AD | 4166 | 4166 | 6481.331 | 12/12 |
| Video-MME long QA | 360 | 360 | 2341.504 | 10/10 |
| EgoSchema QA | 1470 | 825 | 3265.727 | 1/3 |

The formal AD total is 3,398 core requests plus 768 incremental no-context/native requests; the timeline baseline is reused. Long QA is ten conditions × 36 questions. EgoSchema is three conditions × 490 questions. Failed recorded outputs remain in the latency/count inventory. Per-cell counts, missing-latency counts, completed QA model loads and input snapshot hashes are in the JSON.

Both model wrappers use batch size one. AD call latency includes image loading inside the wrapper; QA call latency excludes preceding frame preparation. These sums omit startup/loading except separately recorded QA load time, other preparation, CPU metrics/TTS and unrecorded failed calls/retries. They are not end-to-end runtime.

Utilization samples are partial-window observations only:

- Job 40964: 177 samples on NVIDIA A100-SXM4-40GB, 2026/11/23 16:48:07.667 through 2026/11/23 18:16:08.337; mean sampled utilization 54.6%, 11 zero samples. These do not establish integrated busy hours.
- Job 41197: 191 samples on NVIDIA A100-SXM4-40GB, 2026/11/23 16:41:02.828 through 2026/11/23 18:16:03.600; mean sampled utilization 48.9%, 30 zero samples. These do not establish integrated busy hours.

No USD estimate is assigned to local compute. Update this snapshot after both EgoSchema runs finish and the agent-owned secondary job exits; user-owned 40964 may remain allocated. This report does not cancel or modify any job.

Reproduction: run the scheduler command recorded in JSON and `/scratch/yy5919/pyro-compute-accounting-export.py` (driver SHA recorded there). The exporter reads complete JSONL lines from a byte snapshot; any live partial tail is counted as excluded, never repaired. Raw output files and active drivers remain unchanged.
