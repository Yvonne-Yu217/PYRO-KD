# Agent Log

## 2026-11-22 — initial audit
Human supplied the experiment scope, proposal, repository and Torch data paths; authorized ordinary Slurm GPU jobs, but no paid API budget.
Agent read the attachment, repository README, week1 specification and HPC documentation; fetched origin/main; inspected Slurm, GPU and filesystem; read the linked proposal tab through Google Drive and course page through curl after web retrieval failed.
Verified clean starting worktree and HEAD = origin/main = 3139be9be2ef52f498b5db5fecfdfa5618da03b0.
Mistake: initially called this allocation Torch before checking cluster identity. Corrected: scontrol reports slurm-cloud (Cloud Bursting), node b-19-7; supplied Torch directory is absent. Initial default-Python test failed because pytest is absent; retrying existing Singularity environment.
No experiments or measured scientific findings yet. No data or user jobs modified.

## 2026-11-22 — baseline tests and scope audit
Existing container Python also lacked pytest. Created an isolated venv with system-site-packages at /scratch/yy5919/pyro-test-env using the existing read-only Singularity overlay; installed pytest 9.2.0 and iniconfig 2.3.1. No shared overlay mutation. Started baseline tests with output persisted under outputs/verification/pytest-baseline.log.
NYU official connection documentation identified login.torch.hpc.nyu.edu (earlier guessed hostname was incorrect). Correct hostname responds, but SSH rejects a changed host key. Did not delete or bypass existing trusted keys; requested authenticated Torch access. User dataset counts remain unverified.
Audited implementation against proposal and wrote PROPOSAL_1_5_AUDIT.md; identified absent AV, P1 and QA inference plus statistical/rendered-timing gaps rather than treating existing baseline code as proposal completion.
Baseline suite exited 0: 394 tests and 5911 subtests passed in 22.93s. Verified persisted log. This does not establish real-data GPU smoke success.

Implemented paired JSONL analysis with movie aggregation, exclusions and provenance. Consulted official SciPy Wilcoxon documentation for ties/zeros and permutation method. Four tests passed, including analytically known 2/32 two-sided probability and unequal cluster-size weighting. No real metrics analyzed. User subsequently directed retrying SSH using NYU instructions and NetID yy5919.

## 2026-11-22 — SSH authentication and comparison families
Read NYU's official connection instructions in full after the user's direction. Their documented host-key handling is for multiple login nodes. Retried using per-command options (no persistent SSH configuration changes). One node offered only GSSAPI; another attempt reached keyboard-interactive Microsoft device authentication. Session 70487 is awaiting human MFA; no credentials saved in logs. Previous assertion that host-key mismatch necessarily blocked further progress was overly conservative given the official instructions.
Added scripts/analyze_family.py and Holm correction to preserve declared family size even when comparisons lack data. Distinguishes pointwise confidence intervals from family-corrected p-values. Verification: 41 tests and 11 subtests passed in 3.94s, outputs/verification/pytest-statistics.log. These are software checks, not experiment results.

## 2026-11-22 — material proposal correction
Full Google Docs structure revealed a separate tab titled Revision 1.5 (t.ickd261ntenp). The URL points to the older Reframed and EgoSchema tab. Agent mistake: assumed the supplied URL tab was Revision 1.5 and failed to inspect other tabs initially. This omitted long-video QA, Cue2Narrate, three audio streams, signed event salience, running novelty memory, hierarchy, eight ablations and human detector sanity labels. Corrected by saving full authoritative text and writing REVISION_1_5_REQUIREMENTS.md; marked previous audit historical. No experiment outputs exist, so no measured findings are contaminated by this mistake.
User confirmed MFA complete, but sending ENTER to session 70487 returned connection closed (exit 255). Started replacement session 62852 and reached a fresh device prompt; user must authenticate that session. Authentication codes are not persisted in repo.
Added standalone gap-snapping primitive with explicit advance/lag and word capacity, tested 10 cases successfully (0.08s); not yet integrated or real-data verified. User directed prioritizing data access and locating prior MTG smoke. Session 62852 closed on ENTER; new session 10280 started with connection multiplexing for reuse after authentication. Local checked MTG media directories are empty; found manual-paraphrase evaluation smoke, not yet evidence of real VLM/video smoke.

## CPU data-preparation phase
User allocated a CPU node and explicitly prioritized completing data downloads before GPU work. Verified old job 40006 cancelled and current job 40036 running on b-10-63; prior SSH handles disappeared and socket refused connections. Restarted SSH from CPU node, session 45555, awaiting MFA.
Search mistake: rg obeyed .gitignore and skipped dataset directories. Corrected with --no-ignore, finding 51 local EgoSchema MP4s and REFRAMED annotation files. All 51 probed successfully without audio using existing container ffprobe. Initial probe failed because host PATH lacked ffprobe; reran in existing container. Preserved source files and wrote inventory evidence to outputs/verification/. This is file/stream verification, not complete decoding or experiment success.
MFA succeeded on CPU node; remote hostname and dataset listing verified. Read source README and download summary. Started metadata rsync and queued resumable full source transfer with exclusive lock, reusing 51 existing EgoSchema files. Source roots remain read-only. A du call raced a temporary rsync filename; this is not evidence of transfer failure. Source file totals are reported records, awaiting independent destination validation.
Metadata transfer completed; full video transfer began on CPU allocation. Added scripts/verify_transferred_dataset.py to compare evaluation dataset manifests, source snapshot totals, local IDs/sizes and optional ffprobe duration/stream signatures. Ran on partial EgoSchema destination to explicitly establish incomplete coverage. This audit does not claim hashes or full decode. User asked about idle shutdown; public NYU docs do not establish CPU idle policy, so no keepalive guarantee was made. Confirmed six-hour allocation and persistent storage.
Confirmed transfer direction Torch -> Cloud Bursting to user. Verified all ten configured MTG dev IDs are source-ready, correcting implications of older failed download reports. Queued real SHA256 verification after MTG stage completion; no checksum result claimed yet. Remote repository has the expected source SHA, but inspected directories do not establish historical real GPU smoke evidence.
Queued evaluation video coverage/ffprobe checks after each stage, with live transfer-process guards; tests verified incomplete transfers cannot pass. No source content or official splits changed. Recorded original source failure list separately from local missing-transfer IDs.
Performed metadata-only annotation/availability reconciliation without inspecting held-out answers for tuning. First audit attempt used str.splitlines(), which splits Unicode separators inside MTG JSON strings and raised JSONDecodeError. Corrected to newline-only splitting, matching existing dataset loader behavior. Source mapping audit then succeeded; source missing REFRAMED IDs are all training clips. No metadata or split changes.
Monitored live transfer/checksum processes through MTG completion. sha256sum exited 0 with 413 OK lines; local IDs and sizes separately reconciled against source manifest. All 413 available MTG videos verified. No experiment success inferred from transfer integrity. EgoSchema/REFRAMED still unfinished.
Loaded verified MTG source through existing dataset loader and preprocessed exactly the ten configured development videos on the allocated CPU node, in parallel with network transfer. Used existing implementation, recorded split hash/source SHA, and kept generated caches outside raw source. All ten report ready with audio. This is CPU media preparation, not a GPU model smoke test.
Existing CPU Silero/librosa extraction succeeded for all ten real MTG dev clips. Built and inspected baseline U smoke inputs: 78/189 requested moments emitted, 111 excluded by zero-word policy; all emitted records have four readable-path frames and positive budgets. This exposes a substantial generation-coverage shortfall that must stay visible in final protocol; did not call inputs a completed smoke experiment or relax fixed-K requirements. Historical cap30/gap policy remains an explicit fidelity issue.
While network transfer remained active, checked Revision 1.5 dataset gates using primary paper, official GitHub and official Video-MME README. Cue2Narrate is a method on LongLSMDC, not itself the dataset name; official linked repo currently has no data/code beyond placeholder README/license/ignore file. Recorded provisional access/comparability gaps without declaring every route exhausted. No new bulk download or user messaging to third parties.
Caught REFRAMED layout incompatibility before inference: transferred root test.jsonl is decrypted, but current loader always decrypts test. Built a derived view pointing to preserved encrypted source annotations and extracted only known dataset-key archive members outside the raw root. All three real annotation loaders returned official expected counts. No tuning on held-out content or modification of source data; these reads validate parsing/counts only.
Promoted resumable transfer and dataset-view setup from session artifacts into repository scripts, documented exact reproduction commands. bash -n passed; dataset-view script reran successfully against actual existing views without replacing source files. Did not launch a duplicate transfer; original rsync continues under its lock.
Confirmed EgoSchema successful transfer-stage marker and byte statistics, then observed the real ffprobe verification process running. REFRAMED copying concurrently; variable throughput is not a failure and no restart performed. No final data-ready claim yet.
Observed EgoSchema verifier reaching file 4409/5031, then received successful result from live watcher and inspected the final JSON. All 5031 IDs/sizes/durations/streams match, with zero audio streams; no missing or unexpected files. Marked EgoSchema dataset transfer verified at that scope, without claiming full decoding or SHA256 verification. Recorded visual-only gate decision per Revision 1.5.

## 2026-11-22 — current status clarity and live transfer check
Rechecked live process table: transfer supervisor 10131, REFRAMED rsync 20258 and verification watcher 11171 are running. Slurm confirms CPU allocation 40036 has over five hours remaining. Reopened completed MTG and EgoSchema reports; both complete=true. REFRAMED log advanced to 11%; verification remains pending. Added a current checkpoint above chronological experiment history because the original opening paragraph still described the cancelled GPU allocation and wrong proposal tab. Historical entries are preserved, explicitly labeled, and superseded by the checkpoint. Documentation-only change; no extra transfers, GPU jobs, or source-data mutations.

## 2026-11-22 — offline preparation for real-data smoke
CPU dry run on four prepared real MTG moments exited 0 and opened frames/built prompts. Checked actual model caches: default scratch cache has metric encoders, while home cache contains Qwen2.5-VL-7B revision cc594898137f460bfe9f0759e9844b3ce807cfb5 with all five indexed weight shards present. Offline AutoConfig/AutoProcessor loading passed in the existing container (session 76589 exit 0). Reconciled model choice with authoritative proposal: Qwen2.5-VL-7B fits; default Muse is not the required API condition. Prepared docs/REAL_DATA_SMOKE.md with exact snapshot and later GPU command. No weights downloaded, no GPU jobs submitted, no inference/quality result claimed. Observed long transcript contexts and existing U exclusions remain protocol issues before matrix scaling. REFRAMED copy continues under the same live session.

## 2026-11-22 — real multimodal input encoding check
Ran CPU-only Qwen2.5-VL processor on four actual MTG moment frame sets/prompts in two batches, with the same chat-template tensor/padding options as generate_batch. Session 11471 exited 0; report mtg-real-processor.json has complete=true and both batches contain eight encoded images. Logged tensor shapes, nonpadding token counts, code SHA and input hash. No model weights loaded or GPU inference claimed. Existing transfer remains live at about 22%; instantaneous throughput decreased but bytes continued advancing.

## 2026-11-23 00:07 UTC — all three dataset transfers verified
Transfer session 49851 exited 0; log records REFRAMED completion at 00:04:54 UTC with 1983 files and 199,785,421,068 media bytes. Automatic verifier watcher 73417 subsequently exited 0. Reopened final report: complete=true, expected=verified=1983, missing/unexpected/errors/snapshot_errors empty; 40 source-missing IDs retained. Rechecked MTG and EgoSchema complete flags and updated current status above historical checkpoints. No raw source mutation, fabricated media or GPU submission. Transfer monitoring used live process/session checks throughout; no duplicate transfer was launched. Verification limits remain explicit: hashes for MTG, manifest sizes and ffprobe streams/durations for evaluation media. One monitoring command used tail -n on a carriage-return progress log and produced oversized output; switched back to bounded tail -c.

## 2026-11-23 — P1 implementation during GPU-session handoff
Implemented P1 as a researcher-written guideline summary plus three professional REFRAMED training AD examples, retaining the exact P0 current-video suffix. Added deterministic exemplar preparation excluding every validation/test movie, saved source-file hashes/IDs/times, and froze bundle a51d9076f3dfeb1bb50b1c209aba3b3e2a789843c98eb7dcb5aeaba4db724faf outside raw data. Generator requires P1 bundle, logs its hash, and rejects cross-prior/exemplar resume mixing. Corrected prior requirements-inventory error: actual Revision 1.5 says professional examples, not validation examples. Training-only is a documented implementation choice. Targeted tests passed 63 tests/17 subtests; four real MTG P1 dry-run inputs passed. No GPU submission or model-quality claim. Commands/provenance in docs/PROMPT_PRIORS.md.

## 2026-11-23 — align word budgets with Revision 1.5
Removed the implicit 30-word cap from word_limit_for_gap; known gaps now use floor(gap*3), with historical cap available only by explicit argument. Moment and prediction records retain gap seconds and budget-rule version. Rebuilt MTG U dev into a distinct revision15 file, preserving the old input: all 78 rows have unchanged IDs/times/frames, 53 budgets changed, max upper limit793, and 111 zero-budget exclusions remain. First targeted run caught one remaining test expecting historical cap30 for a15-second gap; corrected expected45 and added row provenance assertions. Final targeted tests:69 passed/794 subtests. Four new-budget actual MTG inputs passed P0 CPU dry run. Evidence mtg-revision15-word-budget.json and pytest-revision-word-budget.log; no GPU inference or claim of fixed-K protocol completion.

## 2026-11-23 — primary-benchmark development CPU preparation
Added scripts/preprocess_reframed_dev.py to preprocess the existing frozen validation selection without downloading metadata, rewriting splits or touching raw videos. Actual run session56368 exited0:10/10 current complete caches,2663 frames, all audio present, maximum annotation-duration drift0.043s. Existing Silero/librosa cache run session32803 exited0:10 computed, no failures/no-audio/aborted cases. Built U moments session31194 exit0:66 valid inputs across10 videos,45 zero-budget exclusions from111 requested. Checked all66 have four existing frames and exact floor(gap*3) positive budgets. These are CPU pipeline artifacts, not results or completion of the three-stream stack or fixed-K comparisons. All evidence lives under outputs/verification/reframed-dev-* and derived artifacts under /scratch/yy5919/pyro-kd-derived. User CPU session remains40036; no GPU submission.

## 2026-11-23 — pinned audio models and offline CPU compatibility
Read official faster-whisper/Transformers/AST/CLAP sources, checked installed packages, queried Hugging Face model revisions/files, and pinned Whisper-small.en, AST AudioSet and CLAP unfused in configs/audio_models.json. Download session1105 exited0 and all expected file sizes match. Separate system-site-packages venv install65832 exited0 (faster-whisper1.2.1; shared overlay untouched). First offline check3984 passed AST/CLAP but failed my assumption that l5lLQICPOcnb had a VAD interval >=2s. Inspected only dev VAD durations and chose fixed dev CxWby8RuUwU6 for the speech check. Corrected run68780 exited0: AST[1,527] and CLAP[1,512] finite outputs;10 Whisper words confined to VAD[0.5,3.3]. CLAP audio-only load has unused text weights; its projected embeddings need normalization for novelty. Script and reproduction in scripts/check_audio_models.py and docs/AUDIO_STACK.md. No GPU, paid API, model-quality claim or fabricated detector labels. Full cached three-stream extraction remains incomplete.

## 2026-11-23 — complete REFRAMED development VAD-gated ASR
Implemented disjoint VAD sample-window transcription and a pinned/hash-keyed CPU Whisper cache. Tests prove silence samples never reach Whisper, long intervals split without bridging gaps, absolute word times restore correctly, out-of-window predictions stay rejected rather than clamped, and zero speech never calls ASR. Tests4 passed/4 subtests. Actual run22477 exited0 for all10 REFRAMED dev clips:1645 words, zero rejected timestamps; eFTqT0_Z632j has zero VAD-positive windows (no claim of true silence). Independent saved-cache verification checked all word/window bounds and exact WAV hashes; code file hashes saved in reframed-dev-speech-check.json. Repeat10686 exited0 and reused all10 caches. No GPU/API, no ASR accuracy claim; event tracks and downstream integration remain pending.

## 2026-11-23 — event-track implementation and pilot
Implemented centered10s/1s-hop AST+CLAP CPU caching, unit-normalized CLAP features and past5-window novelty, positive per-class score changes, and1s RMS dB-rise cues. Retain all class scores but mask10 Speech-descendant/Whispering/Silence labels from onset scoring; checked official AudioSet ontology. Tests3 passed for time grids/edge padding, class masking, cosine normalization and past-only memory. Actual pilot16928 exited0; independent10674 check passed74x527/74x512 shapes, finite values, embedding norms, NPZ hash and all10 masks. Started full dev cache session26247 under existing CPU allocation, reusing pilot; no duplicates or GPU jobs. Explicit limitation:10-second classifier windows are not validated frame-precise events; human detector gate remains pending.

## 2026-11-23 — event-cache audit during live extraction
Added scripts/verify_event_cache.py to audit frozen dev coverage, WAV/source currency, model revisions/vocabulary, config/NPZ hashes, feature shapes/ranges/normalization, one-second timing and recomputed RMS/derived cues. Initial audit verified the pilot and explicitly reported unfinished files; extraction26247 remained live. Process50117 used about3.3 CPU cores and1.8GB RSS with ample memory. Second video HfOMXYYPG5nx completed160 windows; repeat audit run23282 checks completed files without changing them. Full cache remains in progress; no restart, duplicate extraction or GPU submission.

## 2026-11-23 — full REFRAMED development event cache verified
Monitored live extraction26247 through exit0 without duplicate submission. All10 frozen development clips have event caches, totaling1336 time points. Full audit72375 exited0/complete=true/verified10: source/WAV/model/vocabulary/shape/time-grid/hash/numerical checks passed. Repeat66969 exited0 and all10 statuses were current, so no model re-inference was needed. These are consistency checks rather than detector-accuracy validation; no human labels fabricated, and no downstream selector/MLLM results claimed. Updated docs/AUDIO_STACK.md with exact evidence and remaining work.

## 2026-11-23 — audio timeline prompt integration
Implemented explicit legacy/none/timeline audio-text modes, shared across P0/P1; timeline mode excludes the old annotation transcript and requires explicit timeline data before model load. Added mode/timeline-hash resume guards. Attachment script validates WAV/VAD stream consistency and event NPZ hashes, derives thresholded local class rises and nearby gated ASR phrases, and records context/truncation/source provenance without changing selected frames/times/budgets. Current development thresholds0.5/0.1 are explicitly exploratory, not validated or tuned. Actual66 rows preserved all original fields;129 speech entries/92 event estimates,0 omitted entries. Targeted68 tests/17 subtests passed; real four-input P0 and P1 CPU dry runs passed. No GPU/QA/scientific-quality claim.

## 2026-11-23 — quantify placement shortfalls before selector changes
Added development-only audit_gap_feasibility.py using existing frozen Uniform requests and gap-snapping implementation. Actual REFRAMED67458 and MTG93133 runs exited0 across15 exploratory capacity/bound combinations. At capacity>=1 word and symmetric2s bounds, only76/111 REFRAMED and93/189 MTG requests are individually feasible;10s bounds still leave shortfalls. Greedy nearest-placement separation is explicitly not an optimal assignment proof. Used cached usable gaps with margins, distinct from old untrimmed builder; no input changes, unbounded shifts or selector-specific K reduction. Recorded results and unresolved fixed-budget protocol implications in docs/PLACEMENT_FEASIBILITY.md. No GPU experiments or quality conclusions.

Generation provenance preparation: agent added exact source/input/frame/settings fingerprints, strict resume compatibility, pre-load durable provenance, and an early return when no moments are pending. Used shell/apply_patch, pytest and a four-input REFRAMED timeline dry run; 32 tests passed and dry run exited0. First test command incorrectly invoked the container-only Python symlink from the host and failed with missing interpreter; reran inside the existing read-only overlay successfully. No environment changes or raw-data writes. Model-weight hashing and resolved remote revisions are not covered; actual experiments must use a pinned snapshot. User has not answered the optional fixed-K placement question; it remains unresolved. GPU hold remains in effect.

Running visual memory: read Revision1.5 sections3/5 and inspected existing selector/builder. Agent implemented optional memory ranking and connected it to the V builder with recorded mode; historical adjacent-frame ranking retained for ablation. A repeated-scene test confirms selection changes as memory grows; scaling, shot bonuses, separation, shortfalls and invalid embeddings tested. Validation:65 tests/49 subtests (session98942),136 tests/844 subtests (80790), all passed. Initial test command referenced a nonexistent test_visual_features.py and ran no tests; corrected to actual test_visual.py. No real visual cache exists under derived dev caches, so no real visual ranking claim. Existing post-selection gap filtering remains a documented integration limitation; not a finished Revision1.5 V or AV comparison.

Real visual CPU pilot: inspected existing caches and official HF model metadata, pinned OpenCLIP revision, downloaded safetensors only (session9315 exit0) and extracted146 embeddings/19 shots on dev clip CxWby8RuUwU6. Tests83621 passed65/49 subtests. Initial model-info query failed because the container defaults offline; resolved metadata using explicit host HTTPS and download with HF_HUB_OFFLINE=0. Pilot91825 extracted successfully but verification asserted because process_video returns a pre-publication features record without the embedding reference. Corrected checker to reload validated committed features;49106 exit0 reused cache, verified shape/hash and generated local/memory K6 selections. Preserved initial failure log. No accuracy claim, GPU jobs or raw-data mutation.

Full REFRAMED dev visual preparation: CPU extraction93892 exit0 (9 computed,1 reused,0 failures), final audit30957 exit0 verified10/10,2663 embeddings and386 shots. Partial audit91929 exited1 with3/10 while extraction remained live; no process restart. Added scripts/verify_visual_cache.py to check frozen-ID coverage, current source/config, embedding hashes/normalization, float16-tolerant reconstructed novelty and ranking invariants. Initial audit drafting assumed a top-level duration; inspected manifest before execution and corrected to media_info.duration. Exploratory V builders19822 exit0: memory82 rows/local75, each9videos, from111 requests;29/36 zero-word exclusions. Independent row/frame/budget audit passed. No fixed-K accuracy claim; post-selection filtering still unresolved. GPU hold unchanged.

Two-stage selection: implemented independent feasible-placement ranker and development diagnostic. Memory is updated only from selected feasible candidates; signed base scores and full rejection accounting supported. Tests71330 exit0:22passed (placed selection and gap snapping). Real10-clip audit67412 exit0 produced97/111 moments for both local/memory,14shortfall each under exploratory1word/±2s settings. Independent verification checked all20 video/mode cases for exact word capacity, placement bounds,2s separation and exhaustive disjoint candidate accounting. No reference labels used in ranking, no weights tuned, no GPU generation. The optional final fixed-K policy remains unresolved and no protocol was silently frozen.

Speech cue CPU preparation: added gate-bounded transcript chunking, frozen lexical deictic phrases, adjacent text semantic-change tracks and VAD long-silence cues. Eight unit tests passed84673. Used existing pinned MPNet snapshot offline; no download or GPU. Run80086 exit0,10clips294chunks18silencecues0deicticmatches. Independent audit86766 exit0 verified source hashes, frozen IDs, gate bounds, embedding norms/dimensions and cosine scores. No threshold expansion to force matches; no diarization or accuracy claim. Tests cover empty ASR not inventing silence, word/gate limits, substring false matches and malformed vectors. Reproduction and limitations in docs/AUDIO_STACK.md.

A/AV development integration: agent implemented explicit component scoring and audio-only versus fused candidate sets; global signed event role plus optional per-label overrides, speech/silence class mask, separate cue provenance and running visual memory. Tests83827 exit0 (18passed) include A independence from deliberately invalid visual inputs. Real audit33362 exit0: all four A/AV trigger/suppress conditions97/111,14shortfall. Independent40-case audit passed score reconstruction, placement bounds, exact budgets, separation and candidate accounting. Sign flips changed event selections on all10clips for each mode; no accuracy/hypothesis claim. Missing GF/OP, onset-strength, hierarchy and final generation integration documented rather than relabelled complete. No GPU inference or test-split tuning.

Placed generation integration: agent added immutable exporter with explicit frame-center policy, source hash/VAD/media validation and requested-budget accounting. Corrected prompt timestamps and audio context center to use frame-center time when it differs from narration placement; prediction records retain both times and selection provenance. Tests70547 exit0:35passed including timeline and truncation regressions. Export56829 exit0:four97row conditions; preparation31569 exit0:four timelines and8real-input P0/P1 dry runs. Independent388row audit passed; all10devclips represented in every condition. No annotation text or GPU inference. Event-centered frames remain an explicit exploratory choice, not a silent frozen final protocol.

Exploratory dev placement evaluation: implemented report-based REFRAMED evaluator, per-video CSV, raw one-to-one matches, tolerance sweep, bootstrap and three paired differences. Verified ten distinct movie IDs before video bootstrap. Four metric-accounting tests passed91330. Evaluation25290 exit0; independent audit verified all60records/sixsummaries (match identity/tolerance/counts and summary arithmetic). Plot47100 exit0, visually inspected. MacroF1 spans0.330–0.461; all three paired intervals include0. No test annotations, final superiority claim, GPU inference or protocol tuning. One shell lookup used guessed nonexistent statistics.py path; rg located existing stats.py before reuse.

Hierarchical mechanics: implemented explicit content-boundary segmentation, density-based exact integer quotas with quiet floor/capacity reporting, and globally shared memory/separation under optional segment quotas. Tests4597 exit0:25passed, including200randomized budget cases. Short-dev audit13917 exit0:10clips55segments111allocated92selected; independent audit passed contiguous/max-length segments, quota conservation/floor, per-segment counts, global separation and shortfall. No long-video result claim. Asked user asynchronously whether their GPU session is active because only CPU allocation is known; no response yet and GPU hold remains. No processes remain active from this step.

Long-form access work: inspected official Video-MME README and linked LMMS HF release; pinned revision ead1408f75b618502df9a1d8e0950166bf0a2a0b.20archives total100994322505bytes; downloaded405167byte metadata only. Created isolated pyro-data-env with pyarrow26.0.1 after existing container lacked pyarrow; shared overlay unchanged. Froze12long evaluation IDs,2/domain by SHA256(seed0:video_id), reading only metadata columns. Initial web viewer exposed short-video examples; no eligible long questions/answers inspected. Initial ZIP index run58689 exited1 on network timeout. Added exact-range validation, bounded retries and per-archive atomic checkpoints; replacement85580 is active, sixteen indexes/760members cached at last check. Four range-reader tests6555 passed, including refusal of ignored ranges and retry behavior. Eleven selected IDs located so far (~2.58GB compressed); no video payloads/audio verification yet. Do not duplicate live85580.

Inventory85580 is now terminal exit1:16indexes760members with81514bytes transferred; archives17–20 surfaced BadZipFile, not enough evidence to call the release corrupt. Direct17probe9504 returned URLError. Found Python ZipFile may reinterpret OSError transport failures as bad ZIP; wrapped exhausted transport failures as RuntimeError so future diagnostics preserve cause. Tests91698 passed5. Partial map saved selected-archive-members.json:11/12IDs located,2581429970compressedbytes;864unlocated. No video payload extraction or audio claim. All handles from this access step are terminal.

Long-video payload pilot: implemented resumable compressed-range member extraction, exact header/member identity, size/ZIPCRC/SHA checks and ffprobe. Pilot13967 downloaded857 but exited1 because host ffprobe missing; container13676 reused file, exit0, actual33.15min audio+video verified. Tests44236 passed10 including partial-byte resume and refusing corrupt payload publication. Started full located subset download48497 with2workers and incremental report, still active. A fresh official17resolver query succeeded; updated range requests to refresh resolver URLs and launched missing-index recovery33675, reusing16durable checkpoints. No corruption claim or missing-ID replacement. Both jobs remain active; no GPU inference.

Long-media verification/resume: added independent frozen-member checksum/SHA/stream/duration audit with optional3x10s decoded audio probes. Partial92216 exit1 as expected with4/12available; all4had valid nonzero decoded audio and33–49min lengths. Confirmed primary48497 remains live and its completed futures for604/710/729failed,864unlocated; started sequential recovery36189 exclusively for those4IDs using complete map, while primary handles other IDs. Recovery710 succeeded from saved bytes,33.03min AAC/H264. No duplicate writer or restart of a live file task. Both handles still active; full final gate remains unproven.

Transfer continuation: primary48497 and recovery36189 revalidated live; verified-file count advanced7 then604recovered. Confirmed729/891 prior per-file tasks terminal before registering new sequential recovery26985 for only those IDs, keeping active network workers<=3. Its729attempt again failed transport; partial bytes remain. Added per-attempt public resolver nonce so even same-range retries obtain fresh redirects; exact206 Content-Range validation remains mandatory. Tests89097 passed10 including same-byte-range/different-resolver-URL retry behavior. This mitigates stale redirect caching but does not prove the cause of all network failures. Current writers763(primary),864(recovery1),891(recovery2) must not be duplicated. Final all12audit pending.

Long-video acquisition completed: revalidated prior48497/36189/26985 terminal, consolidated61556 recovered729/763 but failed864after453MB. Single-ID73718 resumed864to630054955compressed bytes and exited0. Independent final audit94631 exited0 verified12/12,size/CRC/SHA,30.518–50.501min streams and36nonzero10s decoded audio windows;7.715hours total,3221846667bytes. Reconciled final download report against audit while preserving failure-attempt history. Added reproducible frozen-ID member mapper; tests2696 passed3 and actual12member mappings matched exactly. No questions/answers, test performance, GPU inference or subset replacement. All download/audit handles terminal; long-form data availability gate now passed, experiment scope remains incomplete.

Onset cue integration: added separate cheap spectral-flux cache, explicit opt-in fused score term and source hash propagation. CPU18727 exit0 cached10clips1336windows; tests22770 exit0 (9passed). Exploratory fused run63875 exit0; independent29845audit validated source WAVs/grid/normalization and selection changes. After adding onset hash to export's checked source map, reran45064exit0 to newv2output and confirmed all condition records unchanged. Counts97/99/97/98 reflect greedy feasible selection, not fixed-budget success or accuracy. All handles terminal; no GPU inference or held-out data tuning.

Uniform/oracle diagnostics: implemented common-placement U/O anchor audit79362exit0 (76/85selected of111requested). References are used only in explicit O condition. Extended dev evaluator with optional baseline report and macro matched/requested metric; tests44678passed23, evaluation22564exit0. Independent80record audit and Uniform/source-index audit passed. Plot57893exit0 visually inspected; fixed plotting code's prior hardcoded97/111caption to derive actual coverage. Results are exploratory with unequal realized counts, not a final fixed-generated-budget comparison. No user response to optional protocol question; requestedK+explicit failure remains a working diagnostic assumption, no held-out freeze claimed. No GPU inference.

Extended generation export to explicit U/O/V variants after reading current clean worktree, AGENTS.md, reports and live Slurm state. Agent implementation preserves event-centered frames versus playback time, checks report-specific source hashes and constraints, and never exports reference description text (O uses reference timing only). Tests64011 passed15; real CPU preparation38298 completed355 new rows and eight P0/P1 dry runs. U has one zero-placement video, retained in the ten-video coverage sidecar rather than silently removed from requested coverage. Independent field/frame/context checks recorded in placed-baseline-visual-input-check.json. No user labels, GPU runs, synthetic predictions or protocol-final claims. Used shell/Python, container runtime, pytest and Git; no new user contribution in this step. Both process handles terminal.

Inspected Proposal1.5 GF/OP definition and current selector code. Confirmed hard gap feasibility does not implement the rendered-AD overlap term. Added the actual rendered-audio evaluation path rather than labeling word-rate estimates as measured overlap. Agent used Python/shell, ffmpeg inside the existing CPU container, pytest and Git. Test24655 passed11; synthetic tone exists only as a unit-test fixture, not an experiment artifact or generated description. Checks cover detector union double-counting, overlapping narrations, boundary contacts, video overruns, missing evaluation, malformed intervals/IDs, real1.2x sample measurement and invalid TTS binding. CLI import/help passed53954. No user annotations, model inference or final metric claims; selector GF/OP and full TTS generation still pending. All handles terminal.

Implemented explicit opt-in GF score after checking current proposal and clean worktree/live allocation. Agent choice: bounded C/(C+T) normalization, transparently documented as exploratory; proposal gives relative gap/word capacity but no exact normalization. Kept required-word feasibility, generation caps, nearest-gap choice and requestedK accounting separate. Tests6554 passed43. CPU audit40510 completed392 selections; independent arithmetic check passed. Zero-weight audit97440 reproduced all391 earlier selections and prior condition fields exactly. No reference labels used to choose weights, no claim of tuned gain or measured rendered OP. Used shell/Python/container pytest/Git; all processes terminal and user GPU hold unchanged.

Long-video preprocessing: verified current clean worktree and liveCPU40036, read frozen subset/download audit and existing media pipeline. Implemented label-blind preprocessor checking exact frozen IDs, successful audio-gated download audit and raw MP4 SHA before derived writes. Started sequential CPU process71704 with per-video report checkpoints and standard2fps/768/16k settings. Observed808 and802 complete with10031frames; process remains live for other10. Added independent all-JPEG/full-PCM verifier and a truncated-JPEG regression test. Tests85352 passed7 (earlier43163 passed6). No answers/questions, semantic inspection, GPU inference or final-completion claim. Remaining preparation and verifier must be monitored in subsequent work; do not duplicate active job.

Long-video job71704 re-polled live; seven videos completed at latest log check, remaining active. Separate ffprobe timing90548exit0 checked all12 streams: all audio/video starts0, maximum duration discrepancy~0.081542s; evidence video-mme-stream-timing.json. This supports existing clip-relative timestamps without offset correction. While preprocessing ran, inspected installed faster-whisper language API and added explicit optional language policy, per-VAD-window detection metadata and English-only-weight guards. Defaults preserve existing en schema/settings. Tests56049exit0 passed12 plus4subtests. No multilingual model inference or QA text inspection; pinned multilingual weights/compatibility and semantic-cue language policy remain outstanding.

Monitored existing long-video preprocess71704 through completion, without duplicate submission; all12 passed. Launched independent verifier96153 only after complete report and terminal success. Verifier exited0, decoded all55549JPEGs and read444395013 PCM samples, rechecked raw/source/derived hashes and exact timestamp grid/coverage. Used existing CPU allocation, ffmpeg/PIL/wave/Python and shell process polling. No new downloads, questions/answer labels, semantic inspection, model predictions or synthetic experimental results. Long-form media gate now complete; semantic features/ASR/QA work still pending. All relevant process handles terminal, no GPU jobs submitted.

Started long-VAD CPU14618 on12 independently verified media caches with unchanged VAD defaults; confirmed first604 completion and live process (~886MB RSS at check). Consulted official Systran model card through web tool and HF model metadata API, pinned multilingual-small snapshot536b0662742c02347bc0e980a01041f333bce120 in separate config. Download42077exit0; offline real-dev check57147exit0 with CPUint8, explicit is_multilingual guard, auto language detection and existing VAD slice. Ten actual words stayed in gate, metadata en~0.9944. Recorded all model-file hashes. This checks mechanics on English dev speech, not non-English accuracy; no held-out text or model result inspected. Existing English caches untouched. Download/check handles terminal;14618stilllive. Used CPUcontainer, HFdownload, soundfile/faster-whisper, shell/Git; no GPU/API inference.

Re-polled VAD14618 and inspected completed feature files while it remained live. Added independent structural/source/gap arithmetic verifier; six tests70961passed. Checked actual first six completed video feature records37146exit0 against WAV durations and reconstructed gap complements; allpassed. No threshold changes, no test questions/answers or semantic judgments. Full12subset verification awaits original14618completion; current partial records are not reported as full success. Shell/Python/container pytest/Git used; no new model inference or GPU work in this step.

Monitored existing VAD14618 to terminal0, all12computed/no failures. Full independent audit7172 exited0: all hashes/protocol/durations/speech sums/gap complements passed. Started one long ASR pilot604,35246, only after its VAD had completed; rest of VAD worked on distinct videos. Confirmed active fourthread CPU work (~1.4GBRSS). Added speech-cache verifier that reports counts/language metadata only, checks actual PCM/VAD sample coverage and word gates, and rejects stale protocol/source hashes. Tests53535passed13+4subtests. Pilot remains live and cannot yet be called complete; no scale-up or QA tuning. All media/VAD handles terminal, only ASR35246active. Shell/Python/container/tests/Git; no GPU inference or human-label substitutes.

Long ASR pilot35246 completed5515words/105windows/0rejected; audit59487 passed source/protocol/VAD sample/word gates. Language estimates en102/ko2/pt1 recorded as model outputs only, not human language truth. Started full ASR94991 with unchanged pinned settings, reusing604; remains active. Separately discovered missing frozen EgoSchema pilot config/frames, froze10IDs using public500membership keys and Random0 (never label values or question text), explicitly excluded from490heldout IDs. Freeze tests57913passed4. Media58575exit0 then independent fullJPEG/source/silent-stream verifier37856exit0:10videos/3600frames, allsilent. No QA answer leakage or GPU/model QA predictions; only frame caches. All EgoSchema processes terminal; only long ASR94991 remains live. Source-only code/config/docs committed; large raw/derived artifacts excluded.

Implemented label-free Uniform EgoSchema pilot QA inputs and generic frozen-VLM QA runner. Read proposal held-constant4frames/2s and existing rate5/min; all180s pilot clips therefore15moments/60images. Questions loaded for frozen pilot only; labels never opened by generation builder/runner, nested label fields rejected. Strict standalone-letter parser does not search reasoning for favorable answers. Tests11858passed12 incl mocked resume and no-load dryrun; real10question dryrun40172passed without weights/predictions. Independent150anchor/600frame mapping and split audit passed. No QA accuracy claimed. Long ASR94991 polled live, no duplication. Used shell/Python/container pytest/Git, no GPU or paid API calls. Remaining GPU memory feasibility, V QA inputs, scoring and longQA still explicit.

Added QA evaluator with fixed expected denominator and per-ID failure coverage, separate conditional accuracy and safe retry accounting. Explicitly rejects held-out/pilot mixing, duplicate success, mixed selectors/run hashes, inconsistent parsed answer index and malformed JSON. Labels accessed only by evaluator after nonempty prediction validation, never generation. Fixed invalid-answer categorization to distinguish runner invalid_answer errors from runtime failures; added regression. Tests51222passed23 (earlier63055passed20), including no-label-read/no-score on empty prediction files. All generated predictions/scores in testing are synthetic ephemeral fixtures, not experiment artifacts. Confirmed ASR94991/96112still actively uses4CPUthreads, no new per-video output yet; left original process running. Shell/Python/pytest/Git; no GPU or paidAPI use.

Prepared EgoSchema V features on CPU alongside longASR, using existing pinned OpenCLIP/PySceneDetect defaults and bounded batch8/threads4. Process83383exit0, independent54574audit passed10/10 including embedding hashes/norms/reconstructed novelty/local-memory invariants. Extended label-free QA builder for V_local/V_memory; runner preserves condition/cache provenance and shortfalls. Tests80565passed23. Real20question dryruns88925exit0; paired check confirmed questions/options/budgets/windows/durations/audio policy identical to U and selected times match audited V sources, no predictions written. No labels or QA performance used for selection/tuning. Only ASR94991 remains active; all EgoSchema feature/preparation handles terminal. Shell/Python/container/Git; no GPU or paid API inference.

Started long visual feature preparation72713 on all12 frozen verified videos with unchanged pinned settings and bounded CPUbatch8/4threads; confirmed live process and acceptable memory alongside original ASR94991. Added frozen-long-subset option to existing visual verifier and recorded exact audit command; help/import76046passed, actual full-cache audit deferred until completion. Re-read proposal human gate:about10movie-audio clips ofabout10min and60blinded peer comparisons. Requested existing human artifact locations asynchronously; did not synthesize labels or infer an answer from silence. Both heavy processes remain active; no GPU jobs or QA-label inspection.

Re-polled ASR94991 and visual72713 live; observed616 ASR completion, validated firsttwo via30098exit0 (10465words638windows0rejections). Inspected Slurm sourcejob40036 endtime/partition/resources. To preserve unfinished CPU work across walltime without dummykeepalive or canceling userallocation, prepared dependency-gated CPU continuation with frozen source worktree, per-video reuse and bounded3h/4thread/8GB resources. Syntax checked bash-n. Exact submission and jobID to be recorded after commit; no GPU authorization inferred. Human artifact question remains unanswered.

Committed continuation9e20099, created detached worktree at exact SHA, submitted CPU-only job40363 with afterany40036. Slurm verification confirms PENDING Dependency,4CPU8GB3h,noGPU,correct frozen WorkDir/Command/log. Checked active hostname/cgroup b-10-63/job40036 to confirm compute-node execution. Current userallocation not canceled or extended; no dummykeepalive. OriginalASR94991 andvisual72713 remain active; visual604 cache nowcomplete. Future40363 must still be monitored, validated and reported honestly. User human-data response remains pending; no approval inferred from silence and no GPU hold overridden.

Code inspection exposed a concrete interruption edge case: read_rows skipped a partial last line but subsequent append could concatenate the next prediction onto it, losing a valid new record to parsing. Added readonly strict resume inspection and lossless backed-up atomic tail repair to AD/QA runners; complete-line corruption now fails instead of being hidden on resume. Preserved old general read_rows helper for existing readers, and kept dry-run nonmutating. Regression tests60437passed48, including real byte-level partial-UTF8 repair and mocked two-question resume retaining both IDs. Only synthetic test files repaired; no actual predictions exist. Latest scheduler confirms40036running/40363dependency pending, activeASR/visual continue on compute node. No GPU hold change, no human labels fabricated.

Started bounded long event pilot21486 on604 with unchanged dev-tested AST/CLAP settings, after checking availableRAM~19GB and current CPU workload. It remains active; inspected CLAP unused text-branch warnings rather than assuming all warnings benign (no missing/audio unexpected weights found). Extended audit with explicit frozen-subset/pilot coverage and protocol duration/memory checks; event tests9041passed3 and actual ten-dev-cache reaudit88093passed10/10. Verified newly finished ASR710via28950,6099words329windows0rejections. No long event completion, detector accuracy, human labels or QA results claimed. Both originalASR94991/visual72713 remain live; queuedCPU40363 unaffected and excludes event work. Shell/Python/container/Git; no GPU or API calls.


## Cue2Narrate access re-audit — 2026-11-23

Read-only GitHub API checks found main at
`08c89ea63281b819a0fd73c060d8c954f9e696ef`, containing only `.gitignore`,
`LICENSE`, and `README.md`; the releases list is empty. No runnable implementation,
checkpoints, or paired-window annotations are supplied in that public snapshot.
Evidence: `outputs/verification/cue2narrate-access-audit.json`.
Repository: https://github.com/multimodal-ai-lab/Cue2Narrate

The paper's release plan distinguishes restricted LongLSMDC data from intended
public code/checkpoints: https://arxiv.org/html/2609.01725v1 . The official LSMDC
page routes access through MPII-MD registration and credentialed downloads:
https://sites.google.com/site/describingmovies/download . Its linked form at
https://moviedescreg.mpi-inf.mpg.de/register.php is reachable and requests an email
address. No form was submitted or message sent. Standard LSMDC access alone does
not establish availability of LongLSMDC paired windows or Cue2Narrate weights.
Direct reproduction remains pending those artifacts; the baseline is retained
in scope, and published scores are not substituted for our own evaluation.


Long visual progress audit69272 finished with exit1, explicitly incomplete:
6/12 frozen IDs verified (604,616,710,729,753,763), 26482 frames total.
The other six reported no_visual_features, consistent with the live writer;
no completed cache failed numerical/provenance checks. Evidence:
outputs/verification/video-mme-visual-progress-check.json. Audit checks current
source hashes, 512-dimensional normalized embeddings, reconstructed novelty,
shot bounds and local/memory ranking budget/separation. It is not model accuracy.
Original visual process102860, ASR96112 and event pilot106092 remain live;
no duplicate jobs were started. CPU40036 ends04:22:47UTC;40363 still pending its
afterany dependency. Full final visual audit remains required after completion.


Human event-detector scoring implementation added: src/evaluation/event_detection.py
and scripts/evaluate_event_detector.py. Requires explicit human-origin annotations,
annotator attribution, reviewed classes/intervals, unknown-region exclusions,
source WAV hashes, development protocol IDs and model/window/threshold settings.
Reports per-class/per-video/pooled window-presence P/R/F1, excludes incomplete
review coverage, and checks current source/array/protocol identity and time grid.
Exact boundary contact is not event overlap; undefined ratios are null. This is
not onset accuracy, authentication of human identity, or the completed ~10x10min
human gate. No real labels or experiment scores created. Proposed convention and
input contract appended to docs/AUDIO_STACK.md; protocol/source selection remains
to be frozen before human scoring. Tests18138exit0:19passed, including boundary,
partial/unknown coverage, invalid data, empty input, and changed-WAV rejection.
Initial tests52591exit0:15passed before adding the full CLI provenance regression.
Synthetic labels stay inside temporary tests. GPU hold and human-data question
remain pending; this preparation does not remove those gates.


Added per-batch CPU event recovery in src/utils/event_checkpoints.py and integrated
it into cache_event_features.py for subsequent runs. Each source/settings-digest
namespace stores atomic float32 probability/raw-embedding batches with hashes,
indices and shape checks. Invalid/incomplete batches recompute; final caches and
numerical recipe stay unchanged. Progress logs now report completed windows and
reused batches. Live pilot106092 predates edits and remains untouched (no duplicate
writer or restart). Tests5975exit0:28passed across checkpoint, event feature and
human scoring suites; simulated interrupted inference reuses only completed
batches and reconstructs bit-identical cue arrays. No real interrupted model run
yet. Agent mistake: initial test command76673 referenced nonexistent
 tests/test_event_cache_audit.py, exit4/no tests; corrected to actual test files.
Visual extraction reached8/12 (through808); ASR and event pilot remain active.

Registered isolated real development event-checkpoint validation at /scratch/yy5919/pyro-kd-derived/checkpoint-validation-e233432, clip CxWby8RuUwU6 (73.2s). Source manifest copied unchanged with absolute references to existing raw/frame/audio data; only new event/checkpoint outputs written to isolated root. Source e233432, pinned AST/CLAP, batch4/CPU4. No duplicate writer to production caches. Plan: complete actual extraction, preserve final outputs, replay checkpoints and compare every array; human detector scores and GPU inference are not part of this check.


Real CPU event-checkpoint validation COMPLETE on isolated development clip
CxWby8RuUwU6, source e233432: initial extraction36888exit0 produced74windows and
19batch checkpoints. Final metadata/NPZ were preserved as .first files, then
replay94626exit0 rebuilt final caches reusing19/19batches. Comparison42573exit0
confirmed all7arrays bit-identical and both NPZ SHA256 equal
 e43bf89fab55d1b6fdc6e3a1fe5b9a4a468e642814425015474e073ca2bed69c.
Independent numerical/source verifier91709exit0 passed1/1selected development
clip, explicitly not10/10coverage. Evidence outputs/verification/
event-checkpoint-real-{first,replay}.log, event-checkpoint-real-comparison.json,
and event-checkpoint-real-audit.json. Actual artifacts at
/scratch/yy5919/pyro-kd-derived/checkpoint-validation-e233432/CxWby8RuUwU6.
This exercises real model outputs and missing-final-file recovery, not an OS-kill
experiment or detector accuracy. Production caches/raw data were untouched.
Long ASR96112, visual102860 and event pilot106092 continue; visual9/12logged.


Long-video QA question membership frozen in configs/video_mme_long_questions.json:
36 unique questions, exactly3 for each of12 frozen videos. New script
scripts/freeze_video_mme_questions.py reads only video_id/videoID/duration/domain/
question_id columns after checking the original annotation-file SHA256. It
rejects mismatched video metadata, missing video coverage, duplicate IDs and any
extra content columns. No question text, options or answers inspected. This
freezes the evaluation denominator, not selection/inference settings. Tests52280
exit0:6passed; actual freeze32693exit0; independent membership/source-link checks
passed in outputs/verification/video-mme-question-membership-check.json. Agent
mistake: first attempted the data venv interpreter on the host (exit127); it
requires the container's /ext3 interpreter, and the corrected in_env invocation
succeeded. Visual extraction now10/12logged (through864); longASR/eventpilot live.


Long-video QA evaluator added in scripts/evaluate_video_mme_qa.py. Uses frozen
36-question membership/source hashes; refuses empty predictions before reading
labels, reads only selected answer rows after prediction provenance checks, and
reports per-video/domain plus overall fixed-denominator accuracy. Missing,
invalid and truncated answers count incorrect. Tests76164exit0:19passed across
new and shared QA scoring. No actual answer labels loaded or QA scores created.

Long event pilot21486exit0:604 completed1832windows. Independent audit15996exit0
verified exact pinned models, WAV/source identity, dimensions, normalization,
RMS and reconstructed cues; explicit1/12coverage, not detector accuracy. Prepared
hpc/long_events_resume.sbatch (bash syntax and model pins checked): CPU4/12GB/6h,
revalidates604pilot before extraction, resumes batch checkpoints, audits all12,
per-attempt logs/status and nonzero failures. Plan full event extraction now
insideCPU40036 plus immutable continuation afterany40036; no GPU use. All ordinary
CPU work stays on the compute allocation; no duplicate active cache writers.


Full long event extraction started handle27774 insideCPU40036 from source
65258e9458db0b8a94c468eb682f5eba984b213f, explicit12frozen IDs, batch4/CPU4,
using verified604pilot and per-batch checkpoints. Log
outputs/verification/video-mme-events-full.log. CPU-only continuation40408
SUBMITTED and Slurm-verified PENDING Dependency afterany:40036(unfulfilled),
4CPUs/12GB/6h, noGPU, immutable detached worktree
/scratch/yy5919/pyro-long-events-resume-65258e9 at aboveSHA. VerifiedWorkDir/Command
point there. Submit command from frozenworktree: sbatch --parsable
--dependency=afterany:40036 --output=/home/yy5919/PYRO-KD/logs/pyro-long-events-%j.out
hpc/long_events_resume.sbatch /home/yy5919/PYRO-KD/outputs/verification.
Logs: /home/yy5919/PYRO-KD/logs/pyro-long-events-40408.out; reports:
outputs/verification/long-events-resume-40408-attempt-0/. Pilot is revalidated
before scaling in the continuation, and final12video audit required. Queue
submission is not completion; continue monitoring. Separate pending40363 resumes
ASR/visual and does not write event caches. No duplicate active event writer.
Latest originaljobs: visual11/12logged through891; ASR729computed7361words,
619VADwindows,0rejected (independent729audit stillpending). No GPU inference.


Fourth long speech cache729 independently verified7537exit0:7361words across
619VADwindows,38575990speech samples,0rejected words. Evidence
outputs/verification/video-mme-asr-729-check.json; feature SHA256
3d419e61d58be3dfa03f144ac7fe4fdd851324145d42ea9e852f8349d3617904.
Detected language-window codes are unvalidated model estimates, not ground truth.
Verified ASR coverage now4/12. Livevisual72713/process102860 polled directly and
still active on final898;11complete visual metadata files, no restart. Full event
job27774 has begun616 with batch checkpoints;604 remains sole completed long event
cache. CPU continuation40363 and40408 still PENDING Dependency afterany40036.
Full visual audit remains next after72713terminalsuccess, not yet complete.


Long visual extraction72713exit0:12computed,0reused,0failed. Full independent
audit85259exit0 COMPLETE12/12:55549normalized512-D embeddings and6850shot boundaries,
current source/feature hashes, reconstructed novelty and local/memory ranking
budget/separation invariants passed. Evidence outputs/verification/
video-mme-visual-check.json and .log. No semantic/QA accuracy claim. Current status
header refreshed to distinguish completed preparation from pending experiments.
Long-QA resource inventory using unchanged dev5moments/min and4frames/moment
requires612–1012images/question; evidence video-mme-qa-resource-inventory.json.
This is not a frozen long budget or measured GPU limit. Determine a feasible
held-constant budget using development GPU smoke before opening held-out QA
content; do not silently truncate frames or change K across selectors.


Long QA actual-file integration check68288exit0:3tests passed under the data
venv using temporary synthetic Parquet/JSONL only (tests/test_video_mme_qa_cli.py).
The real CLI filters out an invalid answer outside frozen membership, preserves
missing-question denominators and nonzero incomplete exit status, records source
hashes, and rejects mismatched prediction provenance before attempting answer-file
access. No real benchmark answer values were opened; no experiment QA scores
were created. This resolves the previously untested Parquet filtering/CLI path.

Fifth long ASR cache753 audited45639exit0:6674words/150VADwindows,
32247997speech samples,0rejected words. Evidence
outputs/verification/video-mme-asr-753-check.json; feature hash
9fd67265d0680997dff21ee6da4af81bcdc0bb24a8a6160fdc5ec8c98186b7e9.
Per-window languages are model estimates and not semantic validation.
Long speech/event extraction continues; visual preparation remains verified12/12.


QA paired-analysis bridge implemented in scripts/export_qa_paired_metrics.py.
Exports all frozen per-question outcomes including missing/errors as0, with
video_id as cluster_id. Rejects differing question membership/source hashes,
video mappings, answer labels and inconsistent outcome/summary metrics. Adjacent
provenance records source report and export hashes. Existing paired cluster
bootstrap/Wilcoxon and predeclared-family Holm pipeline can consume it; equal-video
estimand explicit. Tests21299exit0:14passed across new exporter and existing paired
analysis. Fixtures only, no real QA statistical findings created. Documented
--unit cluster requirement, especially3questions/video for Video-MME. Current
CPU40036 elapsed5h05m of6h; continuations40363/40408 remain dependency-pending.
Long event616 progressed700/2646windows, longASR remains live, no duplicate jobs.


Actual CPU QA processor preflight84874exit0 on first EgoSchema U pilot question:
60images encoded into14075input tokens;16new-token allowance fits the pinned
Qwen2.5VL7B config's128000context. New scripts/check_qa_processor.py records source,
config and frame hashes plus tensor dimensions/bytes and explicit coverage;
local processor only, no model weights/GPU/QA predictions. Coverage1/10pilot,
not GPU memory feasibility or long-input capacity. Evidence
outputs/verification/egoschema-qa-processor-pilot.json/.log; reproduction command
in docs/REAL_DATA_SMOKE.md. Shared-template Transformers kwargs-placement warning
recorded; actual encoding passed. Initial read tried nonexistent check_vlm_inputs.py;
corrected discovery found debug_processor.py and the actual shared generation path.
Long ASR96112 and event115097 remain active; no duplicate writers or GPU request.


Processor warning audit85969exit0 inspected installed Transformers
ProcessorMixin.apply_chat_template source (outputs/verification/
processor-template-source.txt). The compatibility branch collects processing
kwargs from **kwargs, warns, then merges them into processor_kwargs before calling
self(..., **processor_kwargs). Thus padding=True is forwarded, not ignored;
the successful QA encoding is not evidence of a padding failure. No generation
code change was made. This resolves the warning concern without altering the
shared AD/QA recipe. Actual GPU feasibility remains untested. Live process check:
ASR96112 and events115097 active at~4CPUthreads each; event616 reached1100/2646
windows with batch checkpoints. CPU40036 has~51minutes remaining; dependent
CPU40363/40408 still queued. No restart, duplicate submission or GPU allocation.


Prepared real long visual hierarchy mechanics audit. VisualCueSelector gains
optional segment IDs/quotas for local and memory selection; defaults preserve
flat behavior. Memory and2s separation remain global across segments. New
scripts/audit_long_visual_hierarchy.py uses all12 verified visual caches, existing
flat audit and dev20–60s/quiet-floor1 settings; boundaries are visual shots only,
density is local visual candidate-score mass/second. Existing5/min budget is
mechanics-only, not final QA capacity. No question/answer access or heldout tuning.
Tests48463exit0:80passed/49subtests, including one-segment equivalence, cross-boundary
spacing and memory not reset; prior87238passed79before added memory regression.
Register next CPU4-or-less audit on40036 with output
outputs/verification/video-mme-visual-hierarchy.json; no feature cache writes.


Real long visual hierarchy mechanics3720exit0 COMPLETE12/12. Source24850af
(full SHA recorded in outputs/verification/video-mme-visual-hierarchy-source.txt).
Using dev20–60s bounds/floor1 and visual-shot boundaries yields1144segments;
local and global-memory hierarchy each select2314/2314requested moments with
0shortfall. Independent arithmetic audit verified contiguous full-duration
segments, <=60s spans, quota sums/floors, unique timestamp membership counts,
per-segment quota bounds and global2s spacing. Evidence
outputs/verification/video-mme-visual-hierarchy.json/.log and
video-mme-visual-hierarchy-check.json. This validates long-video mechanics only;
no QA content/labels inspected, no accuracy/quality gain claimed, and5/min remains
an exploratory resource-heavy budget rather than final GPU-validated protocol.
A/AV hierarchy still requires completed audio features and multilingual policy.


Sixth long ASR cache763 independently verified25909exit0:3959words/423VADwindows,
20860789speech samples,0rejected words. Evidence
outputs/verification/video-mme-asr-763-check.json; feature SHA256
 a22390e534a0325c0ec10bb5f83e43739082334571356cbc4e1bc78af5919c66.
Verified speech coverage6/12. Language estimates remain unvalidated.

Live jobs96112/115097 confirmed at~4CPUthreads each. Event616 has395committed
batch metadata records covering1580windows at this check (not a completed-video
audit). PendingCPU40363/40408 were revalidated through scontrol: both depend on
40036, with correct WorkDir/Command and clean immutable source checkouts at
9e20099a7dcae76a457da2222b18e64e8be15fe4 and
65258e9458db0b8a94c468eb682f5eba984b213f respectively. No duplicate submissions.
CPU40036 had~46minutes remaining; walltime is not extended by activity. Queued
work still needs monitoring through terminal status and final coverage checks.


Audio-native candidate preparation: official Qwen2.5Omni7B release API pinned
 ae9e1690543ffd5c0221dc27f79834d0294cba00. Processor-only download81765exit0 verified
13files/16175017bytes;22.36GBweights and speaker data not downloaded. Actual real
REFRAMED development audio+4image encoding69176exit0 produced2430tokens with both
modalities present; no model weights loaded. New config and check script plus
hash evidence retained. Runtime audit36991exit0/source-file inspection resolved
API difference: generation_mode='text', legacy return_audio via deprecated alias;
full wrapper still needs speaker data. Config warnings remain an actual-load
check, not silently declared safe. Agent metadata mistake: explicit signature
inspection initially named 'supports_return_audio'; corrected to 'declares' and
preserved original report, because **kwargs/decorators support the legacy option.
Audio-native ablation remains incomplete and unfrozen; GPU hold unchanged.


Live event checkpoint audit97214exit0 passed a snapshot of537committed batches /
2148contiguous prefix windows for longvideo616. Recomputed settings+currentWAV
configuration hash6d7a0fcb66d9694e2d1891d4424e34ec1bff6433b8f1341420abc1bb40f8e78d,
then verified every committed payload hash, index/count, float32 dimensions,
finite probabilities and nonzero embeddings through EventCheckpoints.load.
Evidence outputs/verification/video-mme-events-live-checkpoint-audit.json;
reproducer check_live_event_checkpoints.py beside it. Live writer was not changed
or restarted; later batches are explicitly outside this snapshot. This is not
complete616coverage or detector accuracy. Both longASR96112 and event115097 remain
live; CPU40036 had~41minutes left at the initial check, continuation jobs unchanged.


Prepared development-only per-class event/AD association analysis in
scripts/analyze_dev_event_alignment.py and src/evaluation/event_alignment.py.
Uses frozen10REFRAMEDvalidation clips, mergedUS AD starts (.5s merge), existing
AST probability>=.5/positive rise>=.1 and +/-2s AD proximity on the1sgrid.
Reports2x2counts, conditional proximity rates and phi per video/class, with
undefined correlations and support explicit; excludes predefined speech classes.
No test data, p-values, detector-accuracy interpretation or automatic weight/sign
tuning. Tests12801exit0:6passed including known+1/-1/undefined correlations and
inclusive boundaries. Next actual CPU diagnostic uses existing cached events and
output outputs/verification/reframed-dev-event-alignment.json. Initial exploratory
file reads guessed two nonexistent fused-audit names; corrected via rg discovery
to audit_fused_selectors.py before implementing the missing association path.


Development event/AD association34065exit0 COMPLETE10clips/1336windows/517included
classes, source5d0833b. Only11classes have detected onset windows at the existing
probability.5/rise.1 thresholds;10of those occur in just1clip. There are51class-window
detections total. Music appears in all10clips (31windows), equal-video mean phi
0.017814021769080805; this is descriptive, not a significance or causal result.
Sparse support does not justify fitting unconstrained per-class signs from these
clips. No thresholds/weights changed and human detector validation is still needed.
Evidence outputs/verification/reframed-dev-event-alignment.json/.log and independent
count/phi arithmetic audit reframed-dev-event-alignment-check.json. Preserves all
classes and undefined correlations, avoiding selective reporting of favorable
classes. Actual long event616 now2400/2646windows; longASR continues.


Second long event video616 completed2646windows; independent verifier34564exit0
passed pinned model/sourceWAV/array hashes, exact time grid, finite527-class
probabilities, normalized512-D embeddings, RMS and reconstructed cues. Evidence
outputs/verification/video-mme-events-616-check.json (1selected of12, explicit
partialcoverage). Combined independently verified event coverage now604+616:
4478windows,2/12videos. Live handle27774/process115097 moved to710 (1982windows);
no restart or duplicate writer. ProcessorRSS dropped after completing616, and
saved batch checkpoints are retained. Original ASR96112 remains active.
Actual full event extraction is still incomplete; final12video audit required.


Registered long-video spectral-flux onset preparation onCPU40036 using existing
scripts/cache_onset_features.py and unchanged SETTINGS(version1,16kmono,hop160,
FFT512,64mels,centerFalse,1smax bins,per-video-max normalization). Input is the
verified12video WAV cache; no event/ASR dependency and distinct onset files avoid
writer overlap. Command in_env ro 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1 /scratch/yy5919/pyro-audio-env/bin/python -u
scripts/cache_onset_features.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme'.
Log outputs/verification/video-mme-onsets.log. No parameter tuning or label access.
Independent provenance/grid/normalization audit follows completion.


Long spectral-flux onset preparation76263exit0 COMPLETE12/12. Independent audit
66784exit0 verified27780one-second windows across all12: current source/WAV/config
hashes, exact grid, finite nonnegative values and per-video-max normalization.
Full raw-WAV recomputation for604 matched stored strength/normalized arrays;
other11received provenance/numerical checks without redundant reinference.
Evidence outputs/verification/video-mme-onsets-check.json and video-mme-onsets.log;
new reusable scripts/verify_onset_cache.py supports explicit recompute IDs.
Existing onset tests11596exit0:3passed. Same development parameters, no threshold
or normalization tuning, no QA labels inspected. This is spectral-flux consistency,
not human onset accuracy. Learned event/ASR stillactive; event710 now200/1982windows.

Registered full pinned Qwen2.5Omni7B asset download onCPU40036 after successful actual multimodal processor check and runtime API review. Same ae9e1690543ffd5c0221dc27f79834d0294cba00 revision; includes five model shards and speaker data required by the installed wrapper. Reuses processor snapshot. No GPU job, paid API or model instantiation. Config configs/qwen25_omni_model.json; log outputs/verification/qwen25-omni-model-download.log and report qwen25-omni-model-download.json. Intended to make later GPU ablation smoke offline; config warnings and real GPU feasibility remain unresolved.


Full Qwen2.5Omni7B download3293exit0 and independent file/index verifier63745exit0
COMPLETE:19assets/22379295753bytes,5safetensors shards/2448tensor names, pinned
 ae9e1690543ffd5c0221dc27f79834d0294cba00. Includes required speaker file. Every
large file matched upstream LFS SHA256 and plain files matched pinned Git blob
SHA1; all files also have recorded SHA256. Safetensors header keys exactly match
index assignments. Evidence outputs/verification/qwen25-omni-model-check.json/.log,
model-download.json/.log and upstream-file-identities.json. New reusable verifier
scripts/verify_pinned_model_files.py streams8MBchunks without instantiating a model.
Full assets now staged; earlier processor-only status is superseded. GPU loading,
configuration-warning validation and audio-native inference remain untested.

Seventh longASR802 verified32201exit0:7964words/711VADwindows,
36587194speech samples,0rejected. Feature SHA256
 e4e22eccefd41daef9dd51e58da7811dd9771154d248c640afd28d4896d89747;
evidence outputs/verification/video-mme-asr-802-check.json. Language codes remain
model estimates. Long speech/event extraction continues onCPU40036; no GPU request.


Native waveform development-input adapter validated: scripts/attach_native_audio.py
and src/generation/native_audio.py preserve existing selections and attach mono16k
waveform spans centered on frame_center_time +/-15s. Full WAV and decoded slice
SHA256, sample boundaries, declared seconds/rate and finite samples are checked.
Actual U snapped development export:76moments/9videos, all original fields including
frames and word budgets unchanged. Output /scratch/yy5919/pyro-kd-derived/moments/reframed_U_snapped_placed_event_dev_exploratory_native.jsonl.
Evidence outputs/verification/reframed-native-input-check.json records input/output,
source and split hashes; focused tests5passed. First independent audit incorrectly
assumed a strict480000sample ceiling: outward floor/ceil rounding adds one sample
for fractional centers. Corrected audit checks exact floor/ceil and <2sample rounding
tolerance; no data rewritten. No model inference or native runner integration yet.
Verified running ASR96112/event115097 on40036; continuation40363/40408 pending
Dependency. GPU hold remains. Full proposal experiments are incomplete.


Added explicit --backend omni to scripts/run_ad_vlm.py. It requires batch-size1,
no quantization and no reasoning-strength; native context requires this backend.
Existing frame/text modes remain available on the same Omni backend for later
controlled comparisons. Native rows validate full WAV/slice hashes before resume
and again before generation, reject transcript/timeline mixtures and mismatched
visual centers. Output records retain exact native sample metadata; provenance
includes backend and full input JSON. Shared error/truncation/recovery paths remain.
src/generation/omni.py uses local pinned assets, disables talker after model load,
requests generation_mode=text with thinker-specific decoding arguments, and
rejects unexpected audio-output tuples. GPU loading/inference remains untested.
Tests39460exit0:41passed (backend mocked-model contract, native spans, generation
truncation/provenance). Tests75806exit0:16passed+11subtests for recovery/audio
context/prior. Actual native CLI dry-run passed, no prediction file/model load.
Actual shared native processor check29796exit0 encoded the first real frozen dev
U moment:2390tokens, image+audio tensors present. Scope1/76moments only; noGPU
memory or quality claim. Evidence outputs/verification/reframed-native-ad-processor-check.json/.log
and native-ad-dry-run.log. Existing talker config token warnings persist and must
be evaluated in the later real GPU smoke. No GPU submission or paid API usage.


Long event710 independently verified85596exit0:1982windows, current source/model/
configuration/array checks passed. Evidence video-mme-events-710-check.json/.log,
explicit1/12auditcoverage; cumulative604+616+710verified6460windows. Live extraction
has advanced to729. ASR remains7/12verified. Scheduler recheck:40036RUNNING,
ends04:22:47UTC;40363and40408PENDINGafterany40036. Both detached source worktrees
remain clean at9e20099a7dcae76a457da2222b18e64e8be15fe4 and65258e9458db0b8a94c468eb682f5eba984b213f.
No duplicate jobs submitted; completion and full audits still pending.
Native AD runner integration test2716exit0 passed with temporary synthetic
waveform/frame and mocked inference. It verifies waveform delivery, exact output
metadata, settled resume without loading and rejection of changed WAV even when
output is already settled. No actual AD prediction generated. Added exact later
same-Omni none/native GPU smoke commands to docs/REAL_DATA_SMOKE.md; unexecuted
under GPU hold. This does not resolve GPU memory, configuration warnings or quality.


Eighth long speech808 verified64715exit0:5697words/516VADwindows,25454386speech
samples,0rejected. Feature SHA c4b0ac94b781dd7f8851c68ba5323114b8c7e0f157b43e5e6ec5455b155befa0.
Evidence outputs/verification/video-mme-asr-808-check.json/.log. Detected language
codes remain model estimates. Full long speech/event processing continues.

Core-matrix readiness inspection exposed a missing REFRAMED generated-AD evaluator:
existing evaluate_ad.py only loaded MTG. Added explicit dataset adapter using US
cues merged at<=0.5s (same placement diagnostic policy), movie cluster IDs, frozen
membership filtering and explicit missing-selected-video accounting. CLI records
selection hash and reference policy. Mixed backend/audio/selection/fingerprint
values are rejected; absent audio mode is treated as legacy for mixture checks.
Synthetic fixture integration checks merged reference text, missing-video FN,
shared movie IDs, out-of-membership rejection and mixed audio rejection. Existing
MTG schema assertions updated for explicit reference and coverage fields; first
run failed one old exact-schema assertion, then corrected the expected schema.
Final focused tests37644:29passed+27subtests. No real AD outputs or metrics created;
semantic-model freezes, OpenCLIP, rendered overlap and calibrated judges remain.
Documentation: src/evaluation/README.md. No GPU/API action.


Added REFRAMED temporal paired export scripts/export_ad_paired_metrics.py. It
requires complete frozen selected-video coverage, selection hashes and generation
fingerprints; compares perclip reference hashes, movie IDs and temporal settings;
recomputes P/R/F1 from counts. Missing predictions remain0withreference, empty
reference/prediction retainsnull. Exports clip examples/movie clusters, explicit
errors/generatedcounts and source/output hash sidecar. Existing analyze_paired
--unit cluster estimates equal-movie mean within-movie clip metrics. No pooled-F1
or semantic-quality interpretation. evaluate_ad.py now hashes input bytes and
actual merged reference timestamp/text pairs. Final corrected tests66234exit0:38passed+27subtests,
including grouped clips, missing-predictionzero, mismatched references/membership/
protocol/counts and missing generation evidence. Fixtures only, no real scores.
Full matrix, tuned/equal-Kprotocol and semantic/quality/human evaluation remain.


Correction to paired-export validation history: first test57034 exited1 with
37passed/1failed. Exporter F1 reconstruction incorrectly omitted FN in its
denominator. The missing-prediction/cluster fixture detected it. Fixed formula
to2TP/(2TP+FP+FN); rerun66234exit0 passed38tests+27subtests. Agent prematurely
wrote a passing test entry in27cd883 before inspecting the completed result;
that claim is corrected above and this history preserves the mistake. No real
evaluation export or scientific result was produced with the faulty version.


Pre-walltime checkpoint2026-11-23T04:18UTC: ninth longASR857 independently
verified96370exit0 (631words/116VADwindows/2639997speech samples/0rejected).
Feature SHA8e743e1770da419ba2de7077a4401027cb7531f7cddce57efb49fc0739cf3454;
evidence video-mme-asr-857-check.json/.log. Language labels are estimates only.
Current event729 committed-prefix audit89349exit0 verified176batches/704windows
against current WAV SHA and independently audited710recipe, including payload
hashes, contiguous indices, shapes, finite values and valid probabilities.
Digestaf10f8c942e5f352e8d2846e71f2fab9a99294bfef9f644f2fe7bc7fe9ab732f.
Evidence video-mme-events-729-checkpoint-audit.json and reproducible audit.py.
This is a prefix snapshot, not completed729features; completed event coverage
remains3/12. ASR is atomic pervideo, so its in-progress video may need repeating.
No duplicate writer or job restart.40036ends04:22:47UTC regardless ofactivity;
40363and40408remain dependency-pending withclean immutable worktrees. They still
require observed completion/full audits. GPUhold remains; finalexperimentsnotdone.


Core matrix input inventory now explicitly enumerates all20U/V/A/AV/OxP0/P1xopen/API
cells while retaining8exploratory selector variants without choosing finalVmemory
or event signs. Script scripts/audit_core_matrix_inputs.py validates frozen dev
membership, unique moments, four existing frames/timestamps, positive word budgets
and within-video requestedK consistency; records input/split hashes and perclipcounts.
Actual run passed:U76moments/9videos,O85/10,Vlocal/Vmemory/Atrigger/Asuppress/
AVtrigger/AVsuppress97/10each. Current files are inventoried by hash; they need not
represent newer exploratory recipes elsewhere. This confirms unequal coverage and
no final equal-K protocol, including one absent Uclip. No generated output scored.
Evidence outputs/verification/core-matrix-development-inputs-v2.json. Earlier
inventory version hardcoded zero completed experiments; revised tool explicitly
does not inspect execution outputs, avoiding a future stale completion claim.
NativeOmni remains ablation, not API substitution. GPUhold/APIcap/humangates remain.


2026-11-23T13:38UTC resumed in user-provided GPU session40880 onb-19-15,
A100-SXM4-40GB,24h limit ending2026-11-24T13:26:29UTC. GPU initially0MiB/0%
withno compute processes. User condition (finish transfers then obtainGPU) is now
satisfied; proceeding with authorized small GPU smoke, not fullmatrix scaling.
OldCPU40036TIMEOUT as scheduled. Continuations40363COMPLETED0:0 in36m36s and
40408COMPLETED0:0 in3h37m32s. All step statuses0. Final speech12/12,visual12/12,
events12/12auditscomplete; reports underoutputs/verification/long-cpu-resume-40363-attempt-0
andlong-events-resume-40408-attempt-0. No detector/transcriptionaccuracy claim.
Registered first real Qwen2.5VL7B smoke:4MTGdevmoments,P0,noaudio context,batch1,
128newtokens,temp0,bf16,sdpa,pinnedcc594898137f460bfe9f0759e9844b3ce807cfb5.
Output outputs/real-data-smoke/qwen25-mtg-U-P0-none.jsonl and associated metadata;
logoutputs/real-data-smoke/qwen25-mtg-U-P0-none.log. Requires actual completion,
output/frame/wordbudget inspection and evaluation before scaling. No duplicate
GPU writer. Separate CPU user session40879 remains untouched.


First real GPU smoke51403exit0 complete on40880 using source3811351:4MTGmoments
fromonevideo. Load191.8s,generate8.0s,peakallocated16.99GB,0tokenlimitfailures.
Actual output audit found2/4word-budgetviolations (3>2 and6>5), so this did NOT
pass the smoke gate despite old runner n_ok4. Agent frame review also notes
unsupported left-hand/pressing detail in guitar caption; rawdecodesunchanged.
Reviewoutputs/real-data-smoke/qwen25-mtg-U-P0-none.review.json andframes.jpg.
Evaluation61469exit1 expectedpartial:2valid/2budgetfailures,0temporalpairs among
limited4inputslotsvs5references; no semanticmodels loaded/scored. Not a selector
performance claim. Added whitespace-wordbudget failure accounting to runner and
shared outputreader, including legacyunflaggedviolations, retainingfullrawdecode.
Clarifiedstrict wordlimit insharedP0/P1prompt; no post-hoc truncation. Tests8622
firstfailed mockedmodelGPUstatreset onnewGPUhost; fixedtestCUDAisolation. Final
73407exit0:61passed+38subtests. Registered repeat same4inputs innewoutput
qwen25-mtg-U-P0-none-budget-v2.jsonl, same model/settings except sharedpromptfix.
No fullmatrix scaling. Ten-videoREFRAMEDsmoke input also preparedfromearliest
existingpositivebudgetUrowper frozenclip, provenance reframed-ten-video-smoke-inputs.json;
notfinalselectorprotocol andnotyetinferred.


Second GPU smoke47071exit0 completed source8b54a64:4decoded/3withinbudget/1budget
failure,0token-limitfailures. Actualattemptmetadata load14.0s,generate6.0s,
peakGPU17.0GB. Remaining2wordcase output"boat on water." has3words;
newrunner correctly records error and preserves fullrawdecode. Other outputs:
"Person plays guitar with fingers on strings."(7/23),"man closes eyes mouth open hand on face."(8/11),
"man looks at camera."(4/5). Agent frame inspection supports broadactions but
is not humanqualityvalidation. No all-clear smokeclaim/gridexpansion. Nextneeded
is bounded shared word-budget repair with allattempts/cost preserved, then
actualten-video development evaluation including semantic andrenderedoverlap.


Registered optional bounded word-budget repair (--word-budget-retries1), shared
acrossP0/P1 andVLM/Omni backends. One extra same-model/frame-context call only for
nontruncatedoverbudgetoutput; bothprompts/rawdrafts/token counts/errors retained in
generation_attempts, total_new_tokens/generation_calls/repair_latency_s recorded.
Failure remainsfailure, no stringtruncation/no unboundedloop. Policy enters run
fingerprint. Tests46861exit0:12passed covering repair/failure/exception/no-repair,
native runner and provenance. Registered third same4MTGsmoke on40880,output
outputs/real-data-smoke/qwen25-mtg-U-P0-none-budget-repair.jsonl, batch1/temp0/
128tokens/P0/none/sdpa/quantnone/pinnedQwen. No gridscaling. Requires actualinspection.
