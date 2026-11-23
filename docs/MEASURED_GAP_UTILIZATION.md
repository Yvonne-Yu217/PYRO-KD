# Measured narration use of speech-free gaps

The proposal describes silence utilization as “ratio of description duration to available gap length.” This export provides that measured caption-level ratio and an additional bounded all-gap occupancy measure. They answer different questions and must not be substituted without naming the definition. All durations come from existing Piper waveforms accelerated to 1.2x, with waveform hashes and sample-count durations verified against the existing overlap report. Nothing is synthesized again.

For each video, union the VAD speech intervals and VAD-gated Whisper word intervals. Its complement within `[0, video duration]` defines gaps. No placement safety-margin trimming is applied. These are detector-defined speech-free gaps, not human-confirmed silence; music and environmental sound may remain. VAD-only and Whisper-only tracks are also exported.

For every gap G, the primary occupancy is the length of `(union of narration intervals) intersect G`, divided by the full length of G. This lies in [0,1]. Include unused gaps with zero occupancy and videos with no successful captions. The pooled cell ratio divides summed unique occupied seconds by summed gap seconds; macro-video and macro-gap ratios are also reported. A video with no gaps has an undefined ratio, recorded as null, and contributes zero gap duration to the pooled denominator.

The proposal-facing caption ratio divides full measured caption duration by remaining gap length from that caption's start to the gap end. Its mean and eligible/missing counts are reported. Starts in speech or outside the video have no eligible gap and are counted explicitly; zero-output videos have no caption-level observations. This ratio may exceed one. Full-gap nominal duration instead sums full durations of captions starting in the gap, divided by full gap length. Layered intersection divides the sum of individual caption intersections by gap duration; it counts concurrent narrations repeatedly. Both are separate from bounded occupancy. Excess layers are layered seconds minus unique occupied seconds.

Speech collision, out-of-video duration, and narration self-overlap are retained from the original measured interval evaluation, with exact checks against existing reports. Clipping to gaps only defines occupancy; it does not erase those failures. No output or selector is changed, and these descriptive diagnostics are not new hypothesis tests.

## Coverage and provenance

The exporter requires exact final generation-attempt IDs matching prepared moments, validates render coverage for every nonempty successful output, and retains failed/empty counts. It loads all fixed 10 dev or 30 formal test videos, including those absent from older TTS reports because they had zero valid captions. It binds config, split, raw generations, moments, media/VAD/Whisper caches, render manifest, overlap report, waveform hashes and source script; input files must remain unchanged during export. Existing output directories are never overwritten.

The development demonstration uses the original prep configuration (128-token pilot, not the later formal configuration). The compact checked-in JSON retains per-video totals and input hashes. Full per-gap details and individual collision rows remain at the scratch artifact path recorded in that JSON. No narration text or waveform is added to Git.

## Reproduction

Run from the records checkout, inside the compatible existing container, with CUDA hidden and two CPU threads:

```bash
source hpc/config.sh
in_env ro 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/export_measured_gap_utilization.py --config /scratch/yy5919/pyro-revision15-prep/configs/experiments/reframed_local_dev_pilot.toml --tts-root /scratch/yy5919/pyro-kd-derived/tts/local-dev-pilot-v1 --cache-root /scratch/yy5919/pyro-kd-derived/cache/reframed --output-dir /scratch/yy5919/pyro-kd-derived/metrics/measured-gap-dev-v2'
```

For formal core, use `pyro-revision15-formal/configs/experiments/reframed_test_local.toml`, TTS root `/scratch/yy5919/pyro-kd-derived/tts/formal-local-v1`, cache `/scratch/yy5919/pyro-kd-derived/cache/reframed-test`, and a fresh output directory. Run only after all ten cells' TTS reports are complete. `--matrix ablation5` selects the configuration's ablation-five matrix instead. These commands perform CPU measurement validation only.

Regression tests cover overlapping narration (bounded union versus unbounded layered/nominal ratios), boundary crossing with retained speech and out-of-video collisions, unused gaps, zero-output videos, fully spoken videos, half-open gap membership and invalid inputs. The six tests passed; the real dev export validated all 100 video-cell records, 705 valid renders from 804 attempted moments and 99 failed/empty outputs. Initial host-side test invocation could not resolve the container-only virtualenv interpreter; rerunning inside the existing container succeeded. The first derived demo was retained; v2 adds caption-level aggregate ratios required to distinguish the proposal definition from all-gap occupancy.
