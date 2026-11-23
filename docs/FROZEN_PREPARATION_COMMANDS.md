# Frozen CPU preparation

These commands require verified complete source caches. They do not inspect
heldout outcomes or change membership. Run from `work/revision15-formal-prep`
after committing, using CPU allocations and `source hpc/config.sh`. API cost0.
Use the existing dev settings min-silence5/silence-scale30; do not retune on test.

## REFRAMED test (30 frozen videos)

```bash
in_env ro 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python scripts/prepare_speech_cues.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed-test --subset test --split-config configs/reframed_splits.json --min-silence 5 --silence-scale 30 --output /scratch/yy5919/pyro-kd-derived/reframed-test-speech-cues-v1.json'
in_env ro 'CUDA_VISIBLE_DEVICES="" python scripts/bridge_hpc_audio_caches.py --cache-root /scratch/yy5919/pyro-kd-derived/cache/reframed-test --subset test --splits configs/reframed_splits.json --speech-cues /scratch/yy5919/pyro-kd-derived/reframed-test-speech-cues-v1.json --output-root /scratch/yy5919/pyro-kd-derived/cache/reframed-test-pr22-bridge-v1'
```

Speech preparation needs media/VAD/speech caches; the bridge additionally needs
events/CLAP/onsets. Depend on both source feature jobs before bridging.

## Video-MME long (12 frozen videos)

```bash
in_env ro 'CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python scripts/prepare_speech_cues.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/video-mme --subset long --split-config configs/video_mme_long_subset.json --min-silence 5 --silence-scale 30 --output /scratch/yy5919/pyro-kd-derived/video-mme-long-speech-cues-v1.json'
in_env ro 'CUDA_VISIBLE_DEVICES="" python scripts/bridge_hpc_audio_caches.py --cache-root /scratch/yy5919/pyro-kd-derived/cache/video-mme --subset long --splits configs/video_mme_long_subset.json --speech-cues /scratch/yy5919/pyro-kd-derived/video-mme-long-speech-cues-v1.json --output-root /scratch/yy5919/pyro-kd-derived/cache/video-mme-pr22-bridge-v1'
```

## EgoSchema heldout (490 frozen questions)

After all selected media are prepared, repeat for `U`, `V_local`, `V_memory` with
distinct output paths; V additionally requires complete visual features.

```bash
in_env ro 'CUDA_VISIBLE_DEVICES="" python scripts/build_egoschema_qa_inputs.py --questions /scratch/yy5919/pyro-kd-derived/datasets/egoschema/questions.json --cache-dir /scratch/yy5919/pyro-kd-derived/cache/egoschema --split heldout --selector U --output /scratch/yy5919/pyro-kd-derived/moments/egoschema-heldout-U-v1.jsonl'
```

These are preparation commands, not records of completed runs. Existing output
paths are preserved; inspect their provenance before reuse rather than replacing.
