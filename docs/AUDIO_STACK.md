# Three-stream audio preparation

The current container already provides Silero VAD and Transformers. A separate
system-site-packages venv at `/scratch/yy5919/pyro-audio-env` adds
`faster-whisper==1.2.1`; the shared overlay is unchanged. Installed dependencies
and download logs are under `outputs/verification/audio-*`.

Pinned model repositories/revisions and expected file sizes are in
`configs/audio_models.json`. Downloads total about 1.45GB (AST's duplicate
PyTorch weights are omitted in favor of safetensors). The official
[faster-whisper documentation](https://github.com/SYSTRAN/faster-whisper)
describes CPU int8 and word timestamps; the
[AST model card](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)
identifies its AudioSet classifier, and the
[CLAP documentation](https://huggingface.co/docs/transformers/model_doc/clap)
describes projected audio embeddings.

Whisper-small.en is the initial frozen English speech checkpoint; CPU int8 is
recorded explicitly. AST and CLAP run float32 in the CPU check. This is a
preparation choice, not a quality comparison or a passed human-label gate.

```bash
source hpc/config.sh
in_env ro 'python -m venv --system-site-packages /scratch/yy5919/pyro-audio-env && /scratch/yy5919/pyro-audio-env/bin/python -m pip install faster-whisper==1.2.1'
in_env ro 'HF_HUB_OFFLINE=0 python scripts/download_audio_models.py --cache-dir /scratch/yy5919/hf_cache/hub --report outputs/verification/audio-model-download.json'
in_env ro 'OMP_NUM_THREADS=4 /scratch/yy5919/pyro-audio-env/bin/python scripts/check_audio_models.py --cache-dir /scratch/yy5919/pyro-kd-derived/cache/reframed --model-report outputs/verification/audio-model-download.json --report outputs/verification/audio-model-offline-check.json'
```

Actual offline CPU checks passed: first 10seconds of development clip
l5lLQICPOcnb produced finite AST 527-class scores and CLAP 512-dimensional
embeddings. CLAP embeddings are not unit-normalized by this audio-only model;
normalize or use explicit cosine distance when computing novelty. Loading the
audio-only CLAP class reports unused text-model weights; no missing audio
weights were reported. For Whisper, development clip CxWby8RuUwU6's VAD interval
[0.5,3.3] produced 10 word timestamps within its boundaries. No non-VAD samples
were passed to Whisper. Evidence: audio-model-offline-check.json/.log.

Remaining: implement cached one-second-hop event tracks, exclude speech/silence
classes from event scoring, VAD-gated full transcripts with mapped word times,
CLAP novelty, timeline integration, and human-labeled detector evaluation.
The successful compatibility check proves none of those full-pipeline claims.
