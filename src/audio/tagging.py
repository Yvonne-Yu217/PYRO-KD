"""Frame-level AudioSet tagging and CLAP embeddings (proposal §2, third stream).

Both run on sliding windows centred every hop_seconds (default 1 s, as in
§3.1) and return (times, values) on the video clock. Backends are built
lazily so this module imports without torch/transformers:

- AST (MIT/ast-finetuned-audioset-10-10-0.4593) via transformers: sigmoid
  posteriors over the 527 AudioSet classes. Chosen over BEATs/PANNs because it
  ships as a transformers checkpoint (one dependency, offline-cacheable on HPC).
  This is a deviation in tool choice only; the proposal allows any of the three.
- CLAP (laion/clap-htsat-unfused) via transformers: audio embeddings at 48 kHz.

Untested against real checkpoints on this laptop (no transformers installed);
the windowing and output contract are unit-tested with injected models.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

DEFAULT_TAGGER_CONFIG = {
    "backend": "ast",
    "model": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "window_seconds": 3.0,
    "hop_seconds": 1.0,
    "batch_size": 32,
    "keep_top_labels": 64,      # labels stored per video (by max posterior)
    "keep_min_posterior": 0.05,
}
DEFAULT_CLAP_CONFIG = {
    "backend": "clap",
    "model": "laion/clap-htsat-unfused",
    "window_seconds": 3.0,
    "hop_seconds": 1.0,
    "batch_size": 32,
}

# windows (list of float32 mono arrays at sample_rate) -> (n, C) posteriors or (n, D) embeddings
WindowModel = Callable[[list, int], "object"]


def window_centres(duration: float, hop: float) -> list[float]:
    if hop <= 0:
        raise ValueError("hop must be positive")
    return [round(i * hop, 6) for i in range(max(1, int(math.floor(duration / hop)) + 1)) if i * hop <= duration]


def sliding_windows(waveform, sample_rate: int, centres: Sequence[float], window_seconds: float) -> list:
    """Zero-padded windows of window_seconds centred on each time."""
    import numpy as np

    x = np.asarray(waveform, dtype=np.float32)
    width = int(round(window_seconds * sample_rate))
    out = []
    for c in centres:
        start = int(round((c - window_seconds / 2) * sample_rate))
        chunk = np.zeros(width, dtype=np.float32)
        lo, hi = max(0, start), min(len(x), start + width)
        if hi > lo:
            chunk[lo - start:hi - start] = x[lo:hi]
        out.append(chunk)
    return out


def run_windowed(
    waveform, sample_rate: int, model: WindowModel, window_seconds: float, hop_seconds: float, batch_size: int = 32,
):
    """(times, stacked model outputs) over sliding windows."""
    import numpy as np

    duration = len(waveform) / sample_rate
    centres = window_centres(duration, hop_seconds)
    windows = sliding_windows(waveform, sample_rate, centres, window_seconds)
    rows = []
    for i in range(0, len(windows), batch_size):
        batch = np.asarray(model(windows[i:i + batch_size], sample_rate), dtype=np.float32)
        if batch.ndim != 2 or len(batch) != len(windows[i:i + batch_size]):
            raise ValueError(f"window model returned shape {batch.shape} for {len(windows[i:i + batch_size])} windows")
        rows.append(batch)
    return centres, (np.concatenate(rows) if rows else np.zeros((0, 0), dtype=np.float32))


def keep_labels(labels: Sequence[str], posteriors, top: int, min_posterior: float):
    """Restrict to the `top` labels by max posterior that reach min_posterior."""
    import numpy as np

    p = np.asarray(posteriors)
    if p.size == 0:
        return [], p
    peaks = p.max(axis=0)
    order = [int(i) for i in np.argsort(-peaks, kind="stable") if peaks[i] >= min_posterior][:top]
    order.sort()
    return [labels[i] for i in order], p[:, order]


def ast_model(model_name: str = DEFAULT_TAGGER_CONFIG["model"], device: str | None = None):
    """(window_model, labels) for an AST AudioSet checkpoint; sigmoid posteriors."""
    import numpy as np
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ASTFeatureExtractor.from_pretrained(model_name)
    model = ASTForAudioClassification.from_pretrained(model_name).to(device).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    def run(windows, sample_rate):
        if sample_rate != extractor.sampling_rate:
            raise ValueError(f"AST expects {extractor.sampling_rate} Hz audio, got {sample_rate}")
        inputs = extractor(list(windows), sampling_rate=sample_rate, return_tensors="pt").to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
            logits = model(**inputs).logits
        return torch.sigmoid(logits.float()).cpu().numpy().astype(np.float32)

    return run, labels


def clap_model(model_name: str = DEFAULT_CLAP_CONFIG["model"], device: str | None = None):
    """window_model returning unit CLAP audio embeddings (input resampled to 48 kHz)."""
    import librosa
    import numpy as np
    import torch
    from transformers import ClapModel, ClapProcessor

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name).to(device).eval()
    target = processor.feature_extractor.sampling_rate

    def run(windows, sample_rate):
        audio = [librosa.resample(np.asarray(w, dtype=np.float32), orig_sr=sample_rate, target_sr=target) for w in windows]
        try:
            inputs = processor(audio=audio, sampling_rate=target, return_tensors="pt")
        except TypeError:  # older transformers name the argument "audios"
            inputs = processor(audios=audio, sampling_rate=target, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            emb = model.get_audio_features(**inputs).float()
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return emb.cpu().numpy().astype(np.float32)

    return run
