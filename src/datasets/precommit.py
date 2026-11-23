"""Proposal §6 pre-commitment checks, as code that records evidence.

Each check returns {"check", "status": "pass" | "fail" | "blocked", "evidence",
"if_fail"}. "blocked" means the inputs are not available yet (data not
transferred, access not granted); it is never reported as a pass.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Sequence

IF_FAIL = {
    "egoschema_audio": "Treat EgoSchema as visual-only (V vs U) and rely on the long-form audio benchmark for RQ4.",
    "cue2narrate_access": "Replace the baseline comparison with REFRAMED/MTG-AD plus re-implemented baselines; drop Cue2Narrate claims.",
    "longform_qa": "Fall back to concatenated clips for captioning; report QA on EgoSchema only.",
    "detector_sanity": "Raise thresholds, restrict to reliable classes (music, applause, laughter), or rely on CLAP change.",
}


def _result(check: str, status: str, evidence: dict) -> dict:
    return {"check": check, "status": status, "evidence": evidence, "if_fail": IF_FAIL[check]}


def egoschema_audio(video_paths: Sequence[str | Path], sample: int = 20, seed: int = 0, probe=None) -> dict:
    """Probe a seeded sample of EgoSchema clips for an audio stream."""
    paths = [Path(p) for p in video_paths if Path(p).is_file()]
    if not paths:
        return _result("egoschema_audio", "blocked", {"reason": "no EgoSchema videos present"})
    if probe is None:
        from src.utils.media import probe
    chosen = sorted(random.Random(seed).sample(sorted(paths), min(sample, len(paths))))
    with_audio, failed = [], []
    for p in chosen:
        try:
            if probe(p).has_audio:
                with_audio.append(p.name)
        except Exception as exc:  # unreadable file
            failed.append(f"{p.name}: {exc}")
    probed = len(chosen) - len(failed)
    status = "pass" if probed and len(with_audio) == probed else "fail"
    return _result("egoschema_audio", status, {"sampled": len(chosen), "probed": probed,
                                               "with_audio": len(with_audio), "probe_failures": failed})


def cue2narrate_access(root: str | Path | None, baseline_scores: str | Path | None = None) -> dict:
    """Data and published baseline numbers must both be on disk to pass."""
    if not root or not Path(root).exists():
        return _result("cue2narrate_access", "blocked", {"reason": "Cue2Narrate data not available locally"})
    files = [p for p in Path(root).rglob("*") if p.is_file()]
    have_scores = bool(baseline_scores) and Path(baseline_scores).is_file()
    status = "pass" if files and have_scores else "fail"
    return _result("cue2narrate_access", status, {
        "files": len(files), "baseline_scores_file": str(baseline_scores) if baseline_scores else None,
        "baseline_scores_present": have_scores,
        "note": "comparability (same split, metric definitions) must still be confirmed by a person",
    })


def longform_qa(records, min_seconds: float = 600.0, probe=None, sample: int = 20, seed: int = 0) -> dict:
    """Videos present, long enough, and carrying audio (license is a human check)."""
    with_path = [r for r in records if r.video_path is not None]
    if not with_path:
        return _result("longform_qa", "blocked", {"reason": "no long-form QA videos present"})
    if probe is None:
        from src.utils.media import probe
    chosen = random.Random(seed).sample(with_path, min(sample, len(with_path)))
    infos = []
    for r in chosen:
        try:
            info = probe(r.video_path)
            infos.append((info.duration, info.has_audio))
        except Exception:
            infos.append((None, False))
    long_audio = sum(1 for d, a in infos if d is not None and d >= min_seconds and a)
    status = "pass" if long_audio == len(infos) else "fail"
    return _result("longform_qa", status, {"sampled": len(infos), "long_with_audio": long_audio,
                                           "min_seconds": min_seconds, "license": "confirm manually"})


def detector_sanity(hand_labels_path: str | Path | None, pr_result: dict | None = None,
                    min_clips: int = 10, min_precision: float = 0.5) -> dict:
    """Needs the hand-labeled subset (~10 clips) and the detector P/R on it."""
    if not hand_labels_path or not Path(hand_labels_path).is_file():
        return _result("detector_sanity", "blocked", {"reason": "no hand-labeled audio events yet (team task)"})
    if pr_result is None:
        return _result("detector_sanity", "blocked", {"reason": "detector output not computed for the labeled clips"})
    n = pr_result.get("n_videos", 0)
    precision = (pr_result.get("pooled") or {}).get("precision")
    status = "pass" if n >= min_clips and precision is not None and precision >= min_precision else "fail"
    return _result("detector_sanity", status, {"labeled_clips": n, "pooled": pr_result.get("pooled"),
                                               "per_class": pr_result.get("per_class"),
                                               "thresholds": {"min_clips": min_clips, "min_precision": min_precision}})
