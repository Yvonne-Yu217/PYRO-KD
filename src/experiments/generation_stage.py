"""Moments, generation (AD and QA) and generation evaluation.

Moments: every selected time becomes one generation input with the shared
frame window (4 frames over 2 s), transcript context (+/-15 s), word budget
floor(gap x 3) capped at 30, and an audio timeline around the moment. A time
with a zero budget is never sent to a model: it is recorded as
"no_feasible_gap" in the moments report.

Generation: resumable (src/generation/outputs.py decides what is settled),
batched through any backend, temperature 0. The model's text is kept as
model_output; "output" is the text after the word-cap policy, i.e. what would
be rendered. P1 refuses to run if an exemplar comes from a video being
evaluated (leakage).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from src.audio.timeline import format_timeline
from src.evaluation import placement
from src.evaluation.metrics import match_timestamps, temporal_precision_recall_f1
from src.evaluation.stats import bootstrap_mean_ci
from src.experiments.config import ConfigError, get, stable_hash
from src.experiments.data import ground_truth, load_context
from src.experiments.io import append_jsonl, latest_by_key, read_jsonl, update_status, write_json
from src.experiments.matrix import GenerationRun, SelectionRun
from src.experiments.selection_stage import selections_path
from src.generation.backends import GenRequest, GenResult
from src.generation.outputs import final_rows, is_success, settled_ids
from src.generation.prompts import EXAMPLE_PRIORS, build_prompt, build_qa_prompt, enforce_word_limit, transcript_context, word_limit_for_gap
from src.utils.audio_features import available_gap, owned_media_manifest
from src.utils.cache_checks import MEDIA_INCOMPLETE, media_cache_problem, read_manifest
from src.utils.media import cache_is_current, frame_window


def moments_path(exp_dir: Path, run: SelectionRun) -> Path:
    return exp_dir / "moments" / f"{run.run_id}.jsonl"


def _transcript_items(record) -> list:
    return [SimpleNamespace(start=float(t["start"]), end=float(t["end"]), text=str(t["text"])) for t in record.transcript or []]


def native_audio_clip(cfg: dict, wav, time: float, duration: float, visual_span: float) -> dict | None:
    """Explicit native window for controlled ablations; legacy default unchanged."""
    if wav is None:
        return None
    window = get(cfg, "generation.native_audio_window_seconds")
    if window is None:
        return {"path": str(wav), "start": max(0.0, time - visual_span / 2 - 1.0),
                "end": time + visual_span / 2 + 1.0}
    window = float(window)
    if not math.isfinite(window) or window <= 0:
        raise ConfigError("native_audio_window_seconds must be finite and positive")
    return {"path": str(wav), "start": max(0.0, time - window / 2),
            "end": min(duration, time + window / 2)}


def _current_media(cache: Path) -> tuple[dict | None, str | None]:
    manifest, reason = read_manifest(cache)
    if manifest is None:
        return None, reason
    try:
        manifest = owned_media_manifest(manifest, cache)
        problem = media_cache_problem(manifest)
    except (OSError, ValueError):
        return None, MEDIA_INCOMPLETE
    return (None, problem) if problem else (manifest, None)


def build_moments(cfg: dict, run: SelectionRun, records, exp_dir: Path) -> Path:
    """(Re)write the moments file for one selection run from its selections."""
    selections = latest_by_key(read_jsonl(selections_path(exp_dir, run)), "video_id")
    num_frames = int(get(cfg, "selection.num_frames", 4))
    span = float(get(cfg, "selection.span_seconds", 2.0))
    window = float(get(cfg, "generation.timeline_window_seconds", 30.0))
    rows, report = [], []
    for record in records:
        sel = selections.get(record.video_id)
        entry = {"video_id": record.video_id, "selection_status": sel["status"] if sel else "not_selected",
                 "written": 0, "no_feasible_gap": [], "status": "ok"}
        report.append(entry)
        if not sel or sel["status"] not in ("ok", "ok_degraded"):
            entry["status"] = entry["selection_status"]
            continue
        cache = Path(get(cfg, "paths.cache_root")) / record.video_id
        manifest, reason = _current_media(cache)
        if manifest is None:
            entry["status"] = reason
            continue
        context, available = load_context(record, cfg, run.spec.get("cue_set", "audio_cues.json"), need_visual=False)
        speech = [tuple(s) for s in context.audio_features["speech"]] if context.audio_features else None
        wav = manifest["audio_path"]
        cues = context.audio_cues
        transcript = _transcript_items(record)
        for i, t in enumerate(sel["times"]):
            gap = available_gap(t, speech, context.duration) if speech is not None else None
            limit = word_limit_for_gap(gap)
            if limit <= 0:
                entry["no_feasible_gap"].append(t)
                continue
            frames = frame_window(manifest, t, num_frames, span)
            timeline = ""
            if cues is not None:
                timeline = format_timeline(cues.get("events", []), (cues.get("asr") or {}).get("segments", []),
                                           t - window / 2, t + window / 2)
            rows.append({
                "id": f"{record.video_id}_{run.name}_{i:03d}", "video_id": record.video_id, "selector": run.name,
                "selection_run_id": run.run_id, "time": t, "k_requested": sel["k"],
                "frames": [f.path for f in frames], "frame_times": [f.timestamp for f in frames],
                "media_fingerprint": manifest["settings"]["source_fingerprint"],
                "transcript": transcript_context(transcript, t), "word_limit": limit,
                "word_limit_source": "vad" if speech is not None else "fallback_no_vad",
                "audio_timeline": timeline, "timeline_status": "ok" if cues is not None else available.get("audio_cues"),
                "audio_clip": native_audio_clip(cfg, wav, t, context.duration, span),
            })
            entry["written"] += 1
    path = moments_path(exp_dir, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)
    write_json(path.with_suffix(".report.json"), {"selection_run_id": run.run_id, "videos": report})
    return path


def generations_path(exp_dir: Path, run: GenerationRun, dry_run: bool = False) -> Path:
    """Where a run's generations live. Dry runs (echo backend) get their own directory so their
    placeholder rows can never be mistaken for model output by a later real run's resume."""
    return exp_dir / "generations" / ("dry_run" if dry_run else "") / f"{run.run_id}.jsonl"


def check_exemplar_leakage(exemplars: dict, video_ids: set[str]) -> None:
    leaked = sorted({e["video_id"] for e in exemplars["exemplars"]} & video_ids)
    if leaked:
        raise ConfigError(f"P1 exemplars come from evaluated videos {leaked}: rebuild them from the exemplar pool")


def _ad_inputs(cfg: dict, run: GenerationRun, exp_dir: Path, records) -> dict[str, tuple[dict, GenRequest, str]]:
    """Prepare the requests and fingerprints used by both resume and evaluation."""
    path = moments_path(exp_dir, run.selection)
    moments = read_jsonl(path)
    exemplars = None
    if run.prior in EXAMPLE_PRIORS:
        from src.generation.priors import load_exemplars

        exemplars = load_exemplars(get(cfg, "generation.exemplars_path", "configs/p1_exemplars.json"))
        check_exemplar_leakage(exemplars, {m["video_id"] for m in moments} | {r.video_id for r in records or []})
    mode = run.audio_context
    has_audio_input = run.model.get("backend") in ("qwen_omni", "echo")
    if mode == "native" and not has_audio_input:
        raise ConfigError(f"The native audio ablation needs an audio-capable backend for {run.model_label}")
    max_new = int(get(cfg, "generation.max_new_tokens", 64))
    span = float(get(cfg, "selection.span_seconds", 2.0))
    policy = get(cfg, "generation.word_limit_policy", "truncate")
    digests: dict[str, str | None] = {}
    checked_media: set[str] = set()
    inputs = {}

    def file_digest(name: str) -> str | None:
        if name not in digests:
            try:
                with Path(name).open("rb") as stream:
                    digests[name] = hashlib.file_digest(stream, "sha256").hexdigest()
            except OSError:
                digests[name] = None  # the backend reports missing/unreadable request files
        return digests[name]

    for m in moments:
        moment_id = m["id"]
        if moment_id in inputs:
            raise ConfigError(f"Duplicate moment ID {moment_id!r} in {path}")
        media = m.get("media_fingerprint")
        media_key = stable_hash(media, 64)
        if media_key not in checked_media:
            if not isinstance(media, dict) or not cache_is_current({"source": media.get("path"),
                                                                     "settings": {"source_fingerprint": media}}):
                raise ConfigError(f"Media is stale or unverifiable for moment {moment_id!r} in {path}. "
                                  "Refresh the media cache and rerun the moments stage.")
            checked_media.add(media_key)
        use_timeline = mode == "timeline" or (mode is None and run.timeline)
        prompt = build_prompt(run.prior, m["time"], len(m["frames"]), span, m["word_limit"],
                              m.get("transcript", "") if mode is None and not run.timeline else "",
                              audio_timeline=m.get("audio_timeline", "") if use_timeline else "", exemplars=exemplars)
        audio = m.get("audio_clip") if mode == "native" or (mode is None and has_audio_input) else None
        request = GenRequest(moment_id, prompt, list(m["frames"]), audio, max_new)
        timeline_status = m.get("timeline_status") if use_timeline else None
        digest = stable_hash([run.run_id, m["video_id"], m["time"], m["frame_times"], media, prompt, max_new, policy,
                              timeline_status,
                              [(name, file_digest(name)) for name in request.images],
                              (audio, file_digest(audio["path"])) if audio else None], 64)
        inputs[moment_id] = m, request, digest
    return inputs


def _current_generation_rows(path: Path, inputs: dict[str, tuple[dict, GenRequest, str]]) -> list[dict]:
    rows = []
    for row in read_jsonl(path):
        current = inputs.get(row.get("id"))
        if current and row.get("input_fingerprint") == current[2]:
            rows.append(row)
        elif is_success(row):
            raise ConfigError(f"Generation inputs changed or cannot be verified for moment {row.get('id')!r} in {path}. "
                              "Preserve the old outputs and generate into a fresh experiment directory.")
        # Failures from older inputs are retained on disk but cannot settle this request.
    return rows


def run_generation(
    cfg: dict, run: GenerationRun, exp_dir: Path, backend_factory: Callable[[dict], Any], records=None,
    limit: int | None = None, dry_run: bool = False,
) -> Path:
    """Generate AD for every moment of the run's selection; resume only matching requests."""
    out_path = generations_path(exp_dir, run, dry_run)
    stage = "generate_dry_run" if dry_run else "generate"
    max_new = int(get(cfg, "generation.max_new_tokens", 64))
    inputs = _ad_inputs(cfg, run, exp_dir, records)
    done, _ = settled_ids(_current_generation_rows(out_path, inputs), max_new, 0.0)
    todo = [entry for m_id, entry in list(inputs.items())[:limit or None] if m_id not in done]
    policy = get(cfg, "generation.word_limit_policy", "truncate")
    if not todo:
        update_status(exp_dir, stage, run.run_id, {"todo": 0, "done": len(done)})
        return out_path
    spec = run.model
    backend = None
    batch_size = int(spec.get("request_batch", 8))
    n_ok = n_err = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        requests, rejected, base = [], [], {}
        for m, request, digest in batch:
            if run.audio_context == "native" and not request.audio:
                rejected.append(GenResult(m["id"], error="no_native_audio"))
            elif run.audio_context == "timeline" and m.get("timeline_status") != "ok":
                rejected.append(GenResult(m["id"], error="unavailable_audio_timeline"))
            else:
                requests.append(request)
            base[m["id"]] = {"id": m["id"], "video_id": m["video_id"], "selector": m["selector"], "time": m["time"],
                             "word_limit": m["word_limit"], "frame_times": m["frame_times"], "prior": run.prior,
                             "model": run.model_label, "timeline": run.timeline,
                             "timeline_status": m.get("timeline_status") if run.timeline else None,
                             "audio_context": run.audio_context, "input_fingerprint": digest,
                             "run_id": run.run_id, "temperature": 0.0, "max_new_tokens": max_new, "prompt": request.prompt}
        rows = []
        generated = []
        if requests:
            if backend is None:
                backend = backend_factory(spec)
            generated = backend.generate(requests)
        for r in [*generated, *rejected]:
            row = {**base[r.id], "model_id": getattr(backend, "model_id", spec.get("model")),
                   "backend": getattr(backend, "name", spec.get("backend")), **r.to_dict()}
            row["model_output"] = row.pop("text", "")
            if r.error is None:
                capped = enforce_word_limit(row["model_output"], base[r.id]["word_limit"], policy)
                row.update(output=capped["text"], words=capped["words"], raw_words=capped["raw_words"],
                           over_budget=capped["over_budget"], truncated=capped["truncated"])
                if capped["over_budget"] and policy == "flag":
                    row["error"] = "word_budget: generated output exceeds the declared limit"
                    n_err += 1
                else:
                    n_ok += 1
            else:
                row["output"] = row["model_output"]
                n_err += 1
            rows.append(row)
        append_jsonl(out_path, rows)
    update_status(exp_dir, stage, run.run_id, {"ok": n_ok, "errors": n_err, "previously_done": len(done)})
    return out_path


def _optional_scorer(name: str, cfg: dict):
    """Build a text scorer lazily; (callable | None, reason)."""
    try:
        if name == "bertscore":
            from bert_score import score

            lang = get(cfg, "evaluation.bertscore_lang", "en")
            return (lambda c, r: [float(x) for x in score(c, r, lang=lang, verbose=False)[2]]), None
        if name == "sentence_cosine":
            from sentence_transformers import SentenceTransformer

            from src.evaluation.cosine_similarity import DEFAULT_MODEL_NAME

            model = SentenceTransformer(get(cfg, "evaluation.sentence_model", DEFAULT_MODEL_NAME))

            def cos(c, r):
                emb = model.encode(list(c) + list(r), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
                return [float(a @ b) for a, b in zip(emb[: len(c)], emb[len(c):])]
            return cos, None
        if name == "clip_text":
            from src.evaluation.clip_text import openclip_text_encoder, pair_similarities

            encoder = openclip_text_encoder()
            return (lambda c, r: pair_similarities(c, r, encoder)), None
    except Exception as exc:  # missing package or weights
        return None, f"unavailable: {exc!r}"
    return None, f"unknown scorer {name}"


def budget_compliance(rows: list[dict], missing_count: int = 0) -> dict:
    """Budget rate among complete decodes with explicit boolean budget flags.

    A word-budget flag failure is a complete decode and belongs in this rate.
    Token-limit/backend failures and legacy rows without a flag remain unknown;
    missing expected predictions are separate, never silently counted as passes.
    """
    known = []
    for row in rows:
        error = row.get("error")
        complete_decode = (not row.get("hit_token_limit")
                           and (error is None or str(error).startswith("word_budget:"))
                           and isinstance(row.get("output"), str))
        if complete_decode and isinstance(row.get("over_budget"), bool):
            known.append(row)
    n_over = sum(row["over_budget"] for row in known)
    total = len(rows) + missing_count
    return {"n_known": len(known), "n_over_budget": n_over,
            "n_unknown_generated": len(rows) - len(known), "n_missing": missing_count,
            "known_fraction": len(known) / total if total else None,
            "rate": n_over / len(known) if known else None}


def evaluate_generation(cfg: dict, run: GenerationRun, records, exp_dir: Path, scorers: dict | None = None) -> dict:
    """Placement with real word counts, budget compliance and matched-pair text metrics."""
    by_id = {r.video_id: r for r in records}
    inputs = _ad_inputs(cfg, run, exp_dir, records)
    rows = final_rows(_current_generation_rows(generations_path(exp_dir, run), inputs))
    expected_by_video: dict[str, set[str]] = {}
    for moment_id, (moment, _, _) in inputs.items():
        expected_by_video.setdefault(moment["video_id"], set()).add(moment_id)
    missing_ids = sorted(set(inputs) - {r["id"] for r in rows})
    tol = float(get(cfg, "selection.tolerance_seconds", 2.0))
    slack = float(get(cfg, "selection.duration_slack_seconds", 1.0))
    wps, speed = float(get(cfg, "evaluation.tts_words_per_second", 2.5)), float(get(cfg, "evaluation.tts_speed", 1.2))
    names = get(cfg, "evaluation.text_metrics", ["bertscore", "sentence_cosine", "clip_text"]) or []
    if scorers is None:
        scorers = {}
        for name in names:
            scorers[name] = _optional_scorer(name, cfg)
    per_video, pair_texts = [], []
    by_video: dict[str, list[dict]] = {}
    for row in rows:
        by_video.setdefault(row.get("video_id"), []).append(row)
    for video_id, record in by_id.items():
        vrows = by_video.get(video_id, [])
        expected_ids = expected_by_video.get(video_id, set())
        missing = sorted(expected_ids - {r["id"] for r in vrows})
        ok = [r for r in vrows if is_success(r) and r.get("output")]
        context, _ = load_context(record, cfg, need_visual=False)
        truth, _ = ground_truth(record, context.duration, slack)
        times = [r["time"] for r in ok]
        primary = temporal_precision_recall_f1(times, truth, tol)
        speech = [tuple(s) for s in context.audio_features["speech"]] if context.audio_features else None
        placements = [(r["time"], len(r["output"].split())) for r in ok]
        actual_ids = {r["id"] for r in vrows}
        budget = budget_compliance(vrows, len(missing))
        entry = {"video_id": video_id, "n_expected": len(expected_ids), "n_missing": len(missing),
                 "missing_ids": missing, "no_prepared_moments": not expected_ids,
                 "n_moments": len(vrows), "n_ok": len(ok), "n_error": len(vrows) - len(ok),
                 "precision": primary["precision"], "recall": primary["recall"], "f1": primary["f1"],
                 "mean_words": (sum(n for _, n in placements) / len(placements)) if placements else None,
                 "over_budget_rate": budget["rate"], "budget_compliance": budget,
                 "overlap_seconds": (placement.speech_overlap(placements, speech, wps, speed)["overlap_seconds"]
                                     if speech is not None else None)}
        entry.update(n_expected=len(expected_ids), missing_ids=sorted(expected_ids - actual_ids),
                     unexpected_ids=sorted(actual_ids - expected_ids),
                     no_generation_rows=not bool(vrows))
        per_video.append(entry)
        gt_items = sorted(record.ad or [], key=lambda e: e.start)
        gt_items = [e for e in gt_items if e.start in set(truth)]
        for m in match_timestamps(times, [e.start for e in gt_items], tol):
            pair_texts.append((video_id, ok[m.prediction_index]["output"], gt_items[m.ground_truth_index].text))
    text_metrics: dict[str, Any] = {}
    for name, (fn, reason) in scorers.items():
        if fn is None:
            text_metrics[name] = {"status": reason}
            continue
        if not pair_texts:
            text_metrics[name] = {"status": "no matched pairs"}
            continue
        try:
            values = fn([p[1] for p in pair_texts], [p[2] for p in pair_texts])
            per_vid: dict[str, list[float]] = {}
            for (vid, _, _), v in zip(pair_texts, values):
                per_vid.setdefault(vid, []).append(v)
            text_metrics[name] = {"status": "ok", "n_pairs": len(values),
                                  "per_video_mean": bootstrap_mean_ci([sum(v) / len(v) for v in per_vid.values()])}
        except Exception as exc:
            text_metrics[name] = {"status": f"failed: {exc!r}"}
    resamples, seed = int(get(cfg, "evaluation.resamples", 10000)), int(get(cfg, "experiment.seed", 0))
    budget_totals = {key: sum(v["budget_compliance"][key] for v in per_video)
                     for key in ("n_known", "n_over_budget", "n_unknown_generated", "n_missing")}
    budget_total = budget_totals["n_known"] + budget_totals["n_unknown_generated"] + budget_totals["n_missing"]
    budget_totals["known_fraction"] = budget_totals["n_known"] / budget_total if budget_total else None
    budget_totals["rate"] = (budget_totals["n_over_budget"] / budget_totals["n_known"]
                             if budget_totals["n_known"] else None)


    no_inputs = sorted(v for v in by_id if not expected_by_video.get(v))
    n_errors = sum(v["n_error"] for v in per_video)
    status = "incomplete" if missing_ids or no_inputs else ("complete_with_errors" if n_errors else "complete")
    result = {
        "status": status, "n_expected": sum(len(expected_by_video.get(v, set())) for v in by_id),
        "n_observed": sum(v["n_moments"] for v in per_video), "n_missing": len(missing_ids),
        "missing_ids": missing_ids, "videos_without_prepared_moments": no_inputs,
        "n_videos_expected": len(by_id), "n_videos_observed": sum(bool(by_video.get(v)) for v in by_id),
        "n_error": n_errors,
        "run_id": run.run_id, "selector": run.selection.name, "prior": run.prior, "model": run.model_label,
        "timeline": run.timeline, "audio_context": run.audio_context, "oracle": run.selection.is_oracle, "n_videos": len(per_video),
        "expected_videos": len(by_id),
        "missing_video_ids": sorted(set(by_id) - set(by_video)),
        "unexpected_video_ids": sorted(set(by_video) - set(by_id)),
        "expected_moments": len(inputs),
        "missing_moment_ids": missing_ids,
        "macro_f1": bootstrap_mean_ci([v["f1"] for v in per_video], resamples, seed),
        "overlap_seconds": bootstrap_mean_ci([v["overlap_seconds"] for v in per_video], resamples, seed),
        "over_budget_rate": bootstrap_mean_ci([v["over_budget_rate"] for v in per_video], resamples, seed),
        "budget_compliance": {**budget_totals,
            "denominator": "complete decoded rows with explicit boolean over_budget flags; observed evaluated-video rows",
            "unknown_policy": "token/backend failures or absent flags are unknown; missing expected predictions are separate"},
        "text_metrics": text_metrics, "videos": per_video,
    }
    write_json(exp_dir / "eval" / f"generation_{run.run_id}.json", result)
    return result


# --- QA -------------------------------------------------------------------------

def qa_frames(times: Sequence[float], duration: float, max_frames: int) -> list[float]:
    """Selected times, subsampled evenly by order to at most max_frames (always includes the ends)."""
    times = sorted(times)
    if len(times) <= max_frames:
        return times
    if max_frames == 1:
        return [times[len(times) // 2]]
    return [times[round(i * (len(times) - 1) / (max_frames - 1))] for i in range(max_frames)]


def run_qa(
    cfg: dict, run: GenerationRun, records, exp_dir: Path, backend_factory: Callable[[dict], Any],
    dry_run: bool = False,
) -> Path:
    """Zero-shot multiple-choice QA over the run's selected keyframes."""
    from src.evaluation.qa import parse_choice

    selections = latest_by_key(read_jsonl(selections_path(exp_dir, run.selection)), "video_id")
    out_path = generations_path(exp_dir, run, dry_run)
    done = {r["question_id"] for r in read_jsonl(out_path) if r.get("status") == "answered"}
    max_frames = int(get(cfg, "qa.max_frames", 32))
    todo = []
    for record in records:
        for item in record.qa:
            if item.question_id not in done:
                todo.append((record, item))
    if not todo:
        return out_path
    backend = backend_factory(run.model)
    for record, item in todo:
        sel = selections.get(record.video_id)
        base = {"question_id": item.question_id, "video_id": record.video_id, "selector": run.selection.name,
                "prior": run.prior, "model": run.model_label, "timeline": run.timeline, "run_id": run.run_id,
                "answer": item.answer}
        if not sel or sel["status"] not in ("ok", "ok_degraded") or not sel["times"]:
            append_jsonl(out_path, [{**base, "status": "no_selection", "prediction": None,
                                     "selection_status": sel["status"] if sel else "not_selected"}])
            continue
        # One unusable video or one failed request must not stop the questions that follow.
        try:
            cache = Path(get(cfg, "paths.cache_root")) / record.video_id
            manifest, reason = _current_media(cache)
            if manifest is None:
                append_jsonl(out_path, [{**base, "status": "no_frames", "prediction": None, "reason": reason}])
                continue
            times = qa_frames(sel["times"], sel["duration"], max_frames)
            frames = [frame_window(manifest, t, 1, 0.0)[0] for t in times]
            timeline = ""
            if run.timeline:
                context, _ = load_context(record, cfg, need_visual=False)
                if context.audio_cues is not None:
                    timeline = format_timeline(context.audio_cues.get("events", []),
                                               (context.audio_cues.get("asr") or {}).get("segments", []),
                                               config={"max_items": int(get(cfg, "qa.timeline_max_items", 40))})
            prompt = build_qa_prompt(item.question, item.options, [f.timestamp for f in frames], timeline)
            result = backend.generate([GenRequest(item.question_id, prompt, [f.path for f in frames], None,
                                                  int(get(cfg, "qa.max_new_tokens", 8)))])[0]
        except Exception as exc:  # recorded; the batch continues
            append_jsonl(out_path, [{**base, "status": "error", "prediction": None, "error": f"{type(exc).__name__}: {exc}"}])
            continue
        row = {**base, "duration": sel["duration"], "frame_times": [f.timestamp for f in frames],
               "prompt": prompt, "reply": result.text, "error": result.error}
        row["prediction"] = None if result.error else parse_choice(result.text, item.options)
        row["status"] = "error" if result.error else "answered"
        append_jsonl(out_path, [row])
    return out_path


def evaluate_qa(cfg: dict, runs: Sequence[GenerationRun], exp_dir: Path, dry_run: bool = False) -> dict:
    from src.evaluation.qa import accuracy_by_length, paired_condition_comparison, score_items

    results: dict[str, Any] = {}
    final: dict[str, list[dict]] = {}
    for run in runs:
        rows = list(latest_by_key(read_jsonl(generations_path(exp_dir, run, dry_run)), "question_id").values())
        scored = [r for r in rows if r.get("answer") is not None]
        final[run.run_id] = scored
        results[run.run_id] = {"selector": run.selection.name, "model": run.model_label, "timeline": run.timeline,
                               **score_items(scored), "by_length": accuracy_by_length(scored)}
    pairs = []
    for a in runs:
        for b in runs:
            if (a.selection.run_id == b.selection.run_id and a.model_label == b.model_label and a.prior == b.prior
                    and a.timeline and not b.timeline):
                pairs.append({"selector": a.selection.name, "model": a.model_label,
                              "timeline_on_minus_off": paired_condition_comparison(final[a.run_id], final[b.run_id])})
    out = {"runs": results, "timeline_comparisons": pairs}
    write_json(exp_dir / "eval" / ("qa_metrics.dry_run.json" if dry_run else "qa_metrics.json"), out)
    return out
