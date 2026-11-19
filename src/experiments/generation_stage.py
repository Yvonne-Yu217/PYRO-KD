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

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from src.audio.timeline import format_timeline
from src.evaluation import placement
from src.evaluation.metrics import match_timestamps, temporal_precision_recall_f1
from src.evaluation.stats import bootstrap_mean_ci
from src.experiments.config import ConfigError, get
from src.experiments.data import ground_truth, load_context
from src.experiments.io import append_jsonl, latest_by_key, read_jsonl, update_status, write_json
from src.experiments.matrix import GenerationRun, SelectionRun
from src.experiments.selection_stage import selections_path
from src.generation.backends import GenRequest
from src.generation.outputs import final_rows, is_success, settled_ids
from src.generation.prompts import build_prompt, build_qa_prompt, enforce_word_limit, transcript_context, word_limit_for_gap
from src.utils.audio_features import available_gap, resolve_audio_path
from src.utils.cache_checks import read_manifest
from src.utils.media import frame_window


def moments_path(exp_dir: Path, run: SelectionRun) -> Path:
    return exp_dir / "moments" / f"{run.run_id}.jsonl"


def _transcript_items(record) -> list:
    return [SimpleNamespace(start=float(t["start"]), end=float(t["end"]), text=str(t["text"])) for t in record.transcript or []]


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
        manifest, reason = read_manifest(cache)
        if manifest is None:
            entry["status"] = reason
            continue
        context, available = load_context(record, cfg, run.spec.get("cue_set", "audio_cues.json"), need_visual=False)
        speech = [tuple(s) for s in context.audio_features["speech"]] if context.audio_features else None
        try:
            wav = resolve_audio_path(manifest, cache)
        except ValueError:
            wav = None
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
                "transcript": transcript_context(transcript, t), "word_limit": limit,
                "word_limit_source": "vad" if speech is not None else "fallback_no_vad",
                "audio_timeline": timeline, "timeline_status": "ok" if cues is not None else available.get("audio_cues"),
                "audio_clip": ({"path": str(wav), "start": max(0.0, t - span / 2 - 1.0), "end": t + span / 2 + 1.0}
                               if wav is not None else None),
            })
            entry["written"] += 1
    path = moments_path(exp_dir, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)
    write_json(path.with_suffix(".report.json"), {"selection_run_id": run.run_id, "videos": report})
    return path


def generations_path(exp_dir: Path, run: GenerationRun) -> Path:
    return exp_dir / "generations" / f"{run.run_id}.jsonl"


def check_exemplar_leakage(exemplars: dict, video_ids: set[str]) -> None:
    leaked = sorted({e["video_id"] for e in exemplars["exemplars"]} & video_ids)
    if leaked:
        raise ConfigError(f"P1 exemplars come from evaluated videos {leaked}: rebuild them from the exemplar pool")


def run_generation(
    cfg: dict, run: GenerationRun, exp_dir: Path, backend_factory: Callable[[dict], Any], records=None,
    limit: int | None = None,
) -> Path:
    """Generate AD for every moment of the run's selection; resumes from the output file."""
    moments = read_jsonl(moments_path(exp_dir, run.selection))
    if limit:
        moments = moments[:limit]
    out_path = generations_path(exp_dir, run)
    max_new = int(get(cfg, "generation.max_new_tokens", 64))
    done, _ = settled_ids(read_jsonl(out_path), max_new, 0.0)
    todo = [m for m in moments if m["id"] not in done]
    exemplars = None
    if run.prior == "P1":
        from src.generation.priors import load_exemplars

        exemplars = load_exemplars(get(cfg, "generation.exemplars_path", "configs/p1_exemplars.json"))
        check_exemplar_leakage(exemplars, {m["video_id"] for m in moments} | {r.video_id for r in records or []})
    policy = get(cfg, "generation.word_limit_policy", "truncate")
    if not todo:
        update_status(exp_dir, "generate", run.run_id, {"todo": 0, "done": len(done)})
        return out_path
    backend = backend_factory(run.model)
    batch_size = int(run.model.get("request_batch", 8))
    span = float(get(cfg, "selection.span_seconds", 2.0))
    n_ok = n_err = 0
    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        requests, base = [], {}
        for m in batch:
            prompt = build_prompt(run.prior, m["time"], len(m["frames"]), span, m["word_limit"], m.get("transcript", ""),
                                  audio_timeline=m.get("audio_timeline", "") if run.timeline else "", exemplars=exemplars)
            requests.append(GenRequest(m["id"], prompt, list(m["frames"]),
                                       m.get("audio_clip") if getattr(backend, "supports_audio", False) else None, max_new))
            base[m["id"]] = {"id": m["id"], "video_id": m["video_id"], "selector": m["selector"], "time": m["time"],
                             "word_limit": m["word_limit"], "frame_times": m["frame_times"], "prior": run.prior,
                             "model": run.model_label, "model_id": getattr(backend, "model_id", None),
                             "backend": getattr(backend, "name", None), "timeline": run.timeline,
                             "run_id": run.run_id, "temperature": 0.0, "max_new_tokens": max_new, "prompt": prompt}
        results = backend.generate(requests)
        rows = []
        for r in results:
            row = {**base[r.id], **r.to_dict()}
            row["model_output"] = row.pop("text", "")
            if r.error is None:
                capped = enforce_word_limit(row["model_output"], base[r.id]["word_limit"], policy)
                row.update(output=capped["text"], words=capped["words"], raw_words=capped["raw_words"],
                           over_budget=capped["over_budget"], truncated=capped["truncated"])
                n_ok += 1
            else:
                row["output"] = row["model_output"]
                n_err += 1
            rows.append(row)
        append_jsonl(out_path, rows)
    update_status(exp_dir, "generate", run.run_id, {"ok": n_ok, "errors": n_err, "previously_done": len(done)})
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


def evaluate_generation(cfg: dict, run: GenerationRun, records, exp_dir: Path, scorers: dict | None = None) -> dict:
    """Placement with real word counts, budget compliance and matched-pair text metrics."""
    by_id = {r.video_id: r for r in records}
    rows = final_rows(read_jsonl(generations_path(exp_dir, run)))
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
    for video_id, vrows in by_video.items():
        record = by_id.get(video_id)
        if record is None:
            continue
        ok = [r for r in vrows if is_success(r) and r.get("output")]
        context, _ = load_context(record, cfg, need_visual=False)
        truth, _ = ground_truth(record, context.duration, slack)
        times = [r["time"] for r in ok]
        primary = temporal_precision_recall_f1(times, truth, tol)
        speech = [tuple(s) for s in context.audio_features["speech"]] if context.audio_features else None
        placements = [(r["time"], len(r["output"].split())) for r in ok]
        entry = {"video_id": video_id, "n_moments": len(vrows), "n_ok": len(ok), "n_error": len(vrows) - len(ok),
                 "precision": primary["precision"], "recall": primary["recall"], "f1": primary["f1"],
                 "mean_words": (sum(n for _, n in placements) / len(placements)) if placements else None,
                 "over_budget_rate": (sum(bool(r.get("over_budget")) for r in ok) / len(ok)) if ok else None,
                 "overlap_seconds": (placement.speech_overlap(placements, speech, wps, speed)["overlap_seconds"]
                                     if speech is not None else None)}
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
    result = {
        "run_id": run.run_id, "selector": run.selection.name, "prior": run.prior, "model": run.model_label,
        "timeline": run.timeline, "oracle": run.selection.is_oracle, "n_videos": len(per_video),
        "macro_f1": bootstrap_mean_ci([v["f1"] for v in per_video], resamples, seed),
        "overlap_seconds": bootstrap_mean_ci([v["overlap_seconds"] for v in per_video], resamples, seed),
        "over_budget_rate": bootstrap_mean_ci([v["over_budget_rate"] for v in per_video], resamples, seed),
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


def run_qa(cfg: dict, run: GenerationRun, records, exp_dir: Path, backend_factory: Callable[[dict], Any]) -> Path:
    """Zero-shot multiple-choice QA over the run's selected keyframes."""
    from src.evaluation.qa import parse_choice

    selections = latest_by_key(read_jsonl(selections_path(exp_dir, run.selection)), "video_id")
    out_path = generations_path(exp_dir, run)
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
        cache = Path(get(cfg, "paths.cache_root")) / record.video_id
        manifest, _ = read_manifest(cache)
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
        row = {**base, "duration": sel["duration"], "frame_times": [f.timestamp for f in frames],
               "prompt": prompt, "reply": result.text, "error": result.error}
        row["prediction"] = None if result.error else parse_choice(result.text, item.options)
        row["status"] = "error" if result.error else "answered"
        append_jsonl(out_path, [row])
    return out_path


def evaluate_qa(cfg: dict, runs: Sequence[GenerationRun], exp_dir: Path) -> dict:
    from src.evaluation.qa import accuracy_by_length, paired_condition_comparison, score_items

    results: dict[str, Any] = {}
    final: dict[str, list[dict]] = {}
    for run in runs:
        rows = list(latest_by_key(read_jsonl(generations_path(exp_dir, run)), "question_id").values())
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
    write_json(exp_dir / "eval" / "qa_metrics.json", out)
    return out
