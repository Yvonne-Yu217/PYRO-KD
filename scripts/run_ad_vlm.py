#!/usr/bin/env python
"""Generate one audio description per selected moment with a frozen open VLM.

Input: moments JSONL from scripts/build_moments.py. Output JSONL: one record
per moment with the description, raw decode, latency, and token counts.
Finished ids are skipped on restart, so a preempted/requeued SLURM job
resumes where it stopped. <output>.meta.json records GPU, quantization, peak
memory, and timing for the cost section of the report.

    python scripts/run_ad_vlm.py --model qwen3-vl-8b --input data/moments/mtg-ad_U_dev.jsonl --limit 4
    python scripts/run_ad_vlm.py --input data/moments/mtg-ad_U_dev.jsonl --dry-run   # no GPU
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.hf_vlm import MODELS, generate_batch, load_model, resolve_model  # noqa: E402
from src.generation.outputs import merge_meta, read_rows, settled_ids, status_counts  # noqa: E402
from src.generation.prompts import PROMPTS, build_prompt, word_limit_for_gap  # noqa: E402
from src.generation.exemplars import check_resume_prior, load_exemplars  # noqa: E402
from src.generation.audio_timeline import check_audio_resume, render_timeline, timeline_hash  # noqa: E402
from src.generation.jsonl_recovery import read_resume_rows,apply_tail_recovery
from src.generation.provenance import capture_provenance, check_resume_fingerprint  # noqa: E402
from src.generation.native_audio import load_native_audio


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as handle:
        for i, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not {"id", "frames", "time"} <= row.keys():
                raise ValueError(f"{path}:{i} needs 'id', 'frames' and 'time'")
            rows.append(row)
    return rows


def done_ids(path: Path, max_new_tokens: int | None = None, temperature: float = 0.0) -> set[str]:
    """Moment ids that need no more work.

    Successes always count. With max_new_tokens given, so do deterministic token-limit
    failures (temperature 0 and at least this many tokens): retrying them would just
    reproduce the cut-off and append another error. Raise --max-new-tokens to retry
    them. All other errors, such as out-of-memory, are retried. See generation.outputs.
    """
    done, _ = settled_ids(read_rows(path), max_new_tokens, temperature)
    return done


def finalize_generation(
    text: str, raw: str, new_tokens: int, batch_size: int, latency: float, max_new_tokens: int,
    word_limit: int | None = None,
) -> dict:
    """Result record for one generation.

    A generation that reached max_new_tokens was cut off. For Muse that is usually
    before its answer, so `output` still holds the scratchpad. It is recorded as an
    error: evaluate_ad does not score it, and the cut-off text stays in
    output/raw_output for debugging. The row also records max_new_tokens (temperature
    is added by the caller) so a requeue can tell whether retrying could change the
    result: done_ids treats a temperature-0 failure at the same or a larger
    max_new_tokens as settled and does not retry it. Raising --max-new-tokens, or
    sampling with --temperature, makes it eligible again (see generation.outputs).
    """
    hit_limit = new_tokens >= max_new_tokens
    result = {
        "output": text, "raw_output": raw, "new_tokens": new_tokens, "batch_size": batch_size,
        "hit_token_limit": hit_limit, "max_new_tokens": max_new_tokens,
        "batch_latency_s": round(latency, 3),
    }
    if hit_limit:
        result["error"] = (
            f"hit_token_limit: generation reached max_new_tokens={max_new_tokens} "
            "before finishing (output is cut off and was not scored)"
        )
    if word_limit is not None:
        result['word_count'] = len(text.split())
        result['word_limit_exceeded'] = result['word_count'] > word_limit
        if result['word_limit_exceeded']:
            result['error'] = result.get('error', '') + (
                f" word_limit_exceeded: {result['word_count']} words > {word_limit}; full decode retained")
    return result


def resize_max_side(image, max_side: int | None):
    from PIL import Image

    if not max_side or max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.BICUBIC)


def prepare(row: dict, args) -> tuple[list, str, int]:
    word_limit = row.get("word_limit")
    if word_limit is None:
        word_limit = word_limit_for_gap(None)
    if word_limit <= 0:
        raise ValueError("moment is not safely speakable: zero-word budget")
    from PIL import Image

    images = [resize_max_side(Image.open(p).convert("RGB"), args.max_side) for p in row["frames"]]
    audio_context = getattr(args, "audio_context", "legacy")
    transcript = row.get("transcript", "") if audio_context == "legacy" else ""
    timeline = render_timeline(row.get("audio_timeline")) if audio_context == "timeline" else None
    prompt = build_prompt(
        args.prior, row.get("frame_center_time", row["time"]), len(images), args.span, word_limit, transcript,
        exemplars=getattr(args, "exemplar_texts", ()),
        audio_timeline_text=timeline,
    )
    if audio_context == 'native':
        audio = row['native_audio']
        prompt += (f"\nThe supplied audio covers {audio['start_seconds']:.6f} to "
                   f"{audio['end_seconds']:.6f} seconds of the clip. Use it as context "
                   "for the visible scene; do not transcribe dialogue.")
    return images, prompt, word_limit


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3-vl-8b",
                   help=f"short name ({', '.join(MODELS)}) or any HF repo id")
    p.add_argument("--input", default="data/moments/mtg-ad_U_dev.jsonl")
    p.add_argument("--output", default=None,
                   help="default: outputs/<model>/<input stem>_<prior>.jsonl")
    p.add_argument("--prior", choices=sorted(PROMPTS), default="P0")
    p.add_argument("--exemplars", help="frozen training-only bundle required for P1")
    p.add_argument('--backend', choices=('vlm', 'omni'), default='vlm')
    p.add_argument('--word-budget-retries', type=int, choices=(0, 1), default=0,
                   help='optional one model repair of an over-budget draft; all attempts retained')
    p.add_argument("--audio-context", choices=("legacy", "none", "timeline", "native"), default="legacy",
                   help="legacy transcript; no audio text; or cached speech/event timeline")
    p.add_argument("--span", type=float, default=2.0, help="seconds covered by each frame window")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-side", type=int, default=768,
                   help="resize so each frame's longer side is at most this (0 = no resize)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--reasoning-strength", default=None, help="Muse Glimmer only, e.g. low")
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    p.add_argument("--attn-impl", default=None, help="e.g. sdpa or flash_attention_2")
    p.add_argument("--limit", type=int, default=None, help="only the first N moments")
    p.add_argument("--dry-run", action="store_true", help="print prompts without loading a model")
    return p.parse_args()


def main():
    args = parse_args()
    if args.audio_context == 'native' and args.backend != 'omni':
        raise ValueError('native waveform context requires --backend omni')
    if args.backend == 'omni' and (args.batch_size != 1 or args.quant != 'none' or args.reasoning_strength):
        raise ValueError('Omni requires --batch-size 1, --quant none, and no reasoning-strength')
    if (args.prior == "P1") != bool(args.exemplars):
        raise ValueError("--exemplars is required for P1 and forbidden for P0")
    args.exemplar_texts, args.exemplar_sha256 = (
        load_exemplars(args.exemplars) if args.exemplars else ([], None)
    )
    model_id = resolve_model(args.model)
    short = args.model.replace("/", "__") + ("" if args.quant == "none" else f"-{args.quant}")
    out_path = Path(args.output or f"outputs/{short}/{Path(args.input).stem}_{args.prior}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(Path(args.input))
    if args.audio_context == 'native':
        for row in rows:
            if row.get('transcript') or row.get('audio_timeline'):
                raise ValueError('native input must not include transcript/timeline context')
            if row['native_audio']['frame_center_time'] != row.get('frame_center_time', row['time']):
                raise ValueError('native audio and visual center differ')
            load_native_audio(row['native_audio'])
    provenance = capture_provenance(Path(__file__).resolve().parents[1], Path(args.input), rows, {
        "model_id": model_id, "quant": args.quant, "attn_impl": args.attn_impl,
        "prior": args.prior, "exemplar_sha256": args.exemplar_sha256,
        "audio_context": args.audio_context, "span": args.span, "max_side": args.max_side,
        "temperature": args.temperature, "top_p": args.top_p,
        "reasoning_strength": args.reasoning_strength,
        "backend": args.backend,
        "word_budget_retries": args.word_budget_retries,
    })
    if args.limit:
        rows = rows[: args.limit]
    rows_on_disk, recovery = read_resume_rows(out_path)
    check_resume_fingerprint(rows_on_disk, provenance["run_fingerprint"])
    check_resume_prior(rows_on_disk, args.prior, args.exemplar_sha256)
    check_audio_resume(rows_on_disk, rows, args.audio_context)
    if args.audio_context == "timeline":
        for row in rows:
            render_timeline(row.get("audio_timeline"))
    finished, repeats = settled_ids(rows_on_disk, args.max_new_tokens, args.temperature)
    finished &= {r["id"] for r in rows}
    todo = [r for r in rows if r["id"] not in finished]
    print(f"[run_ad] {len(rows)} moments, {len(finished)} already done, {len(todo)} to run", flush=True)
    if repeats & finished:
        print(
            f"[run_ad] {len(repeats & finished)} of the finished moments are failures that would repeat "
            f"at max_new_tokens={args.max_new_tokens}, temperature={args.temperature} (over the word "
            "budget, or cut off at the token limit; raising --max-new-tokens retries only the latter)",
            flush=True)

    if args.dry_run:
        for row in todo[:5]:
            images, prompt, _ = prepare(row, args)
            sizes = sorted({img.size for img in images})
            print(f"--- {row['id']} ({len(images)} frames, sizes={sizes})\n{prompt}\n")
        print("[run_ad] dry run: no model loaded")
        return

    if recovery is not None:
        if not rows_on_disk:
            sidecar = json.loads(out_path.with_suffix(out_path.suffix + '.provenance.json').read_text())
            if sidecar['run_fingerprint'] != provenance['run_fingerprint']:
                raise ValueError('interrupted output provenance mismatch')
        print(json.dumps({'output_recovery': apply_tail_recovery(out_path, recovery)}), flush=True)
    if not todo:
        print("[run_ad] no pending moments; no model loaded")
        return

    # Save before model loading so an interrupted attempt still has source evidence.
    provenance_path = out_path.with_suffix(out_path.suffix + ".provenance.json")
    temporary = provenance_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(provenance, indent=2) + "\n")
    temporary.replace(provenance_path)

    import torch
    import transformers

    t0 = time.time()
    if args.backend == 'omni':
        from src.generation.omni import load_model as load_omni
        processor, model, cls_name = load_omni(model_id, args.attn_impl)
    else:
        processor, model, cls_name = load_model(model_id, args.attn_impl, args.quant)
    load_s = time.time() - t0
    print(f"[run_ad] loaded {model_id} via {cls_name} in {load_s:.0f}s", flush=True)
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)

    gen_s, n_ok, n_err, n_truncated = 0.0, 0, 0, 0
    with open(out_path, "a") as fout:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start: start + args.batch_size]
            prepared, records = [], []
            for row in batch:
                base = {"id": row["id"], "video_id": row.get("video_id"), "temperature": args.temperature,
                        "selector": row.get("selector"), "time": row["time"],
                        "frame_times": row.get("frame_times"), "model": args.model,
                        "model_id": model_id, "quant": args.quant, "prior": args.prior,
                        "exemplar_sha256": args.exemplar_sha256,
                        "gap_seconds": row.get("gap_seconds"),
                        "word_budget_rule": row.get("word_budget_rule"),
                        "audio_context": args.audio_context,
                        "backend": args.backend,
                        "run_fingerprint": provenance["run_fingerprint"],
                        "audio_timeline_sha256": timeline_hash(row["audio_timeline"]) if args.audio_context == "timeline" else None}
                for key in ("event_time", "placement_lag_seconds", "frame_center_policy",
                            "frame_center_time", "selection_condition", "selection_rank",
                            "selection_score", "selection_report_sha256", "protocol_status",
                            "k_requested", "k_shortfall", "gap_start", "gap_end"):
                    if key in row:
                        base[key] = row[key]
                if args.audio_context == 'native':
                    base['native_audio'] = row['native_audio']
                try:
                    images, prompt, word_limit = prepare(row, args)
                    prepared.append((images, prompt))
                    records.append({**base, "prompt": prompt, "word_limit": word_limit})
                except Exception as e:  # unreadable frame etc.
                    fout.write(json.dumps({**base, "error": f"input: {e!r}"}) + "\n")
                    n_err += 1
            if not prepared:
                continue

            # Run the batch; on failure (e.g. OOM) fall back to one item at a time.
            chunks = [list(range(len(prepared)))]
            results = {}
            while chunks:
                idx = chunks.pop(0)
                try:
                    t = time.time()
                    if args.backend == 'omni':
                        from src.generation.omni import generate_one
                        i = idx[0]
                        waveform = (load_native_audio(records[i]['native_audio'])
                                    if args.audio_context == 'native' else None)
                        texts, raws, n_new = generate_one(
                            processor, model, prepared[i][0], prepared[i][1], waveform,
                            args.max_new_tokens, args.temperature, args.top_p)
                    else:
                        texts, raws, n_new = generate_batch(
                            processor, model,
                            [prepared[i][0] for i in idx], [prepared[i][1] for i in idx],
                            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                            top_p=args.top_p, reasoning_strength=args.reasoning_strength,
                        )
                    dt = time.time() - t
                    gen_s += dt
                    for j, i in enumerate(idx):
                        results[i] = finalize_generation(
                            texts[j], raws[j], n_new[j], len(idx), dt, args.max_new_tokens,
                            records[i]['word_limit'])
                except Exception as e:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if len(idx) > 1:
                        chunks = [[i] for i in idx] + chunks
                    else:
                        results[idx[0]] = {"error": f"generate: {e!r}"}

            for i, rec in enumerate(records):
                res = results[i]
                if args.word_budget_retries:
                    from src.generation.budget_repair import repair_once
                    def repair_generate(repair_prompt):
                        start_time = time.time()
                        if args.backend == 'omni':
                            from src.generation.omni import generate_one
                            wave = load_native_audio(rec['native_audio']) if args.audio_context == 'native' else None
                            text, raw, tokens = generate_one(
                                processor, model, prepared[i][0], repair_prompt, wave,
                                args.max_new_tokens, args.temperature, args.top_p)
                        else:
                            text, raw, tokens = generate_batch(
                                processor, model, [prepared[i][0]], [repair_prompt],
                                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                top_p=args.top_p, reasoning_strength=args.reasoning_strength)
                        return finalize_generation(text[0], raw[0], tokens[0], 1,
                            time.time()-start_time, args.max_new_tokens, rec['word_limit'])
                    res = repair_once(res, rec['prompt'], rec['word_limit'], repair_generate)
                    gen_s += res['repair_latency_s']
                n_ok += "error" not in res
                n_err += "error" in res
                n_truncated += bool(res.get("hit_token_limit"))
                fout.write(json.dumps({**rec, **res}) + "\n")
            fout.flush()
            print(f"[run_ad] {start + len(batch)}/{len(todo)} done "
                  f"({n_err} errors, {gen_s:.0f}s generating)", flush=True)

    attempt = {
        "provenance": provenance,
        "model_id": model_id, "model_class": cls_name, "quant": args.quant,
        "n_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0, "args": vars(args),
        "n_ok": n_ok, "n_error": n_err, "n_hit_token_limit": n_truncated,
        "load_s": round(load_s, 1),
        "generate_s": round(gen_s, 1),
        # Summed over all visible GPUs: device_map="auto" splits big models across them.
        "peak_gpu_mem_gb": (
            round(sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9, 2)
            if torch.cuda.is_available() else None
        ),
        "peak_gpu_mem_gb_per_device": (
            [round(torch.cuda.max_memory_allocated(i) / 1e9, 2) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available() else None
        ),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    meta_path = out_path.with_suffix(".meta.json")
    try:
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        previous = None
    # Cumulative across requeued attempts; counts describe the output file as it stands now.
    meta = merge_meta(previous, attempt, status_counts(read_rows(out_path)))
    temporary_meta = meta_path.with_suffix(".json.tmp")
    temporary_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    temporary_meta.replace(meta_path)
    print(f"[run_ad] wrote {out_path} and {meta_path}", flush=True)
    sys.exit(1 if n_ok == 0 and n_err else 0)


if __name__ == "__main__":
    main()
