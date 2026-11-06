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
from src.generation.prompts import PROMPTS, build_prompt, word_limit_for_gap  # noqa: E402


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


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with open(path) as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial line from a killed job
            if not row.get("error"):
                ids.add(row["id"])
    return ids


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
    prompt = build_prompt(
        args.prior, row["time"], len(images), args.span, word_limit, row.get("transcript", ""),
    )
    return images, prompt, word_limit


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen3-vl-8b",
                   help=f"short name ({', '.join(MODELS)}) or any HF repo id")
    p.add_argument("--input", default="data/moments/mtg-ad_U_dev.jsonl")
    p.add_argument("--output", default=None,
                   help="default: outputs/<model>/<input stem>_<prior>.jsonl")
    p.add_argument("--prior", choices=sorted(PROMPTS), default="P0")
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
    model_id = resolve_model(args.model)
    short = args.model.replace("/", "__") + ("" if args.quant == "none" else f"-{args.quant}")
    out_path = Path(args.output or f"outputs/{short}/{Path(args.input).stem}_{args.prior}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(Path(args.input))
    if args.limit:
        rows = rows[: args.limit]
    finished = done_ids(out_path)
    todo = [r for r in rows if r["id"] not in finished]
    print(f"[run_ad] {len(rows)} moments, {len(finished)} already done, {len(todo)} to run", flush=True)

    if args.dry_run:
        for row in todo[:5]:
            images, prompt, _ = prepare(row, args)
            sizes = sorted({img.size for img in images})
            print(f"--- {row['id']} ({len(images)} frames, sizes={sizes})\n{prompt}\n")
        print("[run_ad] dry run: no model loaded")
        return

    import torch
    import transformers

    t0 = time.time()
    processor, model, cls_name = load_model(model_id, args.attn_impl, args.quant)
    load_s = time.time() - t0
    print(f"[run_ad] loaded {model_id} via {cls_name} in {load_s:.0f}s", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    gen_s, n_ok, n_err = 0.0, 0, 0
    with open(out_path, "a") as fout:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start: start + args.batch_size]
            prepared, records = [], []
            for row in batch:
                base = {"id": row["id"], "video_id": row.get("video_id"),
                        "selector": row.get("selector"), "time": row["time"],
                        "frame_times": row.get("frame_times"), "model": args.model,
                        "model_id": model_id, "quant": args.quant, "prior": args.prior}
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
                    texts, raws, n_new = generate_batch(
                        processor, model,
                        [prepared[i][0] for i in idx], [prepared[i][1] for i in idx],
                        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                        top_p=args.top_p, reasoning_strength=args.reasoning_strength,
                    )
                    dt = time.time() - t
                    gen_s += dt
                    for j, i in enumerate(idx):
                        results[i] = {"output": texts[j], "raw_output": raws[j],
                                      "new_tokens": n_new[j], "batch_size": len(idx),
                                      "batch_latency_s": round(dt, 3)}
                except Exception as e:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if len(idx) > 1:
                        chunks = [[i] for i in idx] + chunks
                    else:
                        results[idx[0]] = {"error": f"generate: {e!r}"}

            for i, rec in enumerate(records):
                res = results[i]
                n_ok += "error" not in res
                n_err += "error" in res
                fout.write(json.dumps({**rec, **res}) + "\n")
            fout.flush()
            print(f"[run_ad] {start + len(batch)}/{len(todo)} done "
                  f"({n_err} errors, {gen_s:.0f}s generating)", flush=True)

    meta = {
        "model_id": model_id, "model_class": cls_name, "quant": args.quant,
        "n_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0, "args": vars(args),
        "n_ok": n_ok, "n_error": n_err, "load_s": round(load_s, 1),
        "generate_s": round(gen_s, 1),
        "s_per_item": round(gen_s / n_ok, 3) if n_ok else None,
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[run_ad] wrote {out_path} and {meta_path}", flush=True)
    sys.exit(1 if n_ok == 0 and n_err else 0)


if __name__ == "__main__":
    main()
