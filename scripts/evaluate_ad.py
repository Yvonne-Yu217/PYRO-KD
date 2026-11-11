#!/usr/bin/env python
"""Evaluate MTG-AD generation JSONL with temporal and matched-text metrics.

    python scripts/evaluate_ad.py --input outputs/run.jsonl --split validation \
        --data-dir data/mtg-ad --output outputs/run.evaluation.json

One experiment (selector/model/prior) is evaluated per input file. By default,
only video IDs present in the JSONL are evaluated, including error-only videos.
Use --all-videos to also count missing split videos as empty predictions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import fmean, median
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import mtg_ad  # noqa: E402
from src.evaluation.bertscore import BERTScoreScorer, GeneratedADItem, compute_bertscore  # noqa: E402
from src.evaluation.cosine_similarity import (  # noqa: E402
    DEFAULT_MODEL_NAME, SentenceEncoder, compute_cosine_similarity,
)
from src.evaluation.metrics import match_timestamps  # noqa: E402


_RUN_FIELDS = ("selector", "model", "model_id", "prior", "quant")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def read_generated_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Validate generation rows; failed rows need video_id/time but no output."""
    rows = []
    successful_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_reject_constant)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                if not isinstance(row.get("video_id"), str) or not row["video_id"].strip():
                    raise ValueError("video_id must be a nonempty string")
                time = row.get("time")
                if (
                    isinstance(time, bool) or not isinstance(time, (int, float))
                    or time < 0 or not math.isfinite(float(time))
                ):
                    raise ValueError("time must be finite nonnegative numeric seconds")
                if "error" not in row:
                    if not isinstance(row.get("output"), str) or not row["output"].strip():
                        raise ValueError("successful output must be a nonempty string")
                    if "id" in row:
                        if not isinstance(row["id"], str) or not row["id"].strip():
                            raise ValueError("id, when present, must be a nonempty string")
                        if row["id"] in successful_ids:
                            raise ValueError(f"duplicate successful id: {row['id']!r}")
                        successful_ids.add(row["id"])
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            rows.append(row)
    return rows


def _temporal_summary(
    tp: int, fp: int, fn: int, errors: Sequence[float], tolerance_seconds: float,
) -> dict[str, Any]:
    """Summarize existing match counts without running another matcher."""
    empty = tp + fp + fn == 0
    return {
        "tolerance_seconds": tolerance_seconds,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": None if empty else tp / (tp + fp) if tp + fp else 0.0,
        "recall": None if empty else tp / (tp + fn) if tp + fn else 0.0,
        "f1": None if empty else 2 * tp / (2 * tp + fp + fn),
        "mean_temporal_error_seconds": fmean(errors) if errors else None,
    }


def _score(call: Callable[[], dict[str, Any]], unavailable: dict[str, Any]) -> dict[str, Any]:
    """Keep semantic failures explicit while retaining valid temporal results."""
    try:
        result = call()
    except Exception as exc:
        return {**unavailable, "num_scored_pairs": 0, "pairs": [],
                "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return {**result, "status": "ok" if result["num_scored_pairs"] else "no_matches"}


def _load_bert_scorer(lang: str, model_type: str | None) -> tuple[BERTScoreScorer, str]:
    """Load one reusable official BERTScore model for an evaluation run."""
    from bert_score import BERTScorer

    model = BERTScorer(
        lang=lang, model_type=model_type, idf=False,
        rescale_with_baseline=False, use_fast_tokenizer=False,
    )

    def score(candidates: list[str], references: list[str], **kwargs: Any) -> Any:
        return model.score(
            candidates, references, return_hash=bool(kwargs.get("return_hash", False)),
        )

    return score, model.model_type


def _load_sentence_encoder(model_name: str) -> SentenceEncoder:
    """Load one reusable sentence encoder for an evaluation run."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def evaluate_ad(
    generated_path: str | Path,
    data_dir: str | Path,
    split: str,
    *,
    tolerance_seconds: float = 2.0,
    lang: str = "en",
    bert_model: str | None = None,
    cosine_model: str = DEFAULT_MODEL_NAME,
    bert_scorer: BERTScoreScorer | None = None,
    encoder: SentenceEncoder | None = None,
    all_videos: bool = False,
) -> dict[str, Any]:
    """Load a split and evaluate one generation run, with injectable scorers.

    Match once per video and pass the identical TemporalMatch objects to both
    semantic functions. MTGVideo.ad is used directly. Error rows are skipped
    as predictions, but retained in reports. Unknown video IDs, mixed run
    metadata, and duplicate successful IDs fail before any model is loaded.
    Dataset semantic means/median pool scored pairs, excluding unavailable
    scores. The default scope is input video IDs; all_videos includes the split.
    Rubric evaluation is omitted and no provider API is called.
    """
    if (
        isinstance(tolerance_seconds, bool) or not isinstance(tolerance_seconds, (int, float))
        or tolerance_seconds < 0 or not math.isfinite(float(tolerance_seconds))
    ):
        raise ValueError("tolerance_seconds must be finite nonnegative numeric seconds")
    tolerance_seconds = float(tolerance_seconds)
    rows = read_generated_jsonl(generated_path)
    videos = mtg_ad.load_split(data_dir, split)
    by_id = {video.video_id: video for video in videos}
    if len(by_id) != len(videos):
        raise ValueError("MTG-AD split contains duplicate video IDs")
    input_ids = {row["video_id"] for row in rows}
    unknown = sorted(input_ids - by_id.keys())
    if unknown:
        raise ValueError(f"Generated video IDs absent from MTG-AD {split!r} split: {unknown}")

    metadata: dict[str, Any] = {}
    successful: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for field in _RUN_FIELDS:
            if row.get(field) is not None:
                if field in metadata and metadata[field] != row[field]:
                    raise ValueError(f"Mixed {field} values; evaluate each experiment separately")
                metadata[field] = row[field]
        target = failed if "error" in row else successful
        target[row["video_id"]].append(row)

    results = []
    temporal_errors: list[float] = []
    semantic_errors = []
    active_bert_scorer = bert_scorer
    active_bert_model = bert_model
    active_encoder = encoder

    def score_bert(
        generated: Sequence[GeneratedADItem], ground_truth: Sequence[Any], matches: Sequence[Any],
    ) -> dict[str, Any]:
        nonlocal active_bert_scorer, active_bert_model
        # Preserve the semantic functions' no-match fast path: an empty run must
        # not import dependencies or load models.
        if matches and active_bert_scorer is None:
            active_bert_scorer, active_bert_model = _load_bert_scorer(lang, bert_model)
        return compute_bertscore(
            generated, ground_truth, matches, lang=lang, model_type=active_bert_model,
            scorer=active_bert_scorer,
        )

    def score_cosine(
        generated: Sequence[GeneratedADItem], ground_truth: Sequence[Any], matches: Sequence[Any],
    ) -> dict[str, Any]:
        nonlocal active_encoder
        if matches and active_encoder is None:
            active_encoder = _load_sentence_encoder(cosine_model)
        return compute_cosine_similarity(
            generated, ground_truth, matches, model_name=cosine_model, encoder=active_encoder,
        )

    for video_id in sorted(by_id if all_videos else input_ids):
        video = by_id[video_id]
        generated_rows = successful[video_id]
        generated = [GeneratedADItem(float(row["time"]), row["output"]) for row in generated_rows]
        matches = match_timestamps(
            [item.timestamp for item in generated], [item.timestamp for item in video.ad],
            tolerance_seconds,
        )
        errors = [match.error_seconds for match in matches]
        temporal_errors.extend(errors)
        temporal = _temporal_summary(
            len(matches), len(generated) - len(matches), len(video.ad) - len(matches),
            errors, tolerance_seconds,
        )
        temporal["matches"] = [asdict(match) for match in matches]
        bert = _score(
            lambda: score_bert(generated, video.ad, matches),
            {"precision_mean": None, "recall_mean": None, "f1_mean": None,
             "lang": lang, "model_type": bert_model, "configuration_hash": None},
        )
        cosine = _score(
            lambda: score_cosine(generated, video.ad, matches),
            {"cosine_similarity_mean": None, "cosine_similarity_median": None,
             "model_name": cosine_model},
        )
        status = (
            "no_successful_generations" if not generated else
            "no_ground_truth" if not video.ad else "no_temporal_matches" if not matches else "ok"
        )
        for name, metric in (("bertscore", bert), ("cosine", cosine)):
            if metric["status"] == "error":
                status = "semantic_error"
                semantic_errors.append({"video_id": video_id, "metric": name, "error": metric["error"]})
        results.append({
            "video_id": video_id, "category": video.category, "status": status,
            "num_generated": len(generated), "num_ground_truth": len(video.ad),
            "num_generation_errors": len(failed[video_id]), "generation_errors": failed[video_id],
            "generated_items": [
                {**asdict(item), "metadata": {k: v for k, v in row.items()
                                             if k not in {"video_id", "time", "output"}}}
                for item, row in zip(generated, generated_rows, strict=True)
            ],
            "temporal": temporal, "bertscore": bert, "cosine": cosine,
        })

    totals = {key: sum(video["temporal"][key] for video in results) for key in ("tp", "fp", "fn")}
    bert_pairs = [pair for video in results for pair in video["bertscore"]["pairs"]]
    cosine_values = [pair["cosine_similarity"] for video in results for pair in video["cosine"]["pairs"]]
    generation_errors = [row for row in rows if "error" in row]
    return {
        "status": "partial" if generation_errors or semantic_errors else "ok" if rows else "empty_input",
        "input": str(generated_path), "data_dir": str(data_dir), "split": split,
        "scope": "all_split_videos" if all_videos else "input_video_ids",
        "metadata": metadata,
        "evaluation_config": {
            "tolerance_seconds": tolerance_seconds,
            "bertscore": {"lang": lang, "model_type": bert_model},
            "cosine": {"model_name": cosine_model},
        },
        "coverage": {"num_split_videos": len(videos), "num_evaluated_videos": len(results),
                     "split_video_ids_without_input": sorted(by_id.keys() - input_ids)},
        "num_input_rows": len(rows), "num_successful_rows": len(rows) - len(generation_errors),
        "num_generation_errors": len(generation_errors), "generation_errors": generation_errors,
        "semantic_errors": semantic_errors,
        "videos": results,
        "aggregate": {
            "total_matched_pairs": totals["tp"],
            "temporal": _temporal_summary(**totals, errors=temporal_errors, tolerance_seconds=tolerance_seconds),
            "bertscore": {
                "num_scored_pairs": len(bert_pairs),
                **{f"{key}_mean": fmean(pair[key] for pair in bert_pairs) if bert_pairs else None
                   for key in ("precision", "recall", "f1")},
            },
            "cosine": {
                "num_scored_pairs": len(cosine_values),
                "cosine_similarity_mean": fmean(cosine_values) if cosine_values else None,
                "cosine_similarity_median": median(cosine_values) if cosine_values else None,
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Save a JSON report; return 1 for reported generation/semantic failures."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/mtg-ad"))
    parser.add_argument("--split", choices=mtg_ad.SPLITS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=2.0, help="inclusive start-time tolerance in seconds")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--bert-model", default=None, help="default: BERTScore's language-specific model")
    parser.add_argument("--cosine-model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--all-videos", action="store_true", help="include split videos absent from the input")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input")
    try:
        result = evaluate_ad(
            args.input, args.data_dir, args.split, tolerance_seconds=args.tolerance,
            lang=args.lang, bert_model=args.bert_model, cosine_model=args.cosine_model,
            all_videos=args.all_videos,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(f"Evaluated {len(result['videos'])} videos; {result['aggregate']['total_matched_pairs']} matches; "
          f"{result['num_generation_errors']} generation errors; {len(result['semantic_errors'])} semantic errors. "
          f"Report: {args.output}")
    return 1 if result["status"] == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
