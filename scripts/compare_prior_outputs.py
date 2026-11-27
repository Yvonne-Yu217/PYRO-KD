#!/usr/bin/env python
"""Paired, moment-by-moment comparison of two prompt priors' raw generations (development use).

    python scripts/compare_prior_outputs.py --baseline p0.jsonl --variant p1.jsonl

Moments are paired by id. Both sides must have a complete decode; over-budget decodes count (they
are the behaviour being studied), truncated or failed generations are reported but not paired.
--leak-terms counts captions containing words from the P1 examples in BOTH files, so the variant's
rate can be read against the baseline's.
"""

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.outputs import final_rows, read_rows  # noqa: E402

DEFAULT_LEAK_TERMS = ("beth", "ronni", "peter", "web", "cotton", "hallway")


def complete_text(row: dict) -> str | None:
    """The full decoded caption, or None if there is none (failure or token-limit cut-off)."""
    text = row.get("output")
    if row.get("hit_token_limit") or not isinstance(text, str) or not text.strip():
        return None
    return text.strip()


def measure(text: str, word_limit: int | None, leak_terms) -> dict:
    words = text.split()
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return {
        "words": len(words),
        "sentences": len([s for s in re.split(r"[.!?]+(?:\s|$)", text) if s.strip()]),
        "over_budget": None if word_limit is None else len(words) > word_limit,
        "leaked": sorted(tokens & set(leak_terms)),
    }


def compare(baseline_rows, variant_rows, leak_terms=DEFAULT_LEAK_TERMS, examples: int = 10) -> dict:
    sides = [{r["id"]: r for r in final_rows(rows) if "id" in r} for rows in (baseline_rows, variant_rows)]
    shared = sorted(sides[0].keys() & sides[1].keys())
    pairs, unpaired = [], 0
    for moment in shared:
        a, b = sides[0][moment], sides[1][moment]
        ta, tb = complete_text(a), complete_text(b)
        if a.get("word_limit") != b.get("word_limit"):
            raise ValueError(f"{moment}: word limits differ, so these are not the same inputs")
        if ta is None or tb is None:
            unpaired += 1
            continue
        limit = a.get("word_limit")
        pairs.append({"id": moment, "word_limit": limit, "baseline": ta, "variant": tb,
                      "b": measure(ta, limit, leak_terms), "v": measure(tb, limit, leak_terms)})

    def summary(side):
        if not pairs:
            return None
        known = [p[side]["over_budget"] for p in pairs if p[side]["over_budget"] is not None]
        return {
            "mean_words": sum(p[side]["words"] for p in pairs) / len(pairs),
            "mean_sentences": sum(p[side]["sentences"] for p in pairs) / len(pairs),
            "over_budget": sum(known), "with_budget": len(known),
            "captions_with_leak_terms": sum(bool(p[side]["leaked"]) for p in pairs),
        }

    by_budget = []
    for low, high in ((1, 8), (9, 15), (16, 30), (31, None)):
        group = [p for p in pairs if p["word_limit"] is not None
                 and p["word_limit"] >= low and (high is None or p["word_limit"] <= high)]
        if group:
            by_budget.append({
                "word_limit": f"{low}-{high}" if high else f"{low}+", "n": len(group),
                "baseline_words": sum(p["b"]["words"] for p in group) / len(group),
                "variant_words": sum(p["v"]["words"] for p in group) / len(group),
                "baseline_over": sum(p["b"]["over_budget"] for p in group),
                "variant_over": sum(p["v"]["over_budget"] for p in group),
            })
    worst = sorted(pairs, key=lambda p: (-(p["v"]["words"] - p["b"]["words"]), p["id"]))[:examples]
    return {
        "shared_ids": len(shared), "paired": len(pairs), "unpaired_incomplete": unpaired,
        "baseline_only": len(sides[0]) - len(shared), "variant_only": len(sides[1]) - len(shared),
        "baseline": summary("b"), "variant": summary("v"), "by_budget": by_budget,
        "largest_length_increases": [
            {k: p[k] for k in ("id", "word_limit", "baseline", "variant")} for p in worst],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--variant", required=True, type=Path)
    parser.add_argument("--leak-terms", nargs="*", default=list(DEFAULT_LEAK_TERMS))
    parser.add_argument("--examples", type=int, default=10)
    args = parser.parse_args(argv)
    for path in (args.baseline, args.variant):
        if not path.is_file():
            parser.error(f"not a file: {path}")
    report = compare(read_rows(args.baseline), read_rows(args.variant),
                     [t.lower() for t in args.leak_terms], args.examples)
    print(json.dumps(report, indent=2))
    return 0 if report["paired"] else 1


if __name__ == "__main__":
    sys.exit(main())
