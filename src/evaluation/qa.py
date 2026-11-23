"""Multiple-choice QA scoring (proposal §7D): answer parsing, accuracy, accuracy
by video length, and paired timeline on/off comparisons.

Answers are parsed strictly: a bare option letter, "(C)", "C.", "Answer: C",
or an exact option text. Anything else is "unparsed" and counted wrong (and
reported), never guessed.
"""

from __future__ import annotations

import re
from typing import Sequence

from src.evaluation.significance import mcnemar_exact
from src.evaluation.stats import bootstrap_mean_ci, paired_bootstrap_diff

LETTERS = "ABCDEFGHIJ"
_LETTER_PATTERNS = (
    re.compile(r"^\s*\(?([A-J])\)?\s*[\.\):]?\s*$"),
    re.compile(r"(?i)\b(?:answer|option)\s*(?:is|:)?\s*\(?([A-J])\)?(?![A-Za-z])"),
    re.compile(r"^\s*\(?([A-J])[\.\)]\s"),
)


def parse_choice(text: str, options: Sequence[str]) -> int | None:
    """Index of the chosen option, or None if the reply cannot be parsed unambiguously."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    n = len(options)
    for pattern in _LETTER_PATTERNS:
        match = pattern.search(stripped)
        if match:
            index = LETTERS.index(match.group(1).upper())
            return index if index < n else None
    normalized = stripped.lower().rstrip(".")
    matches = [i for i, o in enumerate(options) if o.strip().lower().rstrip(".") == normalized]
    return matches[0] if len(matches) == 1 else None


def score_items(rows: Sequence[dict]) -> dict:
    """rows: {"question_id", "prediction" (index or None), "answer" (index), optional "duration"}."""
    correct = [r["prediction"] is not None and r["prediction"] == r["answer"] for r in rows]
    unparsed = sum(r["prediction"] is None for r in rows)
    return {
        "n": len(rows),
        "accuracy": bootstrap_mean_ci([float(c) for c in correct]) if rows else None,
        "n_unparsed": unparsed,
    }


def accuracy_by_length(rows: Sequence[dict], edges: Sequence[float] = (180, 600, 1800)) -> list[dict]:
    """Accuracy per duration bin: [0, e1), [e1, e2), ..., [e_last, inf)."""
    bounds = [0.0, *edges, float("inf")]
    out = []
    for lo, hi in zip(bounds, bounds[1:]):
        chosen = [r for r in rows if r.get("duration") is not None and lo <= r["duration"] < hi]
        acc = [float(r["prediction"] == r["answer"]) for r in chosen]
        out.append({"min_seconds": lo, "max_seconds": hi, "n": len(chosen),
                    "accuracy": sum(acc) / len(acc) if acc else None})
    return out


def paired_condition_comparison(a: Sequence[dict], b: Sequence[dict]) -> dict:
    """Same questions under two conditions (e.g. timeline on vs off): accuracy
    difference with a paired bootstrap CI and an exact McNemar test."""
    by_a = {r["question_id"]: r for r in a}
    by_b = {r["question_id"]: r for r in b}
    shared = sorted(by_a.keys() & by_b.keys())
    ca = [by_a[q]["prediction"] == by_a[q]["answer"] for q in shared]
    cb = [by_b[q]["prediction"] == by_b[q]["answer"] for q in shared]
    return {
        "n_shared": len(shared),
        "accuracy_diff": paired_bootstrap_diff([float(x) for x in ca], [float(y) for y in cb]),
        "mcnemar": mcnemar_exact(ca, cb),
    }


from collections import Counter
from src.generation.qa import parse_answer
from src.generation.outputs import split_rows,is_failure


def score_qa(predictions, expected, num_options):
    """expected maps frozen question IDs to answer indices; generation never calls this.

    Retry errors may be superseded by a success. Duplicate successes, unknown
    IDs, mixed run fingerprints and disagreement between text/index are errors.
    Missing, invalid and failed predictions are incorrect in primary accuracy.
    """
    if num_options not in (4,5) or not expected:
        raise ValueError('nonempty expected questions with four/five options required')
    if any(not isinstance(ident,str) or not ident or isinstance(label,bool) or
           not isinstance(label,int) or not 0<=label<num_options for ident,label in expected.items()):
        raise ValueError('invalid evaluation labels')
    predictions=list(predictions)
    if any(row.get('id') not in expected for row in predictions):
        raise ValueError('prediction outside the declared evaluation split')
    fingerprints={row.get('run_fingerprint') for row in predictions}
    if len(fingerprints)>1: raise ValueError('mixed generation runs')
    if len({row.get('selector') for row in predictions})>1:
        raise ValueError('mixed selectors require separate accuracy reports')
    final,superseded=split_rows(predictions)
    if len({row['id'] for row in final})!=len(final): raise ValueError('duplicate successful predictions')
    by_id={row['id']:row for row in final}
    rows=[]
    for ident,label in expected.items():
        prediction=by_id.get(ident);choice=None
        if prediction is None: status='missing'
        elif is_failure(prediction):
            status=('invalid_answer' if str(prediction.get('error','')).startswith('invalid_answer:')
                    and not prediction.get('hit_token_limit') else 'generation_error')
        else:
            choice=parse_answer(prediction.get('output'),num_options)
            recorded=prediction.get('predicted_index')
            if recorded is not None and (isinstance(recorded,bool) or not isinstance(recorded,int) or recorded!=choice):
                raise ValueError('predicted index disagrees with strict text parse')
            status='valid' if choice is not None else 'invalid_answer'
        rows.append({'id':ident,'video_id':prediction.get('video_id',ident) if prediction else ident,
                     'answer_index':label,'predicted_index':choice,'status':status,
                     'correct':status=='valid' and choice==label})
    counts=Counter(row['status'] for row in rows)
    correct=sum(row['correct'] for row in rows)
    return {'complete':counts['valid']==len(expected),'expected':len(expected),
            'attempted':len(final),'valid_predictions':counts['valid'],'correct':correct,
            'accuracy':correct/len(expected),
            'accuracy_among_valid':correct/counts['valid'] if counts['valid'] else None,
            'status_counts':{status:counts[status] for status in ('valid','invalid_answer','generation_error','missing')},
            'superseded_error_attempts':len(superseded),
            'run_fingerprint':next(iter(fingerprints),None),'rows':rows,
            'denominator_policy':'all frozen expected questions; missing/invalid/errors count incorrect'}
