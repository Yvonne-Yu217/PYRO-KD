"""Generation output files: append-only JSONL that grows across requeued jobs.

A requeued job appends to the same file, so one moment id can appear several
times (error rows from earlier attempts, then possibly a success). Everything
that reads these files (the runner deciding what is left to do, the meta file,
and the evaluator) must agree on what each moment's final state is, so that
definition lives here:

- A success supersedes any error rows for the same id.
- Otherwise the latest error row stands, so repeated failures count once.
- Rows without an id pass through unchanged.
- A legacy row with hit_token_limit=true and no error field counts as an error row
  (see is_failure).

Retry policy: a token-limit failure is deterministic when it was produced at
temperature 0 with at least the current max_new_tokens. Running it again would
reproduce the same cut-off and append another error, so it is treated as settled.
Raising --max-new-tokens (or sampling at temperature > 0) makes it eligible for a
retry. A complete temperature-0 decode rejected for its content (over the word
budget, or an invalid answer letter) is settled too, whatever the token budget.
Every other error, such as an out-of-memory failure, is retried.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    """All JSON object rows in file order. Partial lines from a killed job are skipped."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def exceeds_word_budget(row: dict[str, Any]) -> bool:
    """AD word budgets count whitespace-delimited words; preserve the full decode."""
    budget, text = row.get('word_limit'), row.get('output')
    return (isinstance(budget, int) and not isinstance(budget, bool) and budget >= 0
            and isinstance(text, str) and len(text.split()) > budget)


def is_failure(row: dict[str, Any]) -> bool:
    """True for an error row, including a legacy truncated row.

    Before token-limit cut-offs were recorded as errors, a row that hit max_new_tokens
    was written with hit_token_limit=true and no "error" field. Those rows are still on
    disk, and their text is cut off (for Muse, still the scratchpad), so they are
    failures here: never scored, never counted as successes, and retried when the
    token budget is raised. Every reader (runner, meta, evaluator) goes through this
    one definition.
    """
    return ("error" in row or row.get("hit_token_limit") is True
            or row.get('word_limit_exceeded') is True or exceeds_word_budget(row))


def is_success(row: dict[str, Any]) -> bool:
    return not is_failure(row)


def split_rows(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(final rows, superseded error rows), each in file order.

    Final: one state per moment id (see the module docstring). Superseded: error
    rows that no longer stand because the id later succeeded or failed again.
    They are returned, not dropped, so a report can show the retry history.
    Several successes for the same id are all kept as final (that is an upstream
    inconsistency the evaluator reports); only error rows are ever superseded.
    """
    rows = list(rows)
    has_success = {row["id"] for row in rows if "id" in row and is_success(row)}
    last_error: dict[Any, int] = {}
    for index, row in enumerate(rows):
        if "id" in row and not is_success(row):
            last_error[row["id"]] = index
    kept, superseded = [], []
    for index, row in enumerate(rows):
        if "id" not in row or is_success(row):
            kept.append(row)
        elif row["id"] not in has_success and last_error[row["id"]] == index:
            kept.append(row)
        else:
            superseded.append(row)
    return kept, superseded


def final_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The final state of each moment id (see split_rows)."""
    return split_rows(rows)[0]


def status_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Counts over the final state of each moment, not over every line ever written."""
    final = final_rows(rows)
    return {
        "n_ok": sum(is_success(row) for row in final),
        "n_error": sum(not is_success(row) for row in final),
        "n_hit_token_limit": sum(
            bool(row.get("hit_token_limit")) and not is_success(row) for row in final
        ),
    }


def is_deterministic_failure(row: dict[str, Any], max_new_tokens: int, temperature: float) -> bool:
    """True if retrying this error row with the current settings would repeat it.

    Two kinds qualify, both only at temperature 0. A token-limit cut-off repeats unless the budget
    is raised. A complete decode rejected for its content (over the word budget, or not a valid
    option letter) repeats whatever the token budget, because greedy decoding of the same inputs
    returns the same text. Callers must only pass rows from unchanged inputs (the runners check an
    input or run fingerprint first). A failed budget repair is transient and stays retryable.
    """
    error = str(row.get("error") or "").lstrip()
    if (
        temperature == 0 and row.get("temperature") == 0 and row.get("hit_token_limit") is not True
        and not row.get("repair_error")
        and (row.get("word_limit_exceeded") is True or error.startswith(("word_budget:", "invalid_answer:")))
    ):
        return True
    recorded_tokens = row.get("max_new_tokens")
    return bool(
        not is_success(row)
        and row.get("hit_token_limit") is True
        and temperature == 0
        and row.get("temperature") == 0
        and isinstance(recorded_tokens, int)
        and not isinstance(recorded_tokens, bool)
        and recorded_tokens >= max_new_tokens
    )


def settled_ids(
    rows: Iterable[dict[str, Any]], max_new_tokens: int | None = None, temperature: float = 0.0,
) -> tuple[set[str], set[str]]:
    """(ids needing no more work, the subset that are deterministic failures).

    Without max_new_tokens only successes count as settled.
    """
    done: set[str] = set()
    repeats: set[str] = set()
    final = final_rows(rows)
    for row in final:
        if "id" not in row:
            continue
        if is_success(row):
            done.add(row["id"])
        elif max_new_tokens is not None and is_deterministic_failure(row, max_new_tokens, temperature):
            done.add(row["id"])
            repeats.add(row["id"])
    return done, repeats


_ATTEMPT_KEYS = (
    "slurm_job_id", "host", "gpu", "n_gpus", "torch", "transformers", "model_class", "args", "provenance",
    "n_ok", "n_error", "n_hit_token_limit", "load_s", "generate_s",
    "peak_gpu_mem_gb", "peak_gpu_mem_gb_per_device",
)


def merge_meta(
    previous: dict[str, Any] | None, attempt: dict[str, Any], counts: dict[str, int],
) -> dict[str, Any]:
    """Cumulative run record across requeued attempts.

    attempt holds this attempt's own measurements. counts is the final state of
    the output file (status_counts), so n_ok / n_error / n_hit_token_limit describe
    the results as they stand now. load_s and generate_s are summed over attempts
    (the cost of the whole run), peak GPU memory is the maximum, and the per-attempt
    records are kept under "attempts". A single-attempt meta file written before
    this existed is carried over as the first attempt.
    """
    attempts: list[dict[str, Any]] = []
    if previous:
        attempts = [dict(a) for a in previous.get("attempts") or []]
        if not attempts and previous.get("model_id"):
            attempts = [{key: previous.get(key) for key in _ATTEMPT_KEYS if key in previous}]
    attempts.append({key: attempt.get(key) for key in _ATTEMPT_KEYS if key in attempt})

    load_s = sum(a.get("load_s") or 0.0 for a in attempts)
    generate_s = sum(a.get("generate_s") or 0.0 for a in attempts)
    peaks = [a for a in attempts if a.get("peak_gpu_mem_gb") is not None]
    heaviest = max(peaks, key=lambda a: a["peak_gpu_mem_gb"]) if peaks else {}
    meta = {key: attempt[key] for key in (
        "model_id", "model_class", "quant", "n_gpus", "args", "gpu", "torch", "transformers",
        "host", "slurm_job_id", "provenance",
    ) if key in attempt}
    meta.update(
        n_ok=counts["n_ok"],
        n_error=counts["n_error"],
        n_hit_token_limit=counts["n_hit_token_limit"],
        load_s=round(load_s, 1),
        generate_s=round(generate_s, 1),
        s_per_item=round(generate_s / counts["n_ok"], 3) if counts["n_ok"] else None,
        peak_gpu_mem_gb=heaviest.get("peak_gpu_mem_gb"),
        peak_gpu_mem_gb_per_device=heaviest.get("peak_gpu_mem_gb_per_device"),
        n_attempts=len(attempts),
        attempts=attempts,
    )
    return meta
