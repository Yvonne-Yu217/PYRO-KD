"""Wilcoxon signed-rank test, Holm correction and exact McNemar (standard library).

Use with per-video (or per-question) paired values, never per moment: moments
within a video are not independent. Bootstrap CIs live in stats.py.

wilcoxon: zero differences are dropped ("wilcox" method). With no ties and
n <= 25 the two-sided p-value is exact (distribution of W+ by dynamic
programming); otherwise the normal approximation with tie correction and a
continuity correction is used. The method used is reported.
"""

from __future__ import annotations

import math
from typing import Sequence


def _ranks(values: Sequence[float]) -> tuple[list[float], list[int]]:
    """Average ranks (1-based) and the sizes of tie groups."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    ties = []
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        ties.append(j - i + 1)
        i = j + 1
    return ranks, ties


def _exact_two_sided(w_plus: float, n: int) -> float:
    total = n * (n + 1) // 2
    counts = [0] * (total + 1)
    counts[0] = 1
    for r in range(1, n + 1):
        for s in range(total, r - 1, -1):
            counts[s] += counts[s - r]
    denominator = 2 ** n
    w = min(w_plus, total - w_plus)
    tail = sum(counts[s] for s in range(0, int(math.floor(w)) + 1)) / denominator
    return min(1.0, 2.0 * tail)


def wilcoxon_signed_rank(a: Sequence[float | None], b: Sequence[float | None]) -> dict:
    """Paired two-sided test of a vs b; pairs with a None are dropped."""
    if len(a) != len(b):
        raise ValueError("a and b must be aligned")
    diffs = [float(x) - float(y) for x, y in zip(a, b) if x is not None and y is not None]
    n_pairs = len(diffs)
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return {"n_pairs": n_pairs, "n_nonzero": 0, "w_plus": None, "w_minus": None, "p_value": None,
                "method": "no nonzero differences", "median_diff": _median(diffs)}
    ranks, ties = _ranks([abs(d) for d in nonzero])
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    if n <= 25 and all(t == 1 for t in ties):
        p, method = _exact_two_sided(w_plus, n), "exact"
    else:
        mean = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0 - sum(t ** 3 - t for t in ties) / 48.0
        if variance <= 0:
            p, method = 1.0, "normal (degenerate)"
        else:
            z = (abs(w_plus - mean) - 0.5) / math.sqrt(variance)
            p, method = min(1.0, math.erfc(max(z, 0.0) / math.sqrt(2.0))), "normal+tie+continuity"
    return {"n_pairs": n_pairs, "n_nonzero": n, "w_plus": w_plus, "w_minus": w_minus,
            "p_value": p, "method": method, "median_diff": _median(diffs)}


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def holm(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values (None passes through)."""
    indexed = [(p, i) for i, p in enumerate(p_values) if p is not None]
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(indexed)):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test for paired binary outcomes (e.g. QA timeline on/off)."""
    if len(a_correct) != len(b_correct):
        raise ValueError("outcomes must be aligned")
    only_a = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    only_b = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"only_a": only_a, "only_b": only_b, "p_value": min(1.0, 2.0 * tail)}
