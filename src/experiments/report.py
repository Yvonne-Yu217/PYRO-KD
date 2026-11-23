"""Final tables (CSV + Markdown) and plots (SVG, no plotting dependency) from eval/*.json."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Sequence


def _f(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci(stat: dict | None, digits: int = 3) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    return f"{stat['mean']:.{digits}f} [{stat['ci_low']:.{digits}f}, {stat['ci_high']:.{digits}f}]"


def write_table(rows: Sequence[dict], columns: Sequence[str], stem: Path, title: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    stem.with_suffix(".csv").write_text(buf.getvalue(), encoding="utf-8")
    lines = [f"## {title}", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def selection_tables(metrics: dict, out_dir: Path) -> None:
    for key, stem, title in (
        ("summary_common", "selection_summary", "on the videos every core selector scored"),
        ("summary", "selection_summary_all", "each on every video it could run on (rows not comparable when n differs)"),
    ):
        _selection_table(metrics, key, out_dir / stem, title)
    _comparison_table(metrics, out_dir)


def _selection_table(metrics: dict, key: str, stem: Path, title: str) -> None:
    rows = []
    for name, run in metrics["runs"].items():
        s = run.get(key) or run["summary"]
        rows.append({
            "selector": name + (" (oracle)" if run["oracle"] else ""), "ablation": run.get("ablation") or "core",
            "videos": s["n_videos"], "macro_F1 [95% CI]": _ci(s["macro_f1"]),
            "P": _f(s["macro_precision"]["mean"]), "R": _f(s["macro_recall"]["mean"]),
            "overlap_s [95% CI]": _ci(s["overlap_seconds"], 2), "silence_util": _f(s["silence_utilization"]),
            "span": _f(s["span_fraction"]), "busiest_gap": _f(s["busiest_gap_share"]),
            "recall_silent_GT": _f(s["recall_on_silent_gt"]), "shortfall_videos": s["videos_with_shortfall"],
            "not_run": json.dumps(run["not_run"]) if run["not_run"] else "",
        })
    write_table(rows, list(rows[0].keys()) if rows else ["selector"], stem,
                f"Selector-only placement (tolerance +/-{metrics['tolerance_seconds']} s), {title}")


def _comparison_table(metrics: dict, out_dir: Path) -> None:
    comp = []
    for c in metrics["comparisons"]:
        b, w = c["bootstrap"], c["wilcoxon"]
        comp.append({
            "comparison": f"{c['selector']} - {c['baseline']}" + (" (oracle bound)" if c["oracle_upper_bound"] else ""),
            "metric": c["metric"], "n": c["n_shared_videos"],
            "mean_diff [95% CI]": "n/a" if b["mean_diff"] is None else f"{b['mean_diff']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]",
            "wilcoxon_p": _f(w.get("p_value"), 4), "p_holm": _f(w.get("p_holm"), 4), "method": w.get("method"),
        })
    write_table(comp, ["comparison", "metric", "n", "mean_diff [95% CI]", "wilcoxon_p", "p_holm", "method"],
                out_dir / "selection_comparisons", "Paired comparisons against the baseline (per video)")


def generation_table(results: Sequence[dict], out_dir: Path) -> None:
    rows = []
    for r in sorted(results, key=lambda r: (r["selector"], r["prior"], r["model"], r["timeline"])):
        tm = r.get("text_metrics", {})

        def text(name):
            m = tm.get(name, {})
            return _ci(m.get("per_video_mean")) if m.get("status") == "ok" else m.get("status", "n/a")

        rows.append({"selector": r["selector"] + (" (oracle)" if r.get("oracle") else ""), "prior": r["prior"],
                     "model": r["model"], "timeline": r["timeline"], "videos": r["n_videos"],
                     "macro_F1": _ci(r["macro_f1"]), "overlap_s": _ci(r["overlap_seconds"], 2),
                     "over_budget": _ci(r["over_budget_rate"]), "BERTScore": text("bertscore"),
                     "sent_cos": text("sentence_cosine"), "CLIP_text": text("clip_text")})
    write_table(rows, list(rows[0].keys()) if rows else ["selector"], out_dir / "generation_matrix",
                "Generation matrix (selector x prior x model)")


def bar_chart_svg(items: Sequence[tuple[str, float | None, float | None, float | None]], title: str, width: int = 720) -> str:
    """Horizontal bars with CI whiskers: (label, mean, low, high), values in [0, 1]."""
    row_h, left = 26, 170
    height = 40 + row_h * len(items)
    usable = width - left - 40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" font-family="sans-serif" font-size="12">',
           f'<text x="6" y="18" font-weight="bold">{title}</text>']
    for i, (label, mean, low, high) in enumerate(items):
        y = 30 + i * row_h
        out.append(f'<text x="6" y="{y + 15}">{label[:24]}</text>')
        if mean is None:
            out.append(f'<text x="{left}" y="{y + 15}" fill="#888">n/a</text>')
            continue
        out.append(f'<rect x="{left}" y="{y + 3}" width="{usable * max(0.0, min(1.0, mean)):.1f}" height="{row_h - 8}" fill="#4a78b5"/>')
        if low is not None and high is not None:
            x1, x2 = left + usable * max(0.0, low), left + usable * min(1.0, high)
            out.append(f'<line x1="{x1:.1f}" x2="{x2:.1f}" y1="{y + row_h / 2}" y2="{y + row_h / 2}" stroke="#222" stroke-width="1.5"/>')
        out.append(f'<text x="{left + usable * min(1.0, mean) + 6:.1f}" y="{y + 15}">{mean:.3f}</text>')
    out.append("</svg>")
    return "\n".join(out)


def make_report(exp_dir: Path) -> list[Path]:
    written = []
    tables, plots = exp_dir / "tables", exp_dir / "plots"
    sel_path = exp_dir / "eval" / "selection_metrics.json"
    if sel_path.is_file():
        metrics = json.loads(sel_path.read_text(encoding="utf-8"))
        selection_tables(metrics, tables)
        items = [(name + (" (O)" if run["oracle"] else ""), run["summary_common"]["macro_f1"]["mean"],
                  run["summary_common"]["macro_f1"]["ci_low"], run["summary_common"]["macro_f1"]["ci_high"])
                 for name, run in metrics["runs"].items() if run["ablation"] is None]
        plots.mkdir(parents=True, exist_ok=True)
        (plots / "selection_macro_f1.svg").write_text(bar_chart_svg(items, "Core selectors: macro F1 (+/-2 s), 95% bootstrap CI, common videos"),
                                                      encoding="utf-8")
        written += [tables / "selection_summary.md", plots / "selection_macro_f1.svg"]
    gen = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((exp_dir / "eval").glob("generation_*.json"))]
    if gen:
        generation_table(gen, tables)
        written.append(tables / "generation_matrix.md")
    qa_path = exp_dir / "eval" / "qa_metrics.json"
    if qa_path.is_file():
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        rows = [{"run": rid, "selector": r["selector"], "model": r["model"], "timeline": r["timeline"], "n": r["n"],
                 "accuracy": _ci(r["accuracy"]) if r["accuracy"] else "n/a", "unparsed": r["n_unparsed"]}
                for rid, r in qa["runs"].items()]
        write_table(rows, ["run", "selector", "model", "timeline", "n", "accuracy", "unparsed"], tables / "qa_accuracy", "QA accuracy")
        written.append(tables / "qa_accuracy.md")
    return written
