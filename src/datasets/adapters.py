"""Dataset adapters: one interface over every benchmark, plus validators.

Every adapter yields VideoRecord objects with the same fields. Field names are
never assumed to be shared between datasets: each adapter maps its own,
documented source schema, and anything it cannot establish is reported by
validate() instead of guessed.

| adapter      | schema status | source |
|--------------|---------------|--------|
| mtg-ad       | established (HF release rows; parser in mtg_ad.py) | HF layout or the Globus copy: <root>/{validation,test}.jsonl + <root>/videos/<id>.mp4 |
| reframed     | established (HF release; parser in reframed.py) | <root>/{validation,test}.jsonl (+ keys/ for test) |
| egoschema    | established (questions.json + subset_answers.json) | <root>/ + <root>/videos/<q_uid>.mp4 |
| canonical    | defined here (CANONICAL_SCHEMA) | any dataset converted to it, e.g. Cue2Narrate or a long-form QA set |

Cue2Narrate and the long-form QA benchmark (Video-MME long or similar) have
no adapter of their own because their release schemas have not been verified
in this project (proposal §6). Convert them to the canonical JSONL with a
small script once the files are in hand, and validate with validate_canonical.

The Globus copy of MTG-AD also contains transcripts/, audio_descriptions/,
lists/, video_manifest.jsonl, unavailable_videos.json and README.txt. Their
formats have not been seen here, so they are inventoried, not parsed: the
annotation rows already carry the transcript and AD segments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable

from src.datasets import egoschema, mtg_ad, reframed


@dataclass(frozen=True)
class ADEvent:
    start: float
    end: float | None
    text: str


@dataclass(frozen=True)
class QAItem:
    question_id: str
    video_id: str
    question: str
    options: tuple[str, ...]
    answer: int | None  # None when the label is not public


@dataclass
class VideoRecord:
    dataset: str
    video_id: str
    split: str
    duration: float | None
    video_path: Path | None
    ad: list[ADEvent] | None = None           # None: this dataset has no AD
    transcript: list[dict] | None = None      # [{"start", "end", "text"}]; None: not provided
    qa: list[QAItem] = field(default_factory=list)
    group: str | None = None                  # independence cluster (e.g. movie for REFRAMED)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ad_starts(self) -> list[float]:
        return [e.start for e in self.ad or []]


@dataclass
class ValidationReport:
    dataset: str
    root: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, Any] = field(default_factory=dict)
    unvalidated: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"dataset": self.dataset, "root": self.root, "ok": self.ok, "errors": self.errors,
                "warnings": self.warnings, "counts": self.counts, "unvalidated": self.unvalidated}


class MTGADAdapter:
    name = "mtg-ad"
    capabilities = frozenset({"ad_placement", "ad_text", "audio", "transcript"})
    splits = mtg_ad.SPLITS

    def __init__(self, root: str | Path, video_dir: str | Path | None = None):
        self.root = Path(root)
        self.video_dir = Path(video_dir) if video_dir else self.root / "videos"

    def load(self, split: str) -> list[VideoRecord]:
        records = []
        for v in mtg_ad.load_split(self.root, split):
            path = self.video_dir / v.video_filename
            records.append(VideoRecord(
                dataset=self.name, video_id=v.video_id, split=split, duration=v.duration_seconds,
                video_path=path if path.is_file() else None,
                ad=[ADEvent(a.timestamp, None, a.text) for a in v.ad],
                transcript=[{"start": t.start, "end": t.end, "text": t.text} for t in v.transcript],
                group=v.video_id, meta={"category": v.category},
            ))
        return records

    def unavailable_ids(self) -> tuple[set[str], str | None]:
        """IDs listed in unavailable_videos.json, if its shape is a list of IDs or an object keyed by ID."""
        path = self.root / "unavailable_videos.json"
        if not path.is_file():
            return set(), None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return set(), f"unavailable_videos.json unreadable: {exc}"
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return set(data), None
        if isinstance(data, dict) and all(isinstance(k, str) for k in data):
            return set(data), None
        if isinstance(data, list) and all(isinstance(x, dict) and isinstance(x.get("video_id"), str) for x in data):
            return {x["video_id"] for x in data}, None
        return set(), "unavailable_videos.json has an unrecognized shape; not used"

    def validate(self) -> ValidationReport:
        report = ValidationReport(self.name, str(self.root))
        all_ids: set[str] = set()
        for split in self.splits:
            path = self.root / f"{split}.jsonl"
            if not path.is_file():
                report.errors.append(f"missing {path}")
                continue
            try:
                videos = mtg_ad.load_split(self.root, split)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                report.errors.append(f"{split}.jsonl does not parse: {exc}")
                continue
            expected = mtg_ad.EXPECTED_SPLIT_COUNTS[split]
            if len(videos) != expected:
                report.errors.append(f"{split}: {len(videos)} rows, expected {expected}")
            out_of_bounds = sum(1 for v in videos for a in v.ad if a.timestamp > v.duration_seconds + 1.0)
            report.counts[split] = {
                "videos": len(videos), "ad_segments": sum(len(v.ad) for v in videos),
                "no_speech_videos": sum(not v.transcript for v in videos),
                "ad_starts_beyond_duration": out_of_bounds,
            }
            all_ids |= {v.video_id for v in videos}
        unavailable, problem = self.unavailable_ids()
        if problem:
            report.warnings.append(problem)
        if self.video_dir.is_dir():
            present = {p.stem for p in self.video_dir.glob("*.mp4")}
            report.counts["mp4_files"] = len(present)
            report.counts["annotated_with_video"] = len(all_ids & present)
            missing = sorted(all_ids - present - unavailable)
            if missing:
                report.warnings.append(f"{len(missing)} annotated videos have no MP4 and are not listed unavailable "
                                       f"(first: {', '.join(missing[:5])})")
            extra = sorted(present - all_ids)
            if extra:
                report.warnings.append(f"{len(extra)} MP4s are not in any split (first: {', '.join(extra[:5])})")
            report.counts["listed_unavailable"] = len(unavailable)
        else:
            report.errors.append(f"no video directory at {self.video_dir}")
        manifest = self.root / "video_manifest.jsonl"
        if manifest.is_file():
            keys: set[str] = set()
            bad = 0
            for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    keys |= set(row) if isinstance(row, dict) else set()
                except json.JSONDecodeError:
                    bad += 1
            report.counts["video_manifest_keys"] = sorted(keys)
            if bad:
                report.warnings.append(f"video_manifest.jsonl: {bad} unparsable lines")
            report.unvalidated.append("video_manifest.jsonl: keys inventoried, not used")
        for name in ("transcripts", "audio_descriptions", "lists"):
            d = self.root / name
            if d.is_dir():
                files = [p for p in d.rglob("*") if p.is_file()]
                report.counts[f"{name}_files"] = len(files)
                report.counts[f"{name}_suffixes"] = sorted({p.suffix for p in files})
                report.unvalidated.append(f"{name}/: inventoried only; annotation rows are the source of truth")
        if (self.root / "README.txt").is_file():
            report.unvalidated.append("README.txt present: read it and confirm the layout matches this adapter")
        return report


class ReframedAdapter:
    name = "reframed"
    capabilities = frozenset({"ad_placement", "ad_text", "audio", "transcript"})
    splits = ("validation", "test")

    def __init__(self, root: str | Path, video_dir: str | Path | None = None, ad_version: str = "us"):
        self.root = Path(root)
        self.video_dir = Path(video_dir) if video_dir else self.root / "videos"
        self.ad_version = ad_version

    def load(self, split: str) -> list[VideoRecord]:
        out = []
        for c in reframed.load_split(self.root, split):
            path = self.video_dir / f"{c.video_id}.mp4"
            track = reframed.merge_cues(c.ad_track(self.ad_version))
            out.append(VideoRecord(
                dataset=self.name, video_id=c.video_id, split=split, duration=c.video_duration,
                video_path=path if path.is_file() else None,
                ad=[ADEvent(i.start, i.end, i.text) for i in track],
                transcript=[{"start": s.start, "end": s.end, "text": s.text} for s in c.subtitles],
                group=c.movie_id, meta={"ad_version": self.ad_version, "imdb_id": c.imdb_id},
            ))
        return out

    def validate(self) -> ValidationReport:
        report = ValidationReport(self.name, str(self.root))
        for split in self.splits:
            if not (self.root / f"{split}.jsonl").is_file():
                report.errors.append(f"missing {split}.jsonl")
                continue
            try:
                clips = reframed.load_split(self.root, split)
            except Exception as exc:  # decryption keys missing, parse errors
                report.errors.append(f"{split} does not load: {exc}")
                continue
            report.counts[split] = {"clips": len(clips), "movies": len({c.movie_id for c in clips}),
                                    "with_ad": sum(bool(c.ad_us or c.ad_uk) for c in clips)}
        report.unvalidated.append("video downloads: not verified in this project (docs/week1_spec.md)")
        return report


class EgoSchemaAdapter:
    name = "egoschema"
    capabilities = frozenset({"qa"})  # "audio" only after the §6 check passes
    splits = ("subset",)

    def __init__(self, root: str | Path, video_dir: str | Path | None = None, clip_seconds: float = 180.0):
        self.root = Path(root)
        self.video_dir = Path(video_dir) if video_dir else self.root / "videos"
        self.clip_seconds = clip_seconds

    def load(self, split: str = "subset") -> list[VideoRecord]:
        out = []
        for item in egoschema.load_items(self.root, labeled_only=True):
            path = self.video_dir / item.video_filename
            out.append(VideoRecord(
                dataset=self.name, video_id=item.q_uid, split=split, duration=None,
                video_path=path if path.is_file() else None,
                qa=[QAItem(item.q_uid, item.q_uid, item.question, item.options, item.answer)],
                group=item.q_uid, meta={"nominal_seconds": self.clip_seconds},
            ))
        return out

    def validate(self) -> ValidationReport:
        report = ValidationReport(self.name, str(self.root))
        try:
            items = egoschema.load_items(self.root, labeled_only=True)
            report.counts["labeled_questions"] = len(items)
        except Exception as exc:
            report.errors.append(f"annotations do not load: {exc}")
            return report
        if self.video_dir.is_dir():
            report.counts["videos_present"] = sum((self.video_dir / i.video_filename).is_file() for i in items)
        else:
            report.warnings.append(f"no video directory at {self.video_dir}")
        report.unvalidated.append("audio presence: run scripts/precommit_checks.py egoschema-audio (§6)")
        return report


CANONICAL_SCHEMA = {
    "video_id": "str, unique",
    "split": "str",
    "video_path": "str, absolute or relative to the JSONL's directory",
    "duration": "float seconds, optional (probed when absent)",
    "ad": "optional list of {start: float, end: float|null, text: str}",
    "transcript": "optional list of {start: float, end: float, text: str}",
    "qa": "optional list of {question_id: str, question: str, options: [str], answer: int|null}",
    "group": "optional independence cluster id (e.g. movie)",
}


def _finite_nonneg(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0


def validate_canonical_row(row: Any) -> str | None:
    if not isinstance(row, dict):
        return "row is not an object"
    if not isinstance(row.get("video_id"), str) or not row["video_id"]:
        return "video_id must be a nonempty string"
    if not isinstance(row.get("split"), str):
        return "split must be a string"
    if "duration" in row and row["duration"] is not None and not _finite_nonneg(row["duration"]):
        return "duration must be a finite non-negative number"
    for i, e in enumerate(row.get("ad") or []):
        if not isinstance(e, dict) or not _finite_nonneg(e.get("start")) or not isinstance(e.get("text"), str):
            return f"ad[{i}] needs start >= 0 and text"
        if e.get("end") is not None and (not _finite_nonneg(e["end"]) or e["end"] < e["start"]):
            return f"ad[{i}] end must be >= start"
    for i, t in enumerate(row.get("transcript") or []):
        if not isinstance(t, dict) or not _finite_nonneg(t.get("start")) or not _finite_nonneg(t.get("end")) \
                or t["end"] < t["start"] or not isinstance(t.get("text"), str):
            return f"transcript[{i}] needs 0 <= start <= end and text"
    for i, q in enumerate(row.get("qa") or []):
        if not isinstance(q, dict) or not isinstance(q.get("question"), str) or not isinstance(q.get("options"), list) \
                or len(q["options"]) < 2 or not all(isinstance(o, str) for o in q["options"]):
            return f"qa[{i}] needs a question and at least two string options"
        answer = q.get("answer")
        if answer is not None and (isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(q["options"])):
            return f"qa[{i}] answer must be an option index or null"
    return None


class CanonicalAdapter:
    """Any dataset converted to CANONICAL_SCHEMA (one JSON object per video per line)."""

    capabilities = frozenset({"ad_placement", "ad_text", "qa", "audio", "transcript"})

    def __init__(self, path: str | Path, name: str = "canonical"):
        self.path = Path(path)
        self.name = name

    def _rows(self) -> Iterable[tuple[int, Any]]:
        for n, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                yield n, json.loads(line)

    def load(self, split: str | None = None) -> list[VideoRecord]:
        out = []
        for n, row in self._rows():
            problem = validate_canonical_row(row)
            if problem:
                raise ValueError(f"{self.path}:{n}: {problem}")
            if split is not None and row["split"] != split:
                continue
            path = Path(row["video_path"]) if row.get("video_path") else None
            if path is not None and not path.is_absolute():
                path = self.path.parent / path
            out.append(VideoRecord(
                dataset=self.name, video_id=row["video_id"], split=row["split"], duration=row.get("duration"),
                video_path=path if path is not None and path.is_file() else None,
                ad=[ADEvent(float(e["start"]), e.get("end"), e["text"]) for e in row["ad"]] if row.get("ad") is not None else None,
                transcript=row.get("transcript"),
                qa=[QAItem(q.get("question_id", f"{row['video_id']}_{i}"), row["video_id"], q["question"],
                           tuple(q["options"]), q.get("answer")) for i, q in enumerate(row.get("qa") or [])],
                group=row.get("group", row["video_id"]),
            ))
        return out

    def validate(self) -> ValidationReport:
        report = ValidationReport(self.name, str(self.path))
        if not self.path.is_file():
            report.errors.append(f"missing {self.path}")
            return report
        ids: set[str] = set()
        n_rows = 0
        try:
            for n, row in self._rows():
                n_rows += 1
                problem = validate_canonical_row(row)
                if problem:
                    report.errors.append(f"line {n}: {problem}")
                elif row["video_id"] in ids:
                    report.errors.append(f"line {n}: duplicate video_id {row['video_id']}")
                else:
                    ids.add(row["video_id"])
        except json.JSONDecodeError as exc:
            report.errors.append(f"not JSONL: {exc}")
        report.counts["rows"] = n_rows
        return report


def build_adapter(spec: dict):
    """spec: {"name": "mtg-ad"|"reframed"|"egoschema"|"canonical", "root"|"path": ..., ...}."""
    name = spec["name"]
    if name == "mtg-ad":
        return MTGADAdapter(spec["root"], spec.get("video_dir"))
    if name == "reframed":
        return ReframedAdapter(spec["root"], spec.get("video_dir"), spec.get("ad_version", "us"))
    if name == "egoschema":
        return EgoSchemaAdapter(spec["root"], spec.get("video_dir"))
    if name == "canonical" or spec.get("format") == "canonical":
        return CanonicalAdapter(spec["path"], spec.get("label", name))
    raise ValueError(f"unknown dataset adapter {name!r}")
