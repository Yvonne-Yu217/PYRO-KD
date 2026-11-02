"""Mind the Gap (MTG-AD) inline-AD dataset: annotations, videos, fixed splits.

Source: https://huggingface.co/datasets/google/mind-the-gap-inline-ad
(CC BY 4.0), code at https://github.com/google/mtg-ad. 433 YouTube videos:
332 validation (dev, exemplars) and 101 test. Ground-truth AD has start times
only; transcripts are timestamped ASR segments. Videos are fetched from
YouTube with yt-dlp and never committed.
"""

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
import subprocess
from typing import Iterable
import urllib.request

from src.evaluation.parsing import ADItem

from .srt import SubtitleItem

HF_BASE_URL = "https://huggingface.co/datasets/google/mind-the-gap-inline-ad/resolve/main"
SPLITS = ("validation", "test")
ANNOTATION_FILES = {
    "validation.jsonl": "data/annotations/validation.jsonl",
    "test.jsonl": "data/annotations/test.jsonl",
    "validation_video_ids.txt": "validation_video_ids.txt",
    "test_video_ids.txt": "test_video_ids.txt",
}
# <=720p keeps frames sharp enough for the VLM while bounding download size.
YTDLP_FORMAT = "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]/b"


@dataclass(frozen=True)
class MTGVideo:
    """One MTG-AD video with time-sorted AD and transcript segments (seconds)."""

    video_id: str
    split: str
    category: str
    duration_seconds: float
    ad: list[ADItem]
    transcript: list[SubtitleItem]
    visual_info: str | None

    @property
    def youtube_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    @property
    def video_filename(self) -> str:
        return f"{self.video_id}.mp4"


def download_annotations(data_dir: str | Path, overwrite: bool = False) -> list[Path]:
    """Fetch annotation JSONL and video-ID lists (~10 MB) into data_dir."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, remote in ANNOTATION_FILES.items():
        target = data_dir / name
        if overwrite or not target.exists():
            with urllib.request.urlopen(f"{HF_BASE_URL}/{remote}") as response:
                target.write_bytes(response.read())
        paths.append(target)
    return paths


def parse_row(row: dict) -> MTGVideo:
    """Build an MTGVideo from one annotations row, sorting segments by time.

    A few validation scripts list segments out of order, so sorting here keeps
    one-to-one temporal matching and gap computations well defined.
    """
    ad = sorted(
        (ADItem(float(s["start_seconds"]), s["text"].strip()) for s in row["gt_ad_segments"]),
        key=lambda item: item.timestamp,
    )
    transcript = sorted(
        (
            SubtitleItem(float(s["start_seconds"]), float(s["end_seconds"]), s["text"].strip())
            for s in row.get("transcript_segments") or []
        ),
        key=lambda item: item.start,
    )
    return MTGVideo(
        video_id=row["video_id"],
        split=row["split"],
        category=row["category"],
        duration_seconds=float(row["duration_seconds"]),
        ad=ad,
        transcript=transcript,
        visual_info=row.get("visual_info"),
    )


def load_split(data_dir: str | Path, split: str) -> list[MTGVideo]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    videos = []
    with (Path(data_dir) / f"{split}.jsonl").open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                videos.append(parse_row(json.loads(line)))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{split}.jsonl line {line_number}: {exc}") from exc
    return videos


def download_video(video: MTGVideo, output_dir: str | Path, overwrite: bool = False) -> Path:
    """Download one video with yt-dlp as <video_id>.mp4 (raises if unavailable)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / video.video_filename
    if target.exists() and not overwrite:
        return target
    ytdlp = shutil.which("yt-dlp")
    if ytdlp is None:
        raise RuntimeError("yt-dlp not found on PATH (pip install yt-dlp)")
    result = subprocess.run(
        [
            ytdlp, "--no-progress", "--no-playlist", "-f", YTDLP_FORMAT,
            "--merge-output-format", "mp4", "-o", str(output_dir / "%(id)s.%(ext)s"),
            video.youtube_url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(f"yt-dlp failed for {video.video_id}: {result.stderr.strip()[-500:]}")
    return target


def _by_category(videos: Iterable[MTGVideo], seed: int) -> dict[str, list[MTGVideo]]:
    groups: dict[str, list[MTGVideo]] = defaultdict(list)
    for video in sorted(videos, key=lambda v: v.video_id):
        groups[video.category].append(video)
    rng = random.Random(seed)
    for category in sorted(groups):
        rng.shuffle(groups[category])
    return groups


def select_dev(
    videos: Iterable[MTGVideo],
    num_videos: int = 10,
    min_ad_items: int = 3,
    seed: int = 0,
) -> list[MTGVideo]:
    """Deterministic dev/pilot set: category-stratified among videos with at
    least min_ad_items AD segments, so the genre mix matches validation."""
    eligible = [v for v in videos if len(v.ad) >= min_ad_items]
    return select_stratified(eligible, num_videos, seed=seed)


def select_stratified(
    videos: Iterable[MTGVideo],
    num_videos: int,
    seed: int = 0,
) -> list[MTGVideo]:
    """Seeded sample whose category mix matches the population (largest remainder)."""
    groups = _by_category(videos, seed)
    total = sum(len(g) for g in groups.values())
    if num_videos >= total:
        return sorted((v for g in groups.values() for v in g), key=lambda v: v.video_id)
    quotas = {c: num_videos * len(g) / total for c, g in groups.items()}
    counts = {c: int(q) for c, q in quotas.items()}
    for category in sorted(quotas, key=lambda c: (counts[c] - quotas[c], c))[: num_videos - sum(counts.values())]:
        counts[category] += 1
    chosen = [v for c in sorted(groups) for v in groups[c][: counts[c]]]
    return sorted(chosen, key=lambda v: v.video_id)
