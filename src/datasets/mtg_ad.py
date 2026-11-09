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
from src.utils.media import probe

from .srt import SubtitleItem

HF_BASE_URL = "https://huggingface.co/datasets/google/mind-the-gap-inline-ad/resolve/main"
SPLITS = ("validation", "test")
EXPECTED_SPLIT_COUNTS = {"validation": 332, "test": 101}
ANNOTATION_FILES = {
    "validation.jsonl": "data/annotations/validation.jsonl",
    "test.jsonl": "data/annotations/test.jsonl",
    "validation_video_ids.txt": "validation_video_ids.txt",
    "test_video_ids.txt": "test_video_ids.txt",
}
ANNOTATION_DOWNLOAD_TIMEOUT_SECONDS = 60
# <=720p H.264 keeps frames sharp and decodable while bounding download size.
# protocol=https is progressive HTTPS only: DASH (http_dash_segments) and HLS
# (m3u8_native) can stall at 0 bytes. Prefer the original-language audio track.
YTDLP_SOCKET_TIMEOUT_SECONDS = 20
YTDLP_DOWNLOAD_TIMEOUT_SECONDS = 600
YTDLP_FORMAT = (
    "bv*[height<=720][vcodec^=avc1][protocol=https]+ba[ext=m4a][format_note*=original][protocol=https]"
    "/bv*[height<=720][vcodec^=avc1][protocol=https]+ba[ext=m4a][protocol=https]"
    "/b[height<=720][protocol=https]"
)


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


def annotation_file_is_valid(path: str | Path, name: str | None = None) -> bool:
    """True when path is UTF-8 and a JSONL split has its official row count."""
    path = Path(path)
    name = name or path.name
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    if not name.endswith(".jsonl"):
        return True
    split = name.removesuffix(".jsonl")
    expected = EXPECTED_SPLIT_COUNTS.get(split)
    if expected is None:
        return False
    rows = [line for line in text.split("\n") if line.strip()]
    if len(rows) != expected:
        return False
    for line in rows:
        try:
            json.loads(line)
        except json.JSONDecodeError:
            return False
    return True


def download_annotations(data_dir: str | Path, overwrite: bool = False) -> list[Path]:
    """Fetch annotation JSONL and video-ID lists (~10 MB) into data_dir.

    A file is replaced only after the downloaded bytes decode as UTF-8. JSONL
    splits must also contain the official 332 validation and 101 test rows.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, remote in ANNOTATION_FILES.items():
        target = data_dir / name
        if not overwrite and target.exists() and annotation_file_is_valid(target):
            paths.append(target)
            continue
        temporary = target.with_suffix(target.suffix + ".tmp")
        try:
            with urllib.request.urlopen(
                f"{HF_BASE_URL}/{remote}",
                timeout=ANNOTATION_DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                temporary.write_bytes(response.read())
            if not annotation_file_is_valid(temporary, name):
                raise RuntimeError(f"{name} is not complete UTF-8 annotation data")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
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
    with (Path(data_dir) / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                videos.append(parse_row(json.loads(line)))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{split}.jsonl line {line_number}: {exc}") from exc
    return videos


def _existing_video_is_reusable(path: Path) -> bool:
    """A kept download must probe as a video. Undecodable files are fetched again."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found on PATH; install FFmpeg")
    try:
        return probe(path).has_video
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return False


def download_video(
    video: MTGVideo,
    output_dir: str | Path,
    overwrite: bool = False,
    cookies: str | Path | None = None,
    cookies_from_browser: str | None = None,
    extractor_args: str | None = None,
) -> Path:
    """Download one video with yt-dlp as <video_id>.mp4 (raises if unavailable)."""
    if cookies and cookies_from_browser:
        raise ValueError("pass either cookies or cookies_from_browser, not both")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / video.video_filename
    if (
        target.is_file()
        and target.stat().st_size > 0
        and not overwrite
        and _existing_video_is_reusable(target)
    ):
        return target
    target.unlink(missing_ok=True)
    ytdlp = shutil.which("yt-dlp")
    if ytdlp is None:
        raise RuntimeError("yt-dlp not found on PATH (pip install yt-dlp)")
    command = [
        ytdlp, "--no-progress", "--no-playlist", "--retries", "3",
        "--fragment-retries", "3",
        "--socket-timeout", str(YTDLP_SOCKET_TIMEOUT_SECONDS),
        "-f", YTDLP_FORMAT,
        "--merge-output-format", "mp4", "-o", str(output_dir / "%(id)s.%(ext)s"),
    ]
    if cookies:
        command.extend(("--cookies", str(cookies)))
    elif cookies_from_browser:
        command.extend(("--cookies-from-browser", cookies_from_browser))
    if extractor_args:
        command.extend(("--extractor-args", extractor_args))
    command.append(video.youtube_url)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=YTDLP_DOWNLOAD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"yt-dlp timed out after {YTDLP_DOWNLOAD_TIMEOUT_SECONDS}s for {video.video_id}"
        ) from exc
    if result.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
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
