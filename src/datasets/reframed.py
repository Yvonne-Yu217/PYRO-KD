"""REFRAMED movie-AD dataset: metadata download, test decryption, parsing.

Source: https://huggingface.co/datasets/igorsterner/reframed (CC BY-NC 4.0;
source-video rights are separate, so never commit or redistribute videos).
All SRT times (AD US/UK, subtitles, SDH) are relative to the excerpt start;
video_start is the excerpt's offset within the full movie.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any, Callable, Iterable
import urllib.request
import zipfile

from .srt import SubtitleItem, parse_srt

HF_BASE_URL = "https://huggingface.co/datasets/igorsterner/reframed/resolve/main"
SPLITS = ("train", "validation", "test")
METADATA_FILES = ("validation.jsonl", "test.jsonl", "keys.zip", "videos_encrypted.sh")
# From the decrypted videos_encrypted.sh shipped with the dataset.
VIDEO_URL_TEMPLATE = "https://link.theplatform.com/s/NGweTC/media/{video_id}?formats=M3U+none"
TEST_SELECTION_VERSION = "movie-balanced-binary-availability-v2"

Decryptor = Callable[[str], str]


@dataclass(frozen=True)
class ReframedClip:
    """One REFRAMED movie excerpt with parsed SRT tracks."""

    split: str
    rottentomatoes_id: str
    imdb_id: str
    video_id: str
    ad_us: list[SubtitleItem]
    ad_uk: list[SubtitleItem]
    subtitles: list[SubtitleItem]
    sdh: list[SubtitleItem]
    screenplay: dict[str, str] | None
    movie_duration: float
    video_start: float
    video_duration: float

    @property
    def movie_id(self) -> str:
        """Cluster key: excerpts from one movie are not independent."""
        return self.imdb_id

    @property
    def video_url(self) -> str:
        return VIDEO_URL_TEMPLATE.format(video_id=self.video_id)

    def ad_track(self, version: str) -> list[SubtitleItem]:
        if version not in {"us", "uk"}:
            raise ValueError("version must be 'us' or 'uk'")
        return self.ad_us if version == "us" else self.ad_uk


def download_metadata(
    data_dir: str | Path,
    files: Iterable[str] = METADATA_FILES,
    overwrite: bool = False,
) -> list[Path]:
    """Fetch JSONL splits and keys from Hugging Face into data_dir.

    train.jsonl (18 MB, LFS) is excluded by default; pass it in files when
    exemplars are needed. keys.zip is extracted to data_dir/keys.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in files:
        target = data_dir / name
        if overwrite or not target.exists():
            with urllib.request.urlopen(f"{HF_BASE_URL}/{name}") as response:
                target.write_bytes(response.read())
        paths.append(target)
        if name == "keys.zip":
            with zipfile.ZipFile(target) as archive:
                archive.extractall(data_dir / "keys")
    return paths


def load_decryptor(keys_dir: str | Path) -> Decryptor:
    """Build a decrypt(value) function from the dataset's keys.zip contents.

    Follows the dataset card: an RSA-OAEP(SHA-256) private key unwraps a
    Fernet key, which decrypts each test-set field.
    """
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    keys_dir = Path(keys_dir)
    private_key_path = next(keys_dir.rglob("private_key.pem"), None)
    wrapped_key_path = next(keys_dir.rglob("skey.key"), None)
    if private_key_path is None or wrapped_key_path is None:
        raise FileNotFoundError(f"private_key.pem/skey.key not found under {keys_dir}")
    private_key = serialization.load_pem_private_key(
        private_key_path.read_bytes(), password=None
    )
    symmetric_key = private_key.decrypt(
        wrapped_key_path.read_bytes(),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    fernet = Fernet(symmetric_key)
    return lambda value: fernet.decrypt(value.encode("ascii")).decode("utf-8")


def decrypt_row(row: dict[str, str], decrypt: Decryptor) -> dict[str, Any]:
    """Decrypt every field; each decrypts to a JSON-encoded value."""
    return {key: json.loads(decrypt(value)) for key, value in row.items()}


def _parse_screenplay(value: Any) -> dict[str, str] | None:
    if value is None or value == "null":
        return None
    if isinstance(value, str):
        value = json.loads(value)
    return value or None


def parse_row(row: dict[str, Any], split: str) -> ReframedClip:
    """Convert one (decrypted) JSONL row into a ReframedClip."""
    return ReframedClip(
        split=split,
        rottentomatoes_id=row["rottentomatoes_id"],
        imdb_id=row["imdb_id"],
        video_id=row["video_id"],
        ad_us=parse_srt(row.get("audio_description_us")),
        ad_uk=parse_srt(row.get("audio_description_uk")),
        subtitles=parse_srt(row.get("subtitles")),
        sdh=parse_srt(row.get("sdh")),
        screenplay=_parse_screenplay(row.get("screenplay")),
        movie_duration=float(row["movie_duration"]),
        video_start=float(row["video_start"]),
        video_duration=float(row["video_duration"]),
    )


def load_split(
    data_dir: str | Path,
    split: str,
    decrypt: Decryptor | None = None,
) -> list[ReframedClip]:
    """Load a split from data_dir/<split>.jsonl.

    The test split is encrypted; if decrypt is None it is built from
    data_dir/keys. Never tune on test: fix the sample, then leave it until
    the Week 4 full runs.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    data_dir = Path(data_dir)
    if split == "test" and decrypt is None:
        decrypt = load_decryptor(data_dir / "keys")
    clips = []
    with (data_dir / f"{split}.jsonl").open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if split == "test":
                row = decrypt_row(row, decrypt)
            try:
                clips.append(parse_row(row, split))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{split}.jsonl line {line_number}: {exc}") from exc
    return clips


def download_video(
    clip: ReframedClip,
    output_dir: str | Path,
    overwrite: bool = False,
) -> Path:
    """Fetch the excerpt's original program audio + video as H.264/AAC MP4.

    Mirrors the dataset's videos_encrypted.sh. The stream is the theatrical
    mix without AD, so it is safe as model input.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{clip.video_id}.mp4"
    if target.exists() and not overwrite:
        return target
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    partial = target.with_suffix(".partial.mp4")
    result = subprocess.run(
        [
            ffmpeg, "-v", "error", "-y", "-analyzeduration", "1G", "-probesize", "1G",
            "-i", clip.video_url, "-c:v", "libx264", "-crf", "20", "-c:a", "aac",
            str(partial),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"download failed for {clip.video_id}: {result.stderr.strip()}")
    partial.rename(target)
    return target


def merge_cues(items: list[SubtitleItem], max_gap: float = 0.5) -> list[SubtitleItem]:
    """Join AD cues separated by <= max_gap seconds into one description.

    REFRAMED often splits one spoken sentence across consecutive cues, so cue
    counts overstate how many separate descriptions a describer inserted.
    """
    merged: list[SubtitleItem] = []
    for item in items:
        if merged and item.start - merged[-1].end <= max_gap:
            previous = merged[-1]
            merged[-1] = SubtitleItem(
                previous.start, max(previous.end, item.end), f"{previous.text} {item.text}"
            )
        else:
            merged.append(item)
    return merged


def select_pilot(
    clips: Iterable[ReframedClip],
    num_clips: int = 10,
    min_ad_items: int = 3,
    seed: int = 0,
) -> list[ReframedClip]:
    """Development-only pilot: one clip per movie, round-robin until num_clips.

    Within a movie, clips are shuffled with a fixed seed (after sorting by
    video_id) so the pilot is not biased toward unusually dense AD. Only clips
    with at least min_ad_items US cues and nonempty subtitles are eligible.
    """
    by_movie: dict[str, list[ReframedClip]] = {}
    for clip in clips:
        if len(clip.ad_us) >= min_ad_items and clip.subtitles:
            by_movie.setdefault(clip.movie_id, []).append(clip)
    rng = random.Random(seed)
    for movie_id in sorted(by_movie):
        by_movie[movie_id].sort(key=lambda c: c.video_id)
        rng.shuffle(by_movie[movie_id])
    pilot = []
    depth = 0
    while len(pilot) < num_clips and any(len(v) > depth for v in by_movie.values()):
        for movie_id in sorted(by_movie):
            if depth < len(by_movie[movie_id]) and len(pilot) < num_clips:
                pilot.append(by_movie[movie_id][depth])
        depth += 1
    return pilot


def select_test(
    clips: Iterable[ReframedClip], num_clips: int = 30, seed: int = 0,
) -> list[ReframedClip]:
    """Movie-balanced held-out sample with binary evaluability requirements.

    Sort IDs before seeded shuffling, then take one clip per movie per round.
    Require a nonempty US or UK AD track and nonempty subtitles, using only
    boolean availability. Never use cue counts, density, or any text content
    for eligibility/ranking. SDH and screenplay are not inspected. Pilot's
    minimum US cue-count criterion remains development-only.
    """
    if num_clips < 0:
        raise ValueError("num_clips must be nonnegative")
    by_movie: dict[str, list[ReframedClip]] = {}
    for clip in clips:
        if not (bool(clip.ad_us) or bool(clip.ad_uk)) or not bool(clip.subtitles):
            continue
        by_movie.setdefault(clip.movie_id, []).append(clip)
    rng = random.Random(seed)
    movie_ids = sorted(by_movie)
    rng.shuffle(movie_ids)
    for movie_id in movie_ids:
        by_movie[movie_id].sort(key=lambda clip: clip.video_id)
        rng.shuffle(by_movie[movie_id])
    selected = []
    depth = 0
    while len(selected) < num_clips and any(len(group) > depth for group in by_movie.values()):
        for movie_id in movie_ids:
            if depth < len(by_movie[movie_id]) and len(selected) < num_clips:
                selected.append(by_movie[movie_id][depth])
        depth += 1
    return selected
