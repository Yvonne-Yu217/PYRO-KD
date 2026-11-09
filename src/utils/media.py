"""Timestamp-preserving media loading built on FFmpeg/ffprobe.

Every derived artifact (16 kHz mono audio, sparse frames) is written next to a
manifest.json that records the source media info, extraction settings, and
the clip-relative timestamp of each frame. Selectors and generators should
read timestamps from the manifest, never re-derive them from file names.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Sequence

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 2


@dataclass(frozen=True)
class MediaInfo:
    """Probed container/stream metadata. Times are in seconds."""

    path: str
    duration: float
    start_time: float
    has_video: bool
    has_audio: bool
    width: int | None
    height: int | None
    video_fps: float | None
    audio_sample_rate: int | None
    audio_channels: int | None


@dataclass(frozen=True)
class FrameEntry:
    """A cached frame and its clip-relative timestamp in seconds."""

    index: int
    timestamp: float
    path: str


def _require(tool: str) -> str:
    resolved = shutil.which(tool)
    if resolved is None:
        raise RuntimeError(f"{tool} not found on PATH; install FFmpeg")
    return resolved


def _run(command: Sequence[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout


def _parse_rate(rate: str | None) -> float | None:
    if not rate or rate in {"0/0", "N/A"}:
        return None
    numerator, _, denominator = rate.partition("/")
    value = float(numerator) / float(denominator or 1)
    return value if math.isfinite(value) and value > 0 else None


def probe(path: str | Path) -> MediaInfo:
    """Return container and first audio/video stream metadata via ffprobe."""
    output = _run([
        _require("ffprobe"), "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    data = json.loads(output)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})
    duration = float(fmt.get("duration") or (video or audio or {}).get("duration") or 0.0)
    return MediaInfo(
        path=str(path),
        duration=duration,
        start_time=float(fmt.get("start_time") or 0.0),
        has_video=video is not None,
        has_audio=audio is not None,
        width=int(video["width"]) if video and "width" in video else None,
        height=int(video["height"]) if video and "height" in video else None,
        video_fps=_parse_rate(video.get("avg_frame_rate")) if video else None,
        audio_sample_rate=int(audio["sample_rate"]) if audio and "sample_rate" in audio else None,
        audio_channels=int(audio["channels"]) if audio and "channels" in audio else None,
    )


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16_000,
) -> Path:
    """Decode the first audio stream to mono 16-bit PCM WAV at sample_rate."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _run([
        _require("ffmpeg"), "-v", "error", "-y", "-i", str(video_path),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-c:a", "pcm_s16le", str(output_path),
    ])
    return output_path


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 2.0,
    max_side: int | None = 768,
) -> list[FrameEntry]:
    """Sample frames at a fixed rate; frame i is at clip time i / fps.

    The fps filter emits frame i at i / fps after the first video frame,
    choosing the nearest decoded frame, so timestamps are relative to the
    stream start like REFRAMED SRT cues. The container start offset is kept
    in the manifest's media_info for alignment checks. max_side downsizes the
    longer edge (never upsizes) to keep the cache small.
    """
    if not fps > 0:
        raise ValueError("fps must be positive")
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f"{output_dir.name}.incoming")
    backup = output_dir.with_name(f"{output_dir.name}.backup")
    for leftover in (staging, backup):
        if leftover.exists():
            shutil.rmtree(leftover)
    staging.mkdir()
    swapped = False
    try:
        filters = [f"fps={fps}"]
        if max_side:
            filters.append(
                f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
            )
        _run([
            _require("ffmpeg"), "-v", "error", "-y", "-i", str(video_path),
            "-map", "0:v:0", "-vf", ",".join(filters), "-q:v", "2",
            str(staging / "%06d.jpg"),
        ])
        names = [path.name for path in sorted(staging.glob("*.jpg"))]
        if not names:
            raise RuntimeError(f"FFmpeg extracted no frames from {video_path}")
        if output_dir.exists():
            output_dir.rename(backup)
        staging.rename(output_dir)
        swapped = True
    except Exception:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise
    finally:
        if swapped and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
    return [
        FrameEntry(index=i, timestamp=round(i / fps, 6), path=str(output_dir / name))
        for i, name in enumerate(names)
    ]


def config_hash(config: dict[str, Any]) -> str:
    """Stable short hash of a JSON-serializable configuration."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _source_fingerprint(path: Path) -> dict[str, Any]:
    """Return cheap source identity fields that invalidate stale derived media."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def cache_is_complete(manifest: dict[str, Any]) -> bool:
    """A matching manifest is reusable only while all recorded artifacts exist."""
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        return False
    if any(
        not isinstance(frame, dict)
        or not isinstance(frame.get("path"), str)
        or not Path(frame["path"]).is_file()
        for frame in frames
    ):
        return False
    audio_path = manifest.get("audio_path")
    return audio_path is None or (
        isinstance(audio_path, str) and Path(audio_path).is_file()
    )


def preprocess_video(
    video_path: str | Path,
    cache_dir: str | Path,
    fps: float = 2.0,
    sample_rate: int = 16_000,
    max_side: int | None = 768,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Cache audio + frames for one video and return its manifest.

    Re-running with the same settings reuses the cache. Videos without an
    audio stream still get frames; the manifest records audio_path = None so
    A/AV selectors can report the shortfall rather than silently skip.
    """
    video_path = Path(video_path)
    cache_dir = Path(cache_dir)
    settings = {
        "fps": fps,
        "sample_rate": sample_rate,
        "max_side": max_side,
        "manifest_version": MANIFEST_VERSION,
        "source_fingerprint": _source_fingerprint(video_path),
    }
    manifest_path = cache_dir / MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            manifest = {}
        if (
            manifest.get("config_hash") == config_hash(settings)
            and cache_is_complete(manifest)
        ):
            return manifest

    info = probe(video_path)
    if not info.has_video:
        raise ValueError(f"{video_path} has no video stream")
    frames = extract_frames(
        video_path, cache_dir / "frames", fps=fps, max_side=max_side,
    )
    if not frames:
        raise RuntimeError(f"FFmpeg extracted no frames from {video_path}")
    audio_path = None
    if info.has_audio:
        audio_path = str(extract_audio(video_path, cache_dir / "audio_16k.wav", sample_rate))
    else:
        (cache_dir / "audio_16k.wav").unlink(missing_ok=True)

    manifest = {
        "video_id": video_path.stem,
        "source": str(video_path.resolve()),
        "media_info": asdict(info),
        "settings": settings,
        "config_hash": config_hash(settings),
        "audio_path": audio_path,
        "frames": [asdict(frame) for frame in frames],
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest


def load_manifest(cache_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(cache_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))


def frame_window(
    manifest: dict[str, Any],
    center: float,
    num_frames: int = 4,
    span_seconds: float = 2.0,
) -> list[FrameEntry]:
    """Pick num_frames cached frames spread over [center - span/2, center + span/2].

    Each target time maps to the nearest cached frame; the window is clamped
    to the clip and duplicates are kept so every selector gets the same count.
    """
    frames = [FrameEntry(**f) for f in manifest["frames"]]
    if not frames:
        return []
    if num_frames == 1:
        targets = [center]
    else:
        step = span_seconds / (num_frames - 1)
        targets = [center - span_seconds / 2 + i * step for i in range(num_frames)]
    last = frames[-1].timestamp
    window = []
    for target in targets:
        target = min(max(target, 0.0), last)
        window.append(min(frames, key=lambda f: abs(f.timestamp - target)))
    return window


def load_audio(path: str | Path, sample_rate: int = 16_000):
    """Load cached audio as (float32 mono array, sample_rate) via librosa."""
    import librosa

    waveform, sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return waveform, sr


def load_frames(entries: Sequence[FrameEntry]):
    """Open cached frames as RGB PIL images, e.g. for VisualSelector."""
    from PIL import Image

    return [Image.open(entry.path).convert("RGB") for entry in entries]
