from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.utils.media import cache_is_complete, extract_frames, frame_window, preprocess_video, probe

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_test_video(path: Path, seconds: float = 3.0, audio: bool = True) -> None:
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=25:duration={seconds}",
    ]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    command += ["-c:a", "aac", "-ac", "2"] if audio else []
    command += ["-shortest", str(path)]
    subprocess.run(command, check=True)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not installed")
class MediaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.video = self.dir / "clip.mp4"
        make_test_video(self.video)

    def tearDown(self):
        self.tmp.cleanup()

    def test_probe(self):
        info = probe(self.video)
        self.assertTrue(info.has_video and info.has_audio)
        self.assertAlmostEqual(info.duration, 3.0, delta=0.15)
        self.assertEqual((info.width, info.height), (320, 240))
        self.assertAlmostEqual(info.video_fps, 25.0)

    def test_preprocess_writes_audio_frames_manifest_and_caches(self):
        cache = self.dir / "cache"
        manifest = preprocess_video(self.video, cache, fps=2.0, max_side=160)
        self.assertEqual(manifest["video_id"], "clip")
        self.assertEqual([f["timestamp"] for f in manifest["frames"]], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        audio = probe(manifest["audio_path"])
        self.assertEqual((audio.audio_sample_rate, audio.audio_channels), (16000, 1))
        self.assertFalse(audio.has_video)
        first_frame = probe(manifest["frames"][0]["path"])
        self.assertEqual((first_frame.width, first_frame.height), (160, 120))

        mtime = (cache / "manifest.json").stat().st_mtime_ns
        self.assertEqual(preprocess_video(self.video, cache, fps=2.0, max_side=160), manifest)
        self.assertEqual((cache / "manifest.json").stat().st_mtime_ns, mtime)
        changed = preprocess_video(self.video, cache, fps=1.0, max_side=160)
        self.assertEqual(len(changed["frames"]), 3)
        self.assertEqual(len(list((cache / "frames").glob("*.jpg"))), 3)

    def test_video_without_audio(self):
        silent = self.dir / "silent.mp4"
        make_test_video(silent, audio=False)
        manifest = preprocess_video(silent, self.dir / "silent_cache", fps=1.0)
        self.assertIsNone(manifest["audio_path"])
        self.assertEqual(len(manifest["frames"]), 3)

    def test_missing_cached_artifact_is_regenerated(self):
        cache = self.dir / "cache"
        manifest = preprocess_video(self.video, cache, fps=1.0)
        missing = Path(manifest["frames"][0]["path"])
        missing.unlink()
        repaired = preprocess_video(self.video, cache, fps=1.0)
        self.assertTrue(Path(repaired["frames"][0]["path"]).is_file())

    def test_malformed_manifest_is_rebuilt(self):
        cache = self.dir / "cache"
        cache.mkdir()
        (cache / "manifest.json").write_text("{broken")
        manifest = preprocess_video(self.video, cache, fps=1.0)
        self.assertEqual(len(manifest["frames"]), 3)
        self.assertEqual(manifest, preprocess_video(self.video, cache, fps=1.0))

    def test_changed_source_invalidates_cache(self):
        cache = self.dir / "cache"
        first = preprocess_video(self.video, cache, fps=1.0)
        make_test_video(self.video, seconds=2.0)
        second = preprocess_video(self.video, cache, fps=1.0)
        self.assertNotEqual(first["config_hash"], second["config_hash"])
        self.assertEqual(len(second["frames"]), 2)


class FrameExtractionTests(unittest.TestCase):
    def test_empty_artifacts_are_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = Path(tmp, "000001.jpg")
            audio = Path(tmp, "audio.wav")
            frame.touch()
            audio.touch()
            manifest = {"frames": [{"path": str(frame)}], "audio_path": str(audio)}
            self.assertFalse(cache_is_complete(manifest))

            frame.write_bytes(b"jpeg")
            self.assertFalse(cache_is_complete(manifest))

            audio.write_bytes(b"wav")
            self.assertTrue(cache_is_complete(manifest))

    def test_failed_extraction_keeps_existing_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp, "frames")
            frames.mkdir()
            kept = frames / "000001.jpg"
            kept.write_bytes(b"old")
            with (
                patch("src.utils.media._require", return_value="ffmpeg"),
                patch("src.utils.media._run", side_effect=RuntimeError("ffmpeg failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "ffmpeg failed"):
                    extract_frames(Path(tmp, "clip.mp4"), frames, fps=1.0)
            self.assertEqual(kept.read_bytes(), b"old")
            self.assertFalse((Path(tmp) / "frames.incoming").exists())
            self.assertFalse((Path(tmp) / "frames.backup").exists())

    def test_extraction_replaces_frames_only_after_success(self):
        def fake_run(command):
            Path(command[-1]).parent.joinpath("000001.jpg").write_bytes(b"new")
            return ""

        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp, "frames")
            frames.mkdir()
            (frames / "000009.jpg").write_bytes(b"old")
            with (
                patch("src.utils.media._require", return_value="ffmpeg"),
                patch("src.utils.media._run", side_effect=fake_run),
            ):
                entries = extract_frames(Path(tmp, "clip.mp4"), frames, fps=1.0)
            self.assertEqual([entry.timestamp for entry in entries], [0.0])
            self.assertEqual(Path(entries[0].path).read_bytes(), b"new")
            self.assertFalse((frames / "000009.jpg").exists())
            self.assertFalse((Path(tmp) / "frames.incoming").exists())
            self.assertFalse((Path(tmp) / "frames.backup").exists())


class FrameWindowTests(unittest.TestCase):
    MANIFEST = {"frames": [
        {"index": i, "timestamp": i * 0.5, "path": f"{i:06d}.jpg"} for i in range(11)
    ]}

    def test_window_spans_center(self):
        window = frame_window(self.MANIFEST, 2.0, num_frames=4, span_seconds=2.0)
        self.assertEqual([f.timestamp for f in window], [1.0, 1.5, 2.5, 3.0])

    def test_window_clamps_and_keeps_count(self):
        window = frame_window(self.MANIFEST, 0.0, num_frames=4, span_seconds=2.0)
        self.assertEqual(len(window), 4)
        self.assertEqual(window[0].timestamp, 0.0)
        self.assertEqual(frame_window(self.MANIFEST, 9.0, 1)[0].timestamp, 5.0)
        self.assertEqual(frame_window({"frames": []}, 1.0), [])


if __name__ == "__main__":
    unittest.main()
