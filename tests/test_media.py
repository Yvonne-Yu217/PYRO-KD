from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from src.utils.media import (
    FrameEntry, MediaInfo, cache_is_complete, cache_is_current, extract_frames, frame_window,
    load_manifest, preprocess_video, probe,
)

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
        self.assertTrue(cache_is_complete(manifest))
        self.assertEqual(load_manifest(cache), manifest)
        self.assertEqual(Path(manifest["audio_path"]), cache / "audio_16k.wav")
        self.assertTrue(all(Path(f["path"]).parent == cache / "frames" for f in manifest["frames"]))

        features = cache / "audio_features.json"
        features.write_bytes(b"existing features")
        mtime = (cache / "manifest.json").stat().st_mtime_ns
        self.assertEqual(preprocess_video(self.video, cache, fps=2.0, max_side=160), manifest)
        self.assertEqual((cache / "manifest.json").stat().st_mtime_ns, mtime)
        self.assertEqual(features.read_bytes(), b"existing features")
        changed = preprocess_video(self.video, cache, fps=1.0, max_side=160)
        self.assertEqual(len(changed["frames"]), 3)
        self.assertEqual(len(list((cache / "frames").glob("*.jpg"))), 3)
        self.assertFalse(features.exists())
        self.assertTrue(cache_is_complete(changed))
        self.assertEqual(load_manifest(cache), changed)
        self.assertEqual(list(self.dir.glob(".cache.incoming-*")), [])

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

    def test_real_audio_extraction_failure_preserves_complete_cache(self):
        cache = self.dir / "cache"
        manifest = preprocess_video(self.video, cache, fps=1.0)
        before = {p.relative_to(cache): p.read_bytes() for p in cache.rglob("*") if p.is_file()}
        self.assertTrue(cache_is_current(manifest))
        make_test_video(self.video, seconds=2.0)
        # FFmpeg extracts replacement frames, then rejects this audio rate.
        with self.assertRaisesRegex(RuntimeError, "command failed"):
            preprocess_video(self.video, cache, fps=2.0, sample_rate=-1)
        self.assertEqual(
            {p.relative_to(cache): p.read_bytes() for p in cache.rglob("*") if p.is_file()}, before,
        )
        self.assertEqual(load_manifest(cache), manifest)
        self.assertTrue(cache_is_complete(manifest))
        self.assertFalse(cache_is_current(manifest))
        self.assertEqual(list(self.dir.glob(".cache.incoming-*")), [])


class PreprocessingTransactionTests(unittest.TestCase):
    """Inject extraction/filesystem failures without requiring FFmpeg."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.video = self.root / "clip.mp4"
        self.video.write_bytes(b"source video")
        self.cache = self.root / "cache"
        self.content = b"old"
        self.info = MediaInfo(str(self.video), 2.0, 0.0, True, True, 160, 120, 25.0, 44100, 2)
        self.probe = self.enterContext(patch("src.utils.media.probe", return_value=self.info))
        self.frames = self.enterContext(patch("src.utils.media.extract_frames", side_effect=self.make_frames))
        self.audio = self.enterContext(patch("src.utils.media.extract_audio", side_effect=self.make_audio))
        self.manifest = preprocess_video(self.video, self.cache, fps=1.0)
        (self.cache / "audio_features.json").write_bytes(b"old derived features")
        self.before = self.snapshot(self.cache)
        self.content = b"new"
        for mock in (self.probe, self.frames, self.audio):
            mock.reset_mock()

    def make_frames(self, video_path, output_dir, fps, max_side):
        output_dir = Path(output_dir)
        output_dir.mkdir()
        frames = []
        for i in range(2):
            frame = output_dir / f"{i + 1:06d}.jpg"
            frame.write_bytes(self.content + b" frame")
            frames.append(FrameEntry(i, i / fps, str(frame)))
        return frames

    def make_audio(self, video_path, output_path, sample_rate):
        output_path = Path(output_path)
        output_path.write_bytes(self.content + b" audio")
        return output_path

    def snapshot(self, directory):
        return {
            p.relative_to(directory): (p.read_bytes(), p.stat().st_mtime_ns)
            for p in directory.rglob("*") if p.is_file()
        }

    def assert_previous_cache_preserved(self):
        self.assertEqual(self.snapshot(self.cache), self.before)
        self.assertTrue(cache_is_complete(load_manifest(self.cache)))
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["cache", "clip.mp4"])

    def test_successful_replacement_publishes_complete_final_paths(self):
        manifest = preprocess_video(self.video, self.cache, fps=2.0)
        self.assertEqual(load_manifest(self.cache), manifest)
        self.assertEqual(manifest.keys(), self.manifest.keys())
        self.assertTrue(cache_is_complete(manifest))
        self.assertEqual([f["timestamp"] for f in manifest["frames"]], [0.0, 0.5])
        self.assertEqual(Path(manifest["audio_path"]), self.cache / "audio_16k.wav")
        self.assertEqual(Path(manifest["audio_path"]).read_bytes(), b"new audio")
        for frame in manifest["frames"]:
            self.assertEqual(Path(frame["path"]).parent, self.cache / "frames")
            self.assertEqual(Path(frame["path"]).read_bytes(), b"new frame")
        self.assertFalse((self.cache / "audio_features.json").exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["cache", "clip.mp4"])

    def test_failed_frame_extraction_preserves_previous_cache(self):
        def fail_frames(*args, **kwargs):
            self.make_frames(*args, **kwargs)
            raise RuntimeError("frame extraction failed")

        self.frames.side_effect = fail_frames
        with self.assertRaisesRegex(RuntimeError, "frame extraction failed"):
            preprocess_video(self.video, self.cache, overwrite=True)
        self.audio.assert_not_called()
        self.assert_previous_cache_preserved()

    def test_failed_audio_extraction_preserves_previous_cache(self):
        def fail_audio(*args):
            self.make_audio(*args)
            raise RuntimeError("audio extraction failed")

        self.audio.side_effect = fail_audio
        with self.assertRaisesRegex(RuntimeError, "audio extraction failed"):
            preprocess_video(self.video, self.cache, overwrite=True)
        self.frames.assert_called_once()
        self.assert_previous_cache_preserved()

    def test_failed_manifest_write_preserves_previous_cache(self):
        with patch.object(Path, "write_text", side_effect=OSError("manifest write failed")):
            with self.assertRaisesRegex(OSError, "manifest write failed"):
                preprocess_video(self.video, self.cache, overwrite=True)
        self.frames.assert_called_once()
        self.audio.assert_called_once()
        self.assert_previous_cache_preserved()

    def test_publication_failure_rolls_back_previous_cache(self):
        original_replace = Path.replace

        def fail_publication(source, target):
            if source.name == "cache" and source.parent != self.root:
                self.assertEqual(self.snapshot(source.parent / "previous"), self.before)
                raise OSError("publication failed")
            return original_replace(source, target)

        with patch.object(Path, "replace", fail_publication):
            with self.assertRaisesRegex(OSError, "publication failed"):
                preprocess_video(self.video, self.cache, overwrite=True)
        self.assert_previous_cache_preserved()

    def test_failed_backup_rename_leaves_previous_cache_untouched(self):
        with patch.object(Path, "replace", side_effect=OSError("backup failed")):
            with self.assertRaisesRegex(OSError, "backup failed"):
                preprocess_video(self.video, self.cache, overwrite=True)
        self.assert_previous_cache_preserved()

    def test_failed_rollback_retains_recoverable_previous_cache(self):
        original_replace = Path.replace

        def fail_publication_and_rollback(source, target):
            if source.parent != self.root:
                raise OSError("rename failed")
            return original_replace(source, target)

        with patch.object(Path, "replace", fail_publication_and_rollback):
            with self.assertRaisesRegex(RuntimeError, "previous cache retained at"):
                preprocess_video(self.video, self.cache, overwrite=True)
        staging, = self.root.glob(".cache.incoming-*")
        self.assertEqual(self.snapshot(staging / "previous"), self.before)
        self.assertFalse((staging / "cache").exists())

    def test_unchanged_cache_reuses_all_artifacts_without_extraction(self):
        self.assertTrue(cache_is_current(self.manifest))
        self.assertEqual(preprocess_video(self.video, self.cache, fps=1.0), self.manifest)
        self.probe.assert_not_called()
        self.frames.assert_not_called()
        self.audio.assert_not_called()
        self.assert_previous_cache_preserved()

    def test_complete_cache_can_be_stale_for_changed_source(self):
        self.video.write_bytes(b"replacement source video")
        self.assertTrue(cache_is_complete(self.manifest))
        self.assertFalse(cache_is_current(self.manifest))
        self.assert_previous_cache_preserved()

    def test_missing_or_invalid_source_metadata_is_not_current(self):
        for fields in (
            {"source": None}, {"source": ""}, {"settings": None}, {"settings": []},
            {"settings": {}}, {"settings": {"source_fingerprint": "invalid"}},
            {"settings": {"source_fingerprint": {}}},
        ):
            with self.subTest(fields=fields):
                self.assertFalse(cache_is_current({**self.manifest, **fields}))
        without_source = {k: v for k, v in self.manifest.items() if k != "source"}
        self.assertFalse(cache_is_current(without_source))

    def test_unavailable_source_is_not_current(self):
        self.video.unlink()
        self.assertTrue(cache_is_complete(self.manifest))
        self.assertFalse(cache_is_current(self.manifest))

    def test_changed_source_replaces_cache_and_discards_derived_features(self):
        self.video.write_bytes(b"replacement source video")
        manifest = preprocess_video(self.video, self.cache, fps=1.0)
        self.assertNotEqual(manifest["config_hash"], self.manifest["config_hash"])
        self.assertTrue(cache_is_complete(manifest))
        self.assertFalse((self.cache / "audio_features.json").exists())
        self.frames.assert_called_once()
        self.audio.assert_called_once()

    def test_missing_cached_frame_triggers_complete_rebuild(self):
        Path(self.manifest["frames"][0]["path"]).unlink()
        manifest = preprocess_video(self.video, self.cache, fps=1.0)
        self.assertTrue(cache_is_complete(manifest))
        self.assertEqual(Path(manifest["frames"][0]["path"]).read_bytes(), b"new frame")
        self.assertFalse((self.cache / "audio_features.json").exists())
        self.frames.assert_called_once()
        self.audio.assert_called_once()

    def test_incomplete_staged_audio_does_not_replace_previous_cache(self):
        def empty_audio(*args):
            path = self.make_audio(*args)
            path.write_bytes(b"")
            return path

        self.audio.side_effect = empty_audio
        with self.assertRaisesRegex(RuntimeError, "Incomplete extracted media cache"):
            preprocess_video(self.video, self.cache, overwrite=True)
        self.assert_previous_cache_preserved()

    def test_replacement_without_audio_drops_old_audio_and_features(self):
        self.probe.return_value = replace(self.info, has_audio=False)
        manifest = preprocess_video(self.video, self.cache, overwrite=True)
        self.assertIsNone(manifest["audio_path"])
        self.assertTrue(cache_is_complete(manifest))
        self.assertEqual(load_manifest(self.cache), manifest)
        self.assertFalse((self.cache / "audio_16k.wav").exists())
        self.assertFalse((self.cache / "audio_features.json").exists())
        self.audio.assert_not_called()


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
