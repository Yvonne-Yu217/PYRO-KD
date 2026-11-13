"""Batch audio-feature caching: per-video failures and stale-feature invalidation.

The VAD/librosa step is replaced by a fake that stamps the real config hash, so
these run without torch/silero (macOS test env).
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.utils import audio_features
from src.utils.audio_features import (
    FEATURES_NAME, features_are_current, features_config_hash, load_features, write_features,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cache_audio_features  # noqa: E402


def fake_compute(audio_path, config=None, manifest_hash=None):
    if Path(audio_path).read_bytes().startswith(b"corrupt"):
        raise RuntimeError("could not decode WAV")
    return {
        "audio_path": str(audio_path),
        "duration": 10.0,
        "config": {**audio_features.DEFAULT_CONFIG, **(config or {})},
        "config_hash": features_config_hash(audio_path, config, manifest_hash),
        "speech": [[1.0, 2.0]],
        "speech_seconds": 1.0,
        "gaps": [],
    }


class CacheAudioFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = patch.object(
            cache_audio_features, "compute_audio_features", side_effect=fake_compute,
        )
        self.compute = patcher.start()
        self.addCleanup(patcher.stop)

    def add_video(self, video_id, wav=b"RIFF....", manifest_hash="m1", audio=True):
        video_dir = self.root / video_id
        video_dir.mkdir()
        wav_path = video_dir / "audio_16k.wav"
        if audio:
            wav_path.write_bytes(wav)
        manifest = {"config_hash": manifest_hash, "audio_path": str(wav_path) if audio else None}
        (video_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return video_dir

    def run_script(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cache_audio_features.main(["--cache-dir", str(self.root), *extra])
        report = json.loads((self.root / cache_audio_features.REPORT_NAME).read_text())
        return code, report, stdout.getvalue() + stderr.getvalue()

    def computed_ids(self):
        return sorted(Path(call.args[0]).parent.name for call in self.compute.call_args_list)

    # Finding 3: one bad video must not stop the batch.
    def test_corrupt_wav_is_reported_and_later_videos_still_cached(self):
        self.add_video("a")
        bad = self.add_video("b", wav=b"corrupt")
        (bad / FEATURES_NAME).write_text(json.dumps({"config_hash": "old", "speech": [], "gaps": []}))
        self.add_video("c")
        code, report, output = self.run_script()
        self.assertEqual(code, 1)
        self.assertEqual(report["computed"], 2)
        self.assertEqual([f["video_id"] for f in report["failed"]], ["b"])
        self.assertIn("could not decode WAV", report["failed"][0]["error"])
        self.assertIsNotNone(load_features(self.root / "a"))
        self.assertIsNotNone(load_features(self.root / "c"))
        # Stale features of the failed video are removed so A cannot reuse them.
        self.assertFalse((bad / FEATURES_NAME).exists())
        self.assertIn("FAILED 1 videos: b", output)

    def test_broken_manifest_is_a_failure_not_a_crash(self):
        broken = self.root / "broken"
        broken.mkdir()
        (broken / "manifest.json").write_text("{not json")
        self.add_video("good")
        code, report, _ = self.run_script()
        self.assertEqual(code, 1)
        self.assertEqual([f["video_id"] for f in report["failed"]], ["broken"])
        self.assertEqual(self.computed_ids(), ["good"])

    def test_missing_wav_is_a_failure(self):
        video = self.add_video("gone")
        Path(video / "audio_16k.wav").unlink()
        code, report, _ = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("FileNotFoundError", report["failed"][0]["error"])

    def test_clean_run_exits_zero(self):
        self.add_video("a")
        code, report, _ = self.run_script()
        self.assertEqual(code, 0)
        self.assertEqual(report, {"computed": 1, "reused": 0, "no_audio": [], "failed": []})

    # Finding 4: stale features are recomputed, current ones reused.
    def test_current_features_are_reused(self):
        self.add_video("a")
        self.run_script()
        self.compute.reset_mock()
        code, report, _ = self.run_script()
        self.assertEqual(code, 0)
        self.compute.assert_not_called()
        self.assertEqual((report["computed"], report["reused"]), (0, 1))

    def test_overwrite_forces_recompute(self):
        self.add_video("a")
        self.run_script()
        self.compute.reset_mock()
        self.run_script("--overwrite")
        self.assertEqual(self.computed_ids(), ["a"])

    def test_vad_config_change_invalidates_features(self):
        self.add_video("a")
        self.run_script()
        self.compute.reset_mock()
        with patch.dict(audio_features.DEFAULT_CONFIG, {"vad_threshold": 0.6}):
            _, report, _ = self.run_script()
        self.assertEqual(report["computed"], 1)

    def test_feature_version_bump_invalidates_features(self):
        self.add_video("a")
        self.run_script()
        self.compute.reset_mock()
        with patch.object(audio_features, "FEATURES_VERSION", audio_features.FEATURES_VERSION + 1):
            _, report, _ = self.run_script()
        self.assertEqual(report["computed"], 1)

    def test_reextracted_wav_invalidates_features(self):
        video = self.add_video("a")
        self.run_script()
        self.compute.reset_mock()
        wav = video / "audio_16k.wav"
        wav.write_bytes(b"RIFF....new audio")
        _, report, _ = self.run_script()
        self.assertEqual(report["computed"], 1)
        # Same size, newer mtime also counts as a new WAV.
        self.compute.reset_mock()
        wav.write_bytes(b"RIFF....NEW AUDIO")
        stat = wav.stat()
        os.utime(wav, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
        _, report, _ = self.run_script()
        self.assertEqual(report["computed"], 1)

    def test_manifest_hash_change_invalidates_features(self):
        video = self.add_video("a", manifest_hash="m1")
        self.run_script()
        self.compute.reset_mock()
        manifest = json.loads((video / "manifest.json").read_text())
        manifest["config_hash"] = "m2"
        (video / "manifest.json").write_text(json.dumps(manifest))
        _, report, _ = self.run_script()
        self.assertEqual(report["computed"], 1)

    def test_legacy_features_without_hash_are_recomputed(self):
        video = self.add_video("a")
        (video / FEATURES_NAME).write_text(json.dumps({"speech": [], "gaps": []}))
        _, report, _ = self.run_script()
        self.assertEqual(report["computed"], 1)
        self.assertIn("config_hash", load_features(video))

    def test_corrupt_features_json_is_recomputed(self):
        video = self.add_video("a")
        (video / FEATURES_NAME).write_text("{truncated")
        code, report, _ = self.run_script()
        self.assertEqual((code, report["computed"]), (0, 1))

    def test_no_audio_video_is_listed_and_stale_features_dropped(self):
        video = self.add_video("silent_film", audio=False)
        (video / FEATURES_NAME).write_text(json.dumps({"speech": [], "gaps": []}))
        code, report, _ = self.run_script()
        self.assertEqual(code, 0)
        self.assertEqual(report["no_audio"], ["silent_film"])
        self.assertFalse((video / FEATURES_NAME).exists())
        self.compute.assert_not_called()


class FeatureHashTests(unittest.TestCase):
    def test_hash_is_stable_and_sensitive_to_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp, "a.wav")
            wav.write_bytes(b"RIFF")
            base = features_config_hash(wav, None, "m1")
            self.assertEqual(base, features_config_hash(wav, None, "m1"))
            self.assertEqual(base, features_config_hash(wav, dict(audio_features.DEFAULT_CONFIG), "m1"))
            self.assertNotEqual(base, features_config_hash(wav, {"min_gap_seconds": 2.0}, "m1"))
            self.assertNotEqual(base, features_config_hash(wav, None, "m2"))

    def test_hash_requires_the_wav(self):
        with self.assertRaises(OSError):
            features_config_hash("/nonexistent/a.wav")

    def test_features_are_current(self):
        self.assertFalse(features_are_current(None, "abc"))
        self.assertFalse(features_are_current({}, "abc"))
        self.assertFalse(features_are_current({"config_hash": "old"}, "abc"))
        self.assertTrue(features_are_current({"config_hash": "abc"}, "abc"))

    def test_write_features_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_features(tmp, {"config_hash": "abc"})
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), [FEATURES_NAME])


if __name__ == "__main__":
    unittest.main()
