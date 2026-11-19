"""Audio-feature caching: per-video failures, staleness, and downstream checks.

The VAD/librosa step is replaced by a fake that stamps the real config hash, so
these run without torch/silero (macOS test env).
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.utils import audio_features
from src.utils.audio_features import (
    FEATURES_NAME, compute_audio_features, features_are_current, features_config_hash,
    load_current_features, load_features, resolve_audio_path, write_features,
)
from src.utils.media import FrameEntry, MediaInfo, config_hash, preprocess_video

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_moments  # noqa: E402
import cache_audio_features  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def fake_compute(audio_path, config=None, manifest_hash=None):
    if Path(audio_path).read_bytes().startswith(b"corrupt"):
        raise RuntimeError("could not decode WAV")
    return {
        "audio_path": str(audio_path),
        "duration": 60.0,
        "config": {**audio_features.DEFAULT_CONFIG, **(config or {})},
        "config_hash": features_config_hash(audio_path, config),
        "speech": [[1.0, 2.0]],
        "speech_seconds": 1.0,
        "gaps": [{"start": 2.0, "end": 60.0, "rms": 0.0, "onset": 0.0}],
    }


def quiet(call, *args, **kwargs):
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = call(*args, **kwargs)
    return result, stdout.getvalue() + stderr.getvalue()


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

    def add_video(self, video_id, wav=b"RIFF....", manifest_hash="m1", audio=True, audio_path=None):
        video_dir = self.root / video_id
        video_dir.mkdir()
        wav_path = video_dir / "audio_16k.wav"
        if audio:
            wav_path.write_bytes(wav)
        recorded = audio_path or str(wav_path)
        manifest = {"config_hash": manifest_hash, "audio_path": recorded if audio else None}
        (video_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return video_dir

    def run_script(self, *extra):
        code, output = quiet(cache_audio_features.main, ["--cache-dir", str(self.root), *extra])
        report_path = self.root / cache_audio_features.REPORT_NAME
        report = json.loads(report_path.read_text()) if report_path.exists() else None
        return code, report, output

    def computed_ids(self):
        return sorted(Path(call.args[0]).parent.name for call in self.compute.call_args_list)

    # One bad video must not stop the batch, and must not destroy anything.
    def test_corrupt_wav_is_reported_and_later_videos_still_cached(self):
        self.add_video("a")
        bad = self.add_video("b", wav=b"corrupt")
        old = {"config_hash": "old", "speech": [], "gaps": []}
        (bad / FEATURES_NAME).write_text(json.dumps(old))
        self.add_video("c")
        code, report, output = self.run_script()
        self.assertEqual(code, 1)
        self.assertEqual(report["computed"], 2)
        self.assertEqual([f["video_id"] for f in report["failed"]], ["b"])
        self.assertIn("could not decode WAV", report["failed"][0]["error"])
        self.assertIsNotNone(load_features(self.root / "c"))
        # Nothing is deleted; the old file is simply not current for this WAV.
        self.assertEqual(load_features(bad), old)
        self.assertEqual(load_current_features(bad, {"audio_path": str(bad / "audio_16k.wav")}),
                         (None, "stale_audio_features"))
        self.assertIn("FAILED 1 videos: b", output)

    def test_environment_error_keeps_current_features(self):
        """Review item 1: a missing dependency must not wipe a valid cache."""
        for video_id in ("a", "b"):
            self.add_video(video_id)
        self.run_script()
        before = {v: load_features(self.root / v) for v in ("a", "b")}
        self.compute.side_effect = ImportError("No module named 'silero_vad'")
        code, report, output = self.run_script("--overwrite")
        self.assertEqual(code, 2)
        self.assertIn("silero_vad", report["aborted"])
        self.assertEqual(self.compute.call_count, 3)  # 2 first run + stopped at the first video
        self.assertEqual({v: load_features(self.root / v) for v in ("a", "b")}, before)
        self.assertIn("missing dependency", output)

    def test_foreign_cwd_wav_is_never_used_even_when_it_exists(self):
        """Kiran (#16 review): a relative path must not pick another video's WAV."""
        video = self.add_video("a", wav=b"RIFF owning A",
                               audio_path="data/cache/mtg-ad/a/audio_16k.wav")
        foreign = self.root.parent / f"{self.root.name}-foreign"
        decoy = foreign / "data/cache/mtg-ad/a/audio_16k.wav"
        decoy.parent.mkdir(parents=True)
        decoy.write_bytes(b"RIFF another video B")
        self.addCleanup(lambda: __import__("shutil").rmtree(foreign, ignore_errors=True))
        cwd = os.getcwd()
        os.chdir(foreign)
        self.addCleanup(os.chdir, cwd)
        self.assertTrue(Path("data/cache/mtg-ad/a/audio_16k.wav").exists())
        manifest = json.loads((video / "manifest.json").read_text())
        self.assertEqual(resolve_audio_path(manifest, video), video / "audio_16k.wav")
        code, _, _ = self.run_script("--overwrite")
        self.assertEqual(code, 0)
        self.assertEqual(Path(self.compute.call_args.args[0]), video / "audio_16k.wav")
        features, reason = load_current_features(video, manifest)
        self.assertIsNone(reason)
        self.assertEqual(features["config_hash"], features_config_hash(video / "audio_16k.wav"))
        self.assertNotEqual(features["config_hash"], features_config_hash(decoy))

    def test_absolute_custom_path_is_used_only_without_a_local_wav(self):
        custom = self.root / "elsewhere.wav"
        custom.write_bytes(b"RIFF custom")
        video = self.add_video("a", audio=False)
        manifest = {"audio_path": str(custom)}
        self.assertEqual(resolve_audio_path(manifest, video), custom)
        (video / "elsewhere.wav").write_bytes(b"RIFF local")
        self.assertEqual(resolve_audio_path(manifest, video), video / "elsewhere.wav")
        self.assertEqual(resolve_audio_path({"audio_path": "rel/missing.wav"}, video),
                         video / "missing.wav")

    def test_malformed_but_stamped_features_are_recomputed(self):
        video = self.add_video("a")
        wav = video / "audio_16k.wav"
        write_features(video, {"config_hash": features_config_hash(wav), "speech": [[1.0]], "gaps": []})
        code, report, _ = self.run_script()
        self.assertEqual((code, report["computed"], report["reused"]), (0, 1, 0))

    def test_relative_audio_path_from_another_directory_still_resolves(self):
        """Review item 1: manifests may record audio_path relative to another cwd."""
        video = self.add_video("a", audio_path="data/cache/mtg-ad/a/audio_16k.wav")
        code, report, _ = self.run_script()
        self.assertEqual((code, report["computed"]), (0, 1))
        manifest = json.loads((video / "manifest.json").read_text())
        self.assertEqual(resolve_audio_path(manifest, video), video / "audio_16k.wav")
        self.assertIsNone(load_current_features(video, manifest)[1])

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

    def test_missing_or_empty_cache_dir_fails(self):
        """Review item 4: a misspelled --cache-dir must not look like success."""
        code, output = quiet(cache_audio_features.main, ["--cache-dir", str(self.root / "typo")])
        self.assertEqual(code, 1)
        self.assertIn("no */manifest.json", output)
        code, _ = quiet(cache_audio_features.main, ["--cache-dir", str(self.root)])
        self.assertEqual(code, 1)

    def test_clean_run_exits_zero(self):
        self.add_video("a")
        code, report, _ = self.run_script()
        self.assertEqual(code, 0)
        self.assertEqual(report, {"computed": 1, "reused": 0, "no_audio": [], "failed": [],
                                  "aborted": None})

    # Staleness: recompute what changed, reuse what did not.
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

    def test_changed_wav_contents_invalidate_features(self):
        video = self.add_video("a")
        self.run_script()
        self.compute.reset_mock()
        (video / "audio_16k.wav").write_bytes(b"RIFF....NEW!")  # same size, new contents
        _, report, _ = self.run_script()
        self.assertEqual(report["computed"], 1)

    def test_copy_without_mtimes_and_frame_only_changes_keep_features(self):
        """Review item 6: neither a copied cache nor new frame settings forces VAD again."""
        video = self.add_video("a", manifest_hash="m1")
        self.run_script()
        self.compute.reset_mock()
        wav = video / "audio_16k.wav"
        stat = wav.stat()
        os.utime(wav, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**10))
        manifest = json.loads((video / "manifest.json").read_text())
        manifest["config_hash"] = "m2"  # e.g. fps or max_side changed
        (video / "manifest.json").write_text(json.dumps(manifest))
        _, report, _ = self.run_script()
        self.assertEqual((report["computed"], report["reused"]), (0, 1))

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

    def test_no_audio_video_is_listed(self):
        self.add_video("silent_film", audio=False)
        code, report, _ = self.run_script()
        self.assertEqual(code, 0)
        self.assertEqual(report["no_audio"], ["silent_film"])
        self.compute.assert_not_called()


class FeatureHashAndLoaderTests(unittest.TestCase):
    def test_hash_is_content_based_and_sensitive_to_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp, "a.wav")
            wav.write_bytes(b"RIFF")
            base = features_config_hash(wav)
            self.assertEqual(base, features_config_hash(wav, dict(audio_features.DEFAULT_CONFIG)))
            # manifest_hash is accepted for callers such as cache_checks (#18) but ignored.
            self.assertEqual(base, features_config_hash(wav, None, "any manifest hash"))
            self.assertNotEqual(base, features_config_hash(wav, {"min_gap_seconds": 2.0}))
            copy = Path(tmp, "copy.wav")
            shutil.copy(wav, copy)
            self.assertEqual(base, features_config_hash(copy))
            wav.write_bytes(b"RIFX")
            self.assertNotEqual(base, features_config_hash(wav))

    def test_hash_reuses_media_config_hash_convention(self):
        """Review item 8: one cache-key convention."""
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp, "a.wav")
            wav.write_bytes(b"RIFF")
            expected = config_hash({
                "features_version": audio_features.FEATURES_VERSION,
                "config": dict(audio_features.DEFAULT_CONFIG),
                "sample_rate": audio_features.SAMPLE_RATE,
                "hop_length": audio_features.HOP_LENGTH,
                "audio_sha256": audio_features.wav_digest(wav),
            })
            self.assertEqual(features_config_hash(wav), expected)

    def test_hash_requires_the_wav(self):
        with self.assertRaises(OSError):
            features_config_hash("/nonexistent/a.wav")

    def run_compute(self, wav, load_audio):
        with (patch("src.utils.media.load_audio", side_effect=load_audio),
              patch.object(audio_features, "detect_speech", return_value=[(0.0, 1.0)]),
              patch.object(audio_features, "gap_features", return_value=[])):
            return compute_audio_features(wav)

    def test_a_b_a_replacement_cannot_mislabel_features(self):
        """Kiran (#16 review): the digest must describe exactly the bytes decoded."""
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp, "audio_16k.wav")
            wav.write_bytes(b"RIFF A silent")
            decoded = []

            def swap_b_then_restore_a(path, sample_rate):
                wav.write_bytes(b"RIFF B speech")       # concurrent publisher
                decoded.append(Path(path).read_bytes())  # what the decoder sees
                wav.write_bytes(b"RIFF A silent")       # restored before any re-hash
                return [0.0] * 32, sample_rate

            features = self.run_compute(wav, swap_b_then_restore_a)
            self.assertEqual(decoded, [b"RIFF A silent"])  # decoder never saw B
            self.assertEqual(features["config_hash"], features_config_hash(wav))

    def test_stamp_matches_decoded_bytes_when_wav_changes_for_good(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp, "audio_16k.wav")
            wav.write_bytes(b"RIFF A")
            seen = {}

            def replace_permanently(path, sample_rate):
                seen["digest"] = audio_features.wav_digest(path)
                wav.write_bytes(b"RIFF B, re-extracted")
                return [0.0] * 32, sample_rate

            features = self.run_compute(wav, replace_permanently)
            self.assertEqual(features["config_hash"], audio_features._settings_hash(seen["digest"], None))
            # The features describe A, so they are (correctly) stale for B.
            self.assertNotEqual(features["config_hash"], features_config_hash(wav))

    def test_snapshot_is_removed_even_when_decoding_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp, "audio_16k.wav")
            wav.write_bytes(b"RIFF")
            paths = []

            def fail(path, sample_rate):
                paths.append(Path(path))
                raise RuntimeError("decode failed")

            with self.assertRaisesRegex(RuntimeError, "decode failed"):
                self.run_compute(wav, fail)
            self.assertNotEqual(paths[0], wav)
            self.assertFalse(paths[0].exists())

    def test_features_are_current(self):
        self.assertFalse(features_are_current(None, "abc"))
        self.assertFalse(features_are_current({}, "abc"))
        self.assertFalse(features_are_current({"config_hash": "old"}, "abc"))
        self.assertTrue(features_are_current({"config_hash": "abc"}, "abc"))

    def test_load_features_never_raises(self):
        """Review item 3: a corrupt file reads as missing."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_features(tmp))
            Path(tmp, FEATURES_NAME).write_text("{truncated")
            self.assertIsNone(load_features(tmp))
            Path(tmp, FEATURES_NAME).write_bytes(b"\xff\xfe")
            self.assertIsNone(load_features(tmp))
            Path(tmp, FEATURES_NAME).write_text("[1, 2]")
            self.assertIsNone(load_features(tmp))

    def test_load_current_features_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            wav = cache / "audio_16k.wav"
            manifest = {"audio_path": str(wav)}
            self.assertEqual(load_current_features(cache, {"audio_path": None}), (None, "no_audio_stream"))
            wav.write_bytes(b"RIFF")
            self.assertEqual(load_current_features(cache, manifest), (None, "no_audio_features"))
            (cache / FEATURES_NAME).write_text("{bad")
            self.assertEqual(load_current_features(cache, manifest), (None, "unreadable_audio_features"))
            write_features(cache, {"config_hash": features_config_hash(wav), "speech": [[1.0]], "gaps": []})
            self.assertEqual(load_current_features(cache, manifest), (None, "malformed_audio_features"))
            write_features(cache, {"config_hash": "old", "speech": [], "gaps": []})
            self.assertEqual(load_current_features(cache, manifest), (None, "stale_audio_features"))
            current = {"config_hash": features_config_hash(wav), "speech": [], "gaps": []}
            write_features(cache, current)
            self.assertEqual(load_current_features(cache, manifest), (current, None))
            wav.unlink()
            self.assertEqual(load_current_features(cache, manifest), (None, "missing_wav"))

    def test_concurrent_writers_never_share_a_staging_file(self):
        """Kiran (#16 review): B publishes while A still holds its staging file.

        With one shared .tmp name, B rewrites and publishes A's staging inode, so
        A can then modify (or fail to publish) a file B already published.
        Each writer must stage privately; the last rename wins with a complete record.
        """
        with tempfile.TemporaryDirectory() as tmp:
            record_a = {"config_hash": "a", "speech": [[1.0, 2.0]], "gaps": []}
            record_b = {"config_hash": "b", "speech": [[float(i), i + 0.5] for i in range(50)],
                        "gaps": []}
            real_replace = Path.replace
            staged = []

            def replace(self_path, target):
                staged.append(self_path)
                if len(staged) == 1:  # A is about to publish: let B run start to finish
                    write_features(tmp, record_b)
                    self.assertEqual(load_features(tmp), record_b)
                    self.assertTrue(self_path.exists(), "B consumed A's staging file")
                return real_replace(self_path, target)

            with patch.object(Path, "replace", replace):
                write_features(tmp, record_a)
            self.assertNotEqual(staged[0], staged[1])
            self.assertEqual(load_features(tmp), record_a)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), [FEATURES_NAME])

    def test_failed_writer_leaves_the_published_record_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = {"config_hash": "good", "speech": [], "gaps": []}
            write_features(tmp, good)
            with (patch.object(Path, "replace", side_effect=OSError("disk full")),
                  self.assertRaises(OSError)):
                write_features(tmp, {"config_hash": "other", "speech": [], "gaps": []})
            self.assertEqual(load_features(tmp), good)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), [FEATURES_NAME])

    def test_malformed_features_validator(self):
        ok = {"speech": [[0.0, 1.0]], "gaps": [{"start": 1.0, "end": 2.0, "rms": 0.0, "onset": 0.0}]}
        self.assertIsNone(audio_features.malformed_features(ok))
        bad = [
            [], {"speech": None, "gaps": []}, {"speech": [], "gaps": None},
            {"speech": [[1.0]], "gaps": []}, {"speech": [[2.0, 1.0]], "gaps": []},
            {"speech": [[float("nan"), 1.0]], "gaps": []}, {"speech": [[True, 1.0]], "gaps": []},
            {"speech": [[0, 10**400]], "gaps": []}, {"speech": [], "gaps": [{}]},
            {"speech": [], "gaps": [{**ok["gaps"][0], "extra": 1}]},
            {"speech": [], "gaps": [{**ok["gaps"][0], "rms": "loud"}]},
            {"speech": [], "gaps": [{**ok["gaps"][0], "start": 3.0}]},
        ]
        for payload in bad:
            with self.subTest(payload=payload):
                self.assertIsNotNone(audio_features.malformed_features(payload))

    def test_write_features_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_features(tmp, {"config_hash": "abc"})
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), [FEATURES_NAME])


class BuildMomentsStalenessTests(unittest.TestCase):
    """Review items 2 and 7: build_moments itself refuses stale or broken features."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache_root = self.root / "cache"
        self.splits = self.root / "splits.json"
        self.splits.write_text(json.dumps({"k_rule": {"rate_per_minute": 3}}))

        def fake_frames(video_path, output_dir, **kwargs):
            output_dir.mkdir()
            frame = output_dir / "000001.jpg"
            frame.write_bytes(b"frame")
            return [FrameEntry(0, 0.0, str(frame))]

        def fake_audio(video_path, output_path, sample_rate):
            output_path.write_bytes(video_path.read_bytes() + b" audio")
            return output_path

        for video_id in ("fresh", "changed"):
            source = self.root / f"{video_id}.mp4"
            source.write_bytes(f"{video_id} source".encode())
            info = MediaInfo(str(source), 60.0, 0.0, True, True, 160, 120, 25.0, 44100, 2)
            with (patch("src.utils.media.probe", return_value=info),
                  patch("src.utils.media.extract_frames", side_effect=fake_frames),
                  patch("src.utils.media.extract_audio", side_effect=fake_audio)):
                preprocess_video(source, self.cache_root / video_id)
            write_features(self.cache_root / video_id,
                           fake_compute(self.cache_root / video_id / "audio_16k.wav"))

    def build(self, selector):
        output = self.root / f"{selector}.jsonl"
        with (patch.object(sys, "argv", ["build_moments.py", "--selector", selector,
                                         "--splits", str(self.splits),
                                         "--cache-dir", str(self.cache_root), "--output", str(output)]),
              patch.object(build_moments, "load_targets",
                           return_value=[("changed", 60, []), ("fresh", 60, [])])):
            _, stdout = quiet(build_moments.main)
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        report = json.loads(output.with_suffix(".report.json").read_text())
        return rows, {v["video_id"]: v for v in report["videos"]}, stdout

    def test_stale_features_give_no_a_moments_and_are_named(self):
        # WAV re-extracted (e.g. a VAD-relevant change) without rerunning the cache script.
        (self.cache_root / "changed" / "audio_16k.wav").write_bytes(b"re-extracted audio")
        rows, report, stdout = self.build("A")
        self.assertEqual({r["video_id"] for r in rows}, {"fresh"})
        self.assertEqual(report["changed"]["status"], "missing_audio_features")
        self.assertEqual(report["fresh"]["status"], "ok")
        self.assertIn("changed (stale_audio_features)", stdout)

    def test_stale_features_do_not_set_u_word_limits(self):
        (self.cache_root / "changed" / "audio_16k.wav").write_bytes(b"re-extracted audio")
        rows, _, _ = self.build("U")
        limits = {r["video_id"]: r["word_limit"] for r in rows}
        self.assertIsNone(limits["changed"])  # unknown gap, not stale speech
        self.assertIsNotNone(limits["fresh"])

    def test_vad_config_change_without_recaching_is_caught(self):
        with patch.dict(audio_features.DEFAULT_CONFIG, {"min_gap_seconds": 3.0}):
            rows, _, stdout = self.build("A")
        self.assertEqual(rows, [])
        self.assertIn("stale_audio_features", stdout)

    def test_stamped_malformed_payloads_do_not_abort_u_or_a_batches(self):
        wav = self.cache_root / "changed" / "audio_16k.wav"
        payloads = {
            "short speech pair": {"speech": [[1.0]], "gaps": []},
            "empty gap record": {"speech": [], "gaps": [{}]},
            "null speech": {"speech": None, "gaps": []},
            "null gaps": {"speech": [], "gaps": None},
        }
        for label, payload in payloads.items():
            write_features(self.cache_root / "changed",
                           {**payload, "config_hash": features_config_hash(wav)})
            for selector in ("U", "A"):
                with self.subTest(payload=label, selector=selector):
                    rows, report, stdout = self.build(selector)
                    self.assertIn("fresh", {r["video_id"] for r in rows})
                    self.assertEqual(report["fresh"]["status"], "ok")
                    self.assertIn("changed (malformed_audio_features)", stdout)
                    if selector == "A":
                        self.assertNotIn("changed", {r["video_id"] for r in rows})
                    else:
                        self.assertTrue(all(r["word_limit"] is None
                                            for r in rows if r["video_id"] == "changed"))

    def test_corrupt_features_file_does_not_abort_batch(self):
        (self.cache_root / "changed" / FEATURES_NAME).write_text("{truncated")
        rows, report, stdout = self.build("A")
        self.assertEqual({r["video_id"] for r in rows}, {"fresh"})
        self.assertIn("changed (unreadable_audio_features)", stdout)

    def test_failed_recache_leaves_build_moments_safe(self):
        """End to end: corrupt WAV -> cache script fails it -> A reports it missing."""
        (self.cache_root / "changed" / "audio_16k.wav").write_bytes(b"corrupt data")
        with patch.object(cache_audio_features, "compute_audio_features", side_effect=fake_compute):
            code, _ = quiet(cache_audio_features.main, ["--cache-dir", str(self.cache_root)])
        self.assertEqual(code, 1)
        rows, report, _ = self.build("A")
        self.assertEqual({r["video_id"] for r in rows}, {"fresh"})
        self.assertEqual(report["changed"]["status"], "missing_audio_features")


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class PrepareDataExitCodeTests(unittest.TestCase):
    """Review item 7: prepare_data.sh's exit logic under set -euo pipefail.

    Runs the real script against a stub hpc/config.sh whose in_env fails the
    preprocessing and/or audio step on demand and records every command.
    """

    def run_prepare(self, prep_status, audio_status):
        with tempfile.TemporaryDirectory() as tmp:
            hpc = Path(tmp, "hpc")
            hpc.mkdir()
            shutil.copy(REPO / "hpc" / "prepare_data.sh", hpc / "prepare_data.sh")
            (hpc / "config.sh").write_text(
                'in_env() { shift; echo "RUN $*" >> "$LOG";\n'
                f'  case "$*" in *prepare_mtg_ad*) return {prep_status};;'
                f' *cache_audio_features*) return {audio_status};; esac; }}\n'
            )
            log = Path(tmp, "log")
            result = subprocess.run(
                ["bash", str(hpc / "prepare_data.sh")], capture_output=True, text=True,
                env={**os.environ, "LOG": str(log)}, timeout=30,
            )
            return result.returncode, log.read_text().splitlines(), result.stderr

    def test_exit_codes_and_moments_always_built(self):
        cases = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (3, 0): 3, (3, 1): 3}
        for (prep, audio), expected in cases.items():
            with self.subTest(prep=prep, audio=audio):
                code, commands, stderr = self.run_prepare(prep, audio)
                self.assertEqual(code, expected)
                builds = [c for c in commands if "build_moments.py" in c]
                self.assertEqual(len(builds), 2)
                self.assertTrue(any("--selector U" in c for c in builds))
                self.assertTrue(any("--selector A" in c for c in builds))
                if audio:
                    self.assertIn("audio_features_report.json", stderr)


if __name__ == "__main__":
    unittest.main()
