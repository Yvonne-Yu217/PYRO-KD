import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.utils import visual_features as vf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_moments  # noqa: E402
import cache_fixtures as fx  # noqa: E402
import cache_visual_features  # noqa: E402

try:
    import scenedetect  # noqa: F401
    HAS_SCENEDETECT = True
except ImportError:
    HAS_SCENEDETECT = False
HAS_FFMPEG = shutil.which("ffmpeg") is not None


def make_features(times, novelty, shots=()):
    return {"frame_times": list(times), "novelty": list(novelty), "shots": list(shots)}


def spike_features(duration=60.0, spikes=((10.0, 1.0), (30.0, 0.9), (50.0, 0.8)), shots=()):
    times = [i * 0.5 for i in range(int(duration / 0.5))]
    novelty = [0.05] * len(times)
    for t, value in spikes:
        novelty[times.index(t)] = value
    return make_features(times, novelty, shots)


class NoveltyTests(unittest.TestCase):
    def test_novelty_is_one_minus_cosine_to_previous_frame(self):
        curve = vf.novelty_curve([[1, 0], [1, 0], [0, 1], [0, 2]])
        self.assertEqual(curve[:2], [0.0, 0.0])
        self.assertAlmostEqual(curve[2], 1.0)
        self.assertAlmostEqual(curve[3], 0.0)

    def test_edge_cases(self):
        self.assertEqual(vf.novelty_curve(np.zeros((0, 4))), [])
        self.assertEqual(vf.novelty_curve([[0, 0], [0, 0]]), [0.0, 1.0])  # zero vectors do not crash
        with self.assertRaises(ValueError):
            vf.novelty_curve([1, 2, 3])


class SelectorTests(unittest.TestCase):
    def test_picks_novelty_peaks_with_minimum_separation(self):
        features = spike_features(spikes=((10.0, 1.0), (10.5, 0.95), (30.0, 0.9)))
        chosen = vf.VisualCueSelector().select(features, 2)
        self.assertEqual([c.time for c in chosen], [10.0, 30.0])  # 10.5 is within 2 s of 10.0
        self.assertTrue(all(b.time - a.time >= 2.0 for a, b in zip(chosen, chosen[1:])))

    def test_shot_boundary_bonus_goes_to_the_nearest_frame_and_can_reorder(self):
        features = spike_features(spikes=((10.0, 1.0), (30.0, 0.4)), shots=[30.2])
        candidates = {c.time: c for c in vf.VisualCueSelector().candidates(features)}
        self.assertTrue(candidates[30.0].shot_boundary)
        self.assertFalse(candidates[30.5].shot_boundary)
        self.assertAlmostEqual(candidates[30.0].score, 0.4 + 0.5)
        only_one = vf.VisualCueSelector(shot_weight=0.8).select(features, 1)
        self.assertEqual(only_one[0].time, 30.0)  # 0.4 + 0.8 beats the 1.0 spike
        no_bonus = vf.VisualCueSelector(shot_weight=0.0).select(features, 1)
        self.assertEqual(no_bonus[0].time, 10.0)

    def test_static_video_ties_go_to_earlier_times(self):
        features = make_features([i * 0.5 for i in range(40)], [0.0] * 40)
        self.assertEqual([c.time for c in vf.VisualCueSelector().select(features, 3)], [0.0, 2.0, 4.0])

    def test_returns_fewer_than_k_when_separation_leaves_no_room(self):
        features = make_features([0.0, 0.5, 1.0, 1.5], [0.1, 0.2, 0.3, 0.4])
        chosen = vf.VisualCueSelector().select(features, 5)
        self.assertLess(len(chosen), 5)
        self.assertEqual(len(chosen), 1)

    def test_empty_features_and_deterministic_output(self):
        self.assertEqual(vf.VisualCueSelector().select(make_features([], []), 3), [])
        features = spike_features()
        first = vf.VisualCueSelector().select(features, 3)
        self.assertEqual(first, vf.VisualCueSelector().select(features, 3))


class FeatureCacheTests(unittest.TestCase):
    MANIFEST = {
        "source": "video.mp4", "config_hash": "abc123",
        "frames": [{"index": i, "timestamp": i * 0.5, "path": f"{i}.jpg"} for i in range(4)],
    }

    @staticmethod
    def fake_embed(paths):
        return [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]][: len(paths)]

    @staticmethod
    def fake_detect(path, threshold, min_scene):
        return [1.004, 2.0]

    def compute(self, manifest=None, config=None):
        return vf.compute_visual_features(
            manifest or self.MANIFEST, self.fake_embed, self.fake_detect, config)

    def test_compute_records_times_novelty_shots_and_hash(self):
        features, embeddings = self.compute()
        self.assertEqual(features["frame_times"], [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(features["shots"], [1.004, 2.0])
        self.assertAlmostEqual(features["novelty"][2], 1.0)
        self.assertEqual(features["num_frames"], 4)
        self.assertEqual(embeddings.shape, (4, 2))
        self.assertTrue(vf.is_fresh(features, self.MANIFEST))

    def test_staleness_follows_manifest_and_config(self):
        features, _ = self.compute()
        self.assertFalse(vf.is_fresh(features, {**self.MANIFEST, "config_hash": "other"}))
        self.assertFalse(vf.is_fresh(features, self.MANIFEST, {"shot_threshold": 20.0}))
        self.assertFalse(vf.is_fresh(None, self.MANIFEST))

    def test_rejects_wrong_embedding_count_and_empty_manifest(self):
        with self.assertRaises(ValueError):
            vf.compute_visual_features(self.MANIFEST, lambda paths: [[1.0, 0.0]], self.fake_detect)
        with self.assertRaises(ValueError):
            vf.compute_visual_features({"frames": [], "source": "v.mp4"}, self.fake_embed, self.fake_detect)

    def test_round_trip_and_unreadable_cache_is_missing(self):
        features, embeddings = self.compute()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(vf.load_features(tmp))
            self.assertIsNone(vf.load_embeddings(tmp))
            vf.write_features(tmp, features, embeddings)
            self.assertEqual(vf.load_features(tmp), features)
            np.testing.assert_allclose(vf.load_embeddings(tmp), embeddings, atol=1e-3)
            (Path(tmp) / vf.FEATURES_NAME).write_text("{not json", encoding="utf-8")
            self.assertIsNone(vf.load_features(tmp))
            bad = {**features, "novelty": features["novelty"][:2]}
            (Path(tmp) / vf.FEATURES_NAME).write_text(json.dumps(bad), encoding="utf-8")
            self.assertIsNone(vf.load_features(tmp))


class RedundancyTests(unittest.TestCase):
    def test_identical_frames_are_fully_redundant_and_orthogonal_ones_are_not(self):
        embeddings = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        times = [0.0, 0.5, 1.0]
        self.assertAlmostEqual(vf.mean_pairwise_similarity(embeddings, times, [0.0, 0.5]), 1.0)
        self.assertAlmostEqual(vf.mean_pairwise_similarity(embeddings, times, [0.0, 1.0]), 0.0)
        self.assertIsNone(vf.mean_pairwise_similarity(embeddings, times, [0.0]))
        self.assertIsNone(vf.mean_pairwise_similarity(embeddings[:2], times, [0.0, 1.0]))
        self.assertIsNone(vf.mean_pairwise_similarity(None, times, [0.0, 1.0]))


@unittest.skipUnless(HAS_SCENEDETECT and HAS_FFMPEG, "scenedetect/ffmpeg not installed")
class ShotDetectionTests(unittest.TestCase):
    def test_finds_hard_cuts_in_a_synthetic_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "cuts.mp4"
            size = "size=320x240:rate=25:duration=4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-y",
                "-f", "lavfi", "-i", f"testsrc2={size}",
                "-f", "lavfi", "-i", f"smptebars={size}",
                "-f", "lavfi", "-i", f"color=c=red:{size}",
                "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
                "-map", "[v]", "-pix_fmt", "yuv420p", str(video),
            ], check=True)
            shots = vf.detect_shots(video)
        self.assertEqual(len(shots), 2)
        self.assertAlmostEqual(shots[0], 4.0, delta=0.2)
        self.assertAlmostEqual(shots[1], 8.0, delta=0.2)


class LoadCurrentFeaturesTests(unittest.TestCase):
    MANIFEST = FeatureCacheTests.MANIFEST

    def write(self, root, manifest=None):
        manifest = manifest or self.MANIFEST
        features, embeddings = vf.compute_visual_features(
            manifest, FeatureCacheTests.fake_embed, FeatureCacheTests.fake_detect)
        vf.write_features(root, features, embeddings)
        (Path(root) / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return features

    def test_fresh_features_load_and_stale_ones_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            features = self.write(tmp)
            self.assertEqual(vf.load_current_features(tmp), features)
            self.assertEqual(vf.load_current_features(tmp, self.MANIFEST), features)
            # Re-extracted frames (new manifest hash), a new config, or a new version: stale.
            self.assertIsNone(vf.load_current_features(tmp, {**self.MANIFEST, "config_hash": "new"}))
            self.assertIsNone(vf.load_current_features(tmp, self.MANIFEST, {"shot_threshold": 10.0}))
            with patch.object(vf, "FEATURE_VERSION", vf.FEATURE_VERSION + 1):
                self.assertIsNone(vf.load_current_features(tmp))

    def test_unverifiable_features_count_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write(tmp)
            (Path(tmp) / "manifest.json").write_text("{not json", encoding="utf-8")
            self.assertIsNone(vf.load_current_features(tmp))
            (Path(tmp) / "manifest.json").unlink()
            self.assertIsNone(vf.load_current_features(tmp))
            self.assertIsNone(vf.load_current_features(Path(tmp) / "nowhere"))


class CacheScriptTests(unittest.TestCase):
    """Hardening of scripts/cache_visual_features.py (see its module docstring)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make_cache(self, video_id, frames=4, manifest_hash="h"):
        cache = self.root / video_id
        (cache / "frames").mkdir(parents=True)
        paths = []
        for i in range(frames):
            path = cache / "frames" / f"{i}.jpg"
            path.write_bytes(b"jpg")
            paths.append(str(path))
        manifest = {
            "source": f"{video_id}.mp4", "config_hash": manifest_hash, "audio_path": None,
            "frames": [{"index": i, "timestamp": i * 0.5, "path": p} for i, p in enumerate(paths)],
        }
        (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return cache

    def run_script(self, *extra, embed=None, embedder_error=None):
        embed = embed or FeatureCacheTests.fake_embed
        calls = {"embedder_built": 0}

        def embed_factory(**kwargs):
            calls["embedder_built"] += 1
            if embedder_error:
                raise embedder_error
            return embed

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.object(vf, "clip_embedder", embed_factory),
            patch.object(vf, "detect_shots", FeatureCacheTests.fake_detect),
            redirect_stdout(stdout), redirect_stderr(stderr),
        ):
            code = cache_visual_features.main(["--cache-dir", str(self.root), *extra])
        return code, stdout.getvalue() + stderr.getvalue(), calls["embedder_built"]

    def report(self):
        return json.loads((self.root / cache_visual_features.REPORT_NAME).read_text())

    def test_computes_reuses_and_survives_a_bad_video(self):
        good = self.make_cache("good")
        bad = self.make_cache("bad", frames=0)
        (bad / vf.FEATURES_NAME).write_text("{}", encoding="utf-8")  # unverifiable leftovers
        code, out, _ = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("FAILED bad", out)
        self.assertIsNotNone(vf.load_current_features(good))
        self.assertFalse((bad / vf.FEATURES_NAME).exists())
        self.assertEqual((self.report()["computed"], self.report()["reused"]), (1, 0))
        self.assertIn("bad", self.report()["failures"])
        # A second run reuses the fresh features and only retries the failure.
        code, _, built = self.run_script()
        self.assertEqual((code, self.report()["reused"], self.report()["computed"]), (1, 1, 0))

    def test_a_failed_overwrite_keeps_features_that_are_still_current(self):
        cache = self.make_cache("good")
        self.run_script()
        before = (cache / vf.FEATURES_NAME).read_bytes()

        def explode(paths):
            raise RuntimeError("CUDA out of memory")

        code, out, _ = self.run_script("--overwrite", embed=explode)
        self.assertEqual(code, 1)
        self.assertIn("kept current features", out)
        self.assertEqual((cache / vf.FEATURES_NAME).read_bytes(), before)
        self.assertIsNotNone(vf.load_current_features(cache))

    def test_unresolvable_frame_paths_do_not_delete_current_features(self):
        # Running from another working directory: relative frame paths do not resolve.
        cache = self.make_cache("good")
        self.run_script()
        manifest = json.loads((cache / "manifest.json").read_text())
        for frame in manifest["frames"]:
            frame["path"] = f"data/cache/good/frames/{frame['index']}.jpg"  # relative to repo root
        (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        code, out, _ = self.run_script("--overwrite")
        self.assertEqual(code, 1)
        self.assertIn("do not resolve from this working directory", out)
        # The manifest hash is unchanged, so the features are still current and kept.
        self.assertIsNotNone(vf.load_current_features(cache))

    def test_stale_features_are_removed_when_recompute_fails(self):
        cache = self.make_cache("good")
        self.run_script()
        # Frames were re-extracted (new manifest hash) and one frame file is now missing.
        manifest = json.loads((cache / "manifest.json").read_text())
        manifest["config_hash"] = "re-extracted"
        Path(manifest["frames"][1]["path"]).unlink()
        (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        code, out, _ = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("removed stale features", out)
        self.assertFalse((cache / vf.FEATURES_NAME).exists())
        self.assertFalse((cache / vf.EMBEDDINGS_NAME).exists())

    def test_missing_dependency_aborts_the_run_and_touches_nothing(self):
        first = self.make_cache("a")
        self.make_cache("b")
        self.run_script()
        before = (first / vf.FEATURES_NAME).read_bytes()
        report_before = (self.root / cache_visual_features.REPORT_NAME).read_text()
        code, out, built = self.run_script(
            "--overwrite", embedder_error=ModuleNotFoundError("No module named 'open_clip'"))
        self.assertEqual(code, 2)
        self.assertEqual(built, 1)  # stopped at the first video, did not retry for every video
        self.assertIn("missing dependency", out)
        self.assertEqual((first / vf.FEATURES_NAME).read_bytes(), before)
        self.assertEqual((self.root / cache_visual_features.REPORT_NAME).read_text(), report_before)

    def test_missing_or_empty_cache_dir_is_not_success(self):
        for target in (self.root / "nowhere", self.root):
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = cache_visual_features.main(["--cache-dir", str(target)])
            self.assertEqual(code, 1)
            self.assertIn("no cached videos found", stderr.getvalue())
        self.assertFalse((self.root / cache_visual_features.REPORT_NAME).exists())

    def test_frame_cache_changing_during_compute_is_discarded(self):
        cache = self.make_cache("good")
        self.run_script()
        old = (cache / vf.FEATURES_NAME).read_bytes()

        def reextract_midway(paths):
            manifest = json.loads((cache / "manifest.json").read_text())
            manifest["config_hash"] = "changed-by-another-job"
            (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return FeatureCacheTests.fake_embed(paths)

        code, out, _ = self.run_script("--overwrite", embed=reextract_midway)
        self.assertEqual(code, 1)
        self.assertIn("changed while features were being computed", out)
        # The old features no longer match the new manifest, so they were removed rather than reused.
        self.assertFalse((cache / vf.FEATURES_NAME).exists())
        self.assertNotEqual(old, b"")


class BuildMomentsVisualTests(unittest.TestCase):
    """build_moments --selector V against real (synthetic) media caches."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        splits = self.root / "splits.json"
        splits.write_text(json.dumps({"k_rule": {"rate_per_minute": 3}}))
        self.splits = splits

    def make_cache(self, video_id, features=None):
        paths = fx.make_media_cache(self.root, video_id, audio=False)
        if features is not None:
            stamped = {**features, "config": vf.DEFAULT_CONFIG,
                       "config_hash": vf.features_hash(paths["manifest"], vf.DEFAULT_CONFIG)}
            (paths["cache"] / vf.FEATURES_NAME).write_text(json.dumps(stamped), encoding="utf-8")
        return paths

    def run_v(self, videos):
        output = self.root / "moments.jsonl"
        stdout = io.StringIO()
        with (
            patch.object(sys, "argv", ["build_moments.py", "--selector", "V", "--splits", str(self.splits),
                                      "--cache-dir", str(self.root), "--output", str(output)]),
            patch.object(build_moments, "load_targets", return_value=videos),
            patch.object(build_moments, "frame_window", return_value=[]),
            redirect_stdout(stdout),
        ):
            build_moments.main()
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        report = json.loads(output.with_suffix(".report.json").read_text())
        return rows, report, stdout.getvalue()

    def test_v_moments_come_from_visual_features_and_missing_features_are_reported(self):
        self.make_cache("good", spike_features())
        self.make_cache("nofeatures")
        rows, report, stdout = self.run_v([("good", 60, []), ("nofeatures", 60, [])])
        self.assertEqual([r["time"] for r in rows], [10.0, 30.0, 50.0])
        self.assertTrue(all(r["selector"] == "V" and r["video_id"] == "good" for r in rows))
        self.assertTrue(all(r["word_limit"] is None for r in rows))  # no audio features cached
        by_id = {v["video_id"]: v for v in report["videos"]}
        self.assertEqual(by_id["good"]["status"], "ok")
        self.assertEqual(by_id["nofeatures"]["status"], "missing_visual_features")
        self.assertEqual(by_id["nofeatures"]["selection_shortfall"], 0)
        self.assertIn("no visual features; zero V moments: nofeatures", stdout)

    def test_stale_features_on_disk_are_treated_as_missing(self):
        paths = self.make_cache("vid", spike_features())
        rows, report, _ = self.run_v([("vid", 60, [])])
        self.assertEqual(len(rows), 3)
        self.assertEqual(report["videos"][0]["status"], "ok")
        # The frames are re-extracted (new manifest hash) but nobody reran the caching script.
        manifest = {**paths["manifest"], "config_hash": "m2"}
        (paths["cache"] / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        rows, report, _ = self.run_v([("vid", 60, [])])
        self.assertEqual(rows, [])
        self.assertEqual(report["videos"][0]["status"], "missing_visual_features")

    def test_registered_selector_signature_is_unchanged(self):
        self.assertEqual(build_moments.SELECTORS["V"](60.0, 3, None), [])
        self.assertEqual(build_moments.SELECTORS["V"](60.0, 3, spike_features()), [10.0, 30.0, 50.0])


if __name__ == "__main__":
    unittest.main()
