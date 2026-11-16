import io
import json
from contextlib import redirect_stdout
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
import build_moments  # noqa: E402
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


class CacheScriptTests(unittest.TestCase):
    def make_cache(self, root, video_id, frames=4):
        cache = root / video_id
        cache.mkdir()
        manifest = {
            "source": f"{video_id}.mp4", "config_hash": "h",
            "frames": [{"index": i, "timestamp": i * 0.5, "path": f"{i}.jpg"} for i in range(frames)],
        }
        (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return cache

    def run_script(self, root, *extra):
        def embed_factory(**kwargs):
            return FeatureCacheTests.fake_embed

        real_compute = vf.compute_visual_features
        stdout = io.StringIO()
        with (
            patch.object(sys, "argv", ["cache_visual_features.py", "--cache-dir", str(root), *extra]),
            patch.object(vf, "clip_embedder", embed_factory),
            patch.object(vf, "detect_shots", FeatureCacheTests.fake_detect),
            patch.object(vf, "compute_visual_features", side_effect=real_compute) as compute,
            redirect_stdout(stdout),
        ):
            try:
                cache_visual_features.main()
                code = 0
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), compute.call_count

    def test_computes_reuses_and_survives_a_bad_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = self.make_cache(root, "good")
            bad = self.make_cache(root, "bad", frames=0)
            (bad / vf.FEATURES_NAME).write_text("{}", encoding="utf-8")  # stale leftovers
            code, out, calls = self.run_script(root)
            self.assertEqual(code, 1)  # one video failed
            self.assertIn("FAILED bad", out)
            self.assertEqual(calls, 2)
            self.assertIsNotNone(vf.load_features(good))
            self.assertFalse((bad / vf.FEATURES_NAME).exists())  # outdated features removed
            report = json.loads((root / "visual_features_report.json").read_text())
            self.assertEqual((report["computed"], report["reused"]), (1, 0))
            self.assertIn("bad", report["failures"])
            # A second run reuses the fresh features and only retries the failure.
            code, out, calls = self.run_script(root)
            self.assertEqual(calls, 1)
            self.assertEqual(json.loads((root / "visual_features_report.json").read_text())["reused"], 1)
            # --overwrite recomputes everything that can be computed.
            _, _, calls = self.run_script(root, "--overwrite")
            self.assertEqual(calls, 2)


class BuildMomentsVisualTests(unittest.TestCase):
    def run_v(self, videos, visual, audio=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits = root / "splits.json"
            splits.write_text(json.dumps({"k_rule": {"rate_per_minute": 3}}))
            output = root / "moments.jsonl"
            frame = root / "frame.jpg"
            frame.write_bytes(b"jpg")
            for video_id, _, _ in videos:
                (root / video_id).mkdir()
                (root / video_id / "manifest.json").write_text("{}", encoding="utf-8")
            manifest = {"frames": [{"index": 0, "timestamp": 0.0, "path": str(frame)}], "audio_path": None}
            stdout = io.StringIO()
            with (
                patch.object(sys, "argv", ["build_moments.py", "--selector", "V", "--splits", str(splits),
                                          "--cache-dir", str(root), "--output", str(output)]),
                patch.object(build_moments, "load_targets", return_value=videos),
                patch.object(build_moments, "load_manifest", return_value=manifest),
                patch.object(build_moments, "load_features", side_effect=lambda p: (audio or {}).get(p.name)),
                patch.object(build_moments, "load_visual_features", side_effect=lambda p: visual.get(p.name)),
                patch.object(build_moments, "frame_window", return_value=[]),
                redirect_stdout(stdout),
            ):
                build_moments.main()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            report = json.loads(output.with_suffix(".report.json").read_text())
            return rows, report, stdout.getvalue()

    def test_v_moments_come_from_visual_features_and_missing_features_are_reported(self):
        rows, report, stdout = self.run_v(
            [("good", 60, []), ("nofeatures", 60, [])], {"good": spike_features()},
        )
        self.assertEqual([r["time"] for r in rows], [10.0, 30.0, 50.0])
        self.assertTrue(all(r["selector"] == "V" and r["video_id"] == "good" for r in rows))
        self.assertTrue(all(r["word_limit"] is None for r in rows))  # no audio features cached
        by_id = {v["video_id"]: v for v in report["videos"]}
        self.assertEqual(by_id["good"]["status"], "ok")
        self.assertEqual(by_id["nofeatures"]["status"], "missing_visual_features")
        self.assertEqual(by_id["nofeatures"]["selection_shortfall"], 0)
        self.assertIn("no visual features; zero V moments: nofeatures", stdout)

    def test_registered_selector_signature_is_unchanged(self):
        self.assertEqual(build_moments.SELECTORS["V"](60.0, 3, None), [])
        self.assertEqual(build_moments.SELECTORS["V"](60.0, 3, spike_features()), [10.0, 30.0, 50.0])


if __name__ == "__main__":
    unittest.main()
