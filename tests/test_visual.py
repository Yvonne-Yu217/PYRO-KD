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


def strip(record):
    """A features record without the embeddings reference write_features adds."""
    return {key: value for key, value in record.items() if key != "embeddings"}


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

    def test_round_trip_commits_features_and_embeddings_as_a_pair(self):
        features, embeddings = self.compute()
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(vf.load_features(tmp))
            self.assertIsNone(vf.load_embeddings(tmp))
            vf.write_features(tmp, features, embeddings)
            loaded = vf.load_features(tmp)
            self.assertEqual(strip(loaded), features)
            self.assertNotIn("embeddings", features)  # the caller's record is not modified
            np.testing.assert_allclose(vf.load_embeddings(tmp), embeddings, atol=1e-3)
            generations = sorted(p.name for p in Path(tmp).iterdir() if p.name.startswith(vf.EMBEDDINGS_PREFIX))
            self.assertEqual(generations, [loaded["embeddings"]["file"]])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir() if p.name.endswith(".tmp")), [])
            (Path(tmp) / vf.FEATURES_NAME).write_text("{not json", encoding="utf-8")
            self.assertIsNone(vf.load_features(tmp))

    def test_write_rejects_inconsistent_inputs_and_writes_nothing(self):
        features, embeddings = self.compute()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                vf.write_features(tmp, features, embeddings[:2])  # wrong number of rows
            with self.assertRaises(ValueError):
                vf.write_features(tmp, features, np.full(embeddings.shape, np.nan))
            with self.assertRaises(ValueError):
                vf.write_features(tmp, {**features, "novelty": None}, embeddings)
            self.assertEqual(list(Path(tmp).iterdir()), [])


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
            self.assertEqual(strip(vf.load_current_features(tmp)), features)
            self.assertEqual(strip(vf.load_current_features(tmp, self.MANIFEST)), features)
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


MALFORMED_RECORDS = {
    "novelty null": {"novelty": None},
    "novelty not a list": {"novelty": 3},
    "times string": {"frame_times": "abc"},
    "shots object": {"shots": {"a": 1}},
    "shots missing": {"shots": None},
    "length mismatch": {"novelty": [0.1]},
    "string time": {"frame_times": ["a", 0.5, 1.0, 1.5]},
    "boolean time": {"frame_times": [True, 0.5, 1.0, 1.5]},
    "nan novelty": {"novelty": [0.0, float("nan"), 0.1, 0.1]},
    "negative novelty": {"novelty": [0.0, -0.5, 0.1, 0.1]},
    "unsorted times": {"frame_times": [0.0, 1.0, 0.5, 1.5]},
    "duplicate times": {"frame_times": [0.0, 0.5, 0.5, 1.5]},
    "negative shot": {"shots": [-1.0]},
    "time integer too large for a float": {"frame_times": [0.0, 0.5, 10**400, 1.5]},
    "novelty integer too large for a float": {"novelty": [0.0, 10**400, 0.1, 0.1]},
    "shot integer too large for a float": {"shots": [10**400]},
}


class MalformedRecordTests(unittest.TestCase):
    """A correctly stamped record can still carry a payload that would crash a consumer."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def cache_with(self, **overrides):
        paths = fx.make_visual_cache(
            self.root, "v", make_features([0.0, 0.5, 1.0, 1.5], [0.0, 0.2, 0.9, 0.1]))
        path = paths["cache"] / vf.FEATURES_NAME
        path.write_text(json.dumps({**json.loads(path.read_text()), **overrides}), encoding="utf-8")
        return paths

    def test_every_malformed_record_is_unavailable_not_an_exception(self):
        for name, override in MALFORMED_RECORDS.items():
            with self.subTest(defect=name):
                paths = self.cache_with(**override)
                self.assertIsNone(vf.load_features(paths["cache"]))
                self.assertIsNone(vf.load_current_features(paths["cache"]))
                self.assertEqual(vf.load_validated_features(paths["cache"]), (None, vf.MALFORMED_VISUAL_FEATURES))

    def test_a_clean_record_loads_and_the_reasons_are_distinct(self):
        paths = self.cache_with()
        features, reason = vf.load_validated_features(paths["cache"])
        self.assertIsNone(reason)
        self.assertEqual(features["frame_times"], [0.0, 0.5, 1.0, 1.5])
        (paths["cache"] / vf.FEATURES_NAME).write_text("{no", encoding="utf-8")
        self.assertEqual(vf.load_validated_features(paths["cache"]), (None, vf.UNREADABLE_VISUAL_FEATURES))
        (paths["cache"] / vf.FEATURES_NAME).unlink()
        self.assertEqual(vf.load_validated_features(paths["cache"]), (None, vf.NO_VISUAL_FEATURES))

    def test_build_moments_v_does_not_abort_on_a_malformed_record(self):
        for name in ("novelty null", "unsorted times", "nan novelty"):
            with self.subTest(defect=name):
                paths = self.cache_with(**MALFORMED_RECORDS[name])
                rows, report = run_build_moments(self.root, [("v", 60, [])])
                self.assertEqual(rows, [])
                self.assertEqual(report["videos"][0]["status"], "missing_visual_features")


class ManifestAlignmentTests(unittest.TestCase):
    """Kiran #19 P2: matching hash is not enough; rows must line up with the cached frames."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def swapped_timeline(self, times):
        paths = fx.make_visual_cache(
            self.root, "v", make_features([0.0, 2.0, 4.0], [0.0, 0.9, 0.1]), embeddings=np.eye(3))
        path = paths["cache"] / vf.FEATURES_NAME
        record = json.loads(path.read_text())
        record["frame_times"] = times
        record["novelty"] = record["novelty"][: len(times)]
        path.write_text(json.dumps(record), encoding="utf-8")  # hash untouched
        return paths

    def test_shifted_truncated_or_extended_timelines_are_rejected_everywhere(self):
        for name, times in (("shifted", [10.0, 30.0, 50.0]), ("truncated", [0.0, 2.0]),
                            ("slightly off", [0.0, 2.0, 4.5])):
            with self.subTest(timeline=name):
                paths = self.swapped_timeline(times)
                self.assertEqual(
                    vf.load_validated_features(paths["cache"]), (None, vf.INCONSISTENT_VISUAL_FEATURES))
                self.assertIsNone(vf.load_current_features(paths["cache"]))
                rows, report = run_build_moments(self.root, [("v", 60, [])])
                self.assertEqual(rows, [])
                self.assertEqual(report["videos"][0]["status"], "missing_visual_features")

    def test_matching_timeline_is_accepted(self):
        paths = fx.make_visual_cache(self.root, "v", make_features([0.0, 2.0, 4.0], [0.0, 0.9, 0.1]))
        self.assertIsNone(vf.load_validated_features(paths["cache"])[1])


class TransactionalPublishTests(unittest.TestCase):
    """Kiran #19 P2: features and embeddings are published as one unit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.features, self.embeddings = vf.compute_visual_features(
            FeatureCacheTests.MANIFEST, FeatureCacheTests.fake_embed, FeatureCacheTests.fake_detect)
        vf.write_features(self.dir, self.features, self.embeddings)
        self.old_json = (self.dir / vf.FEATURES_NAME).read_bytes()
        self.old_embeddings = vf.load_embeddings(self.dir)
        self.new_embeddings = self.embeddings[::-1].copy()  # a different matrix, so a different generation
        self.old_files = sorted(p.name for p in self.dir.iterdir())

    def fail_json_commit(self, before_failure=None):
        """Patch Path.replace so the first rename onto the JSON record fails (after running before_failure)."""
        real_replace = Path.replace
        state = {"armed": True}

        def replace(source, target):
            if Path(target).name == vf.FEATURES_NAME and state["armed"]:
                state["armed"] = False
                if before_failure:
                    before_failure()
                raise OSError("disk full")
            return real_replace(source, target)

        return patch.object(Path, "replace", replace)

    def test_failure_after_embeddings_are_published_leaves_the_old_pair_intact(self):
        with self.fail_json_commit():
            with self.assertRaises(OSError):
                vf.write_features(self.dir, self.features, self.new_embeddings)
        self.assertEqual((self.dir / vf.FEATURES_NAME).read_bytes(), self.old_json)
        np.testing.assert_array_equal(vf.load_embeddings(self.dir), self.old_embeddings)
        self.assertIsNotNone(vf.load_current_features(self.dir, FeatureCacheTests.MANIFEST))
        leftovers = sorted(set(p.name for p in self.dir.iterdir()) - set(self.old_files))
        self.assertEqual(len(leftovers), 1)  # the immutable generation stays; no staging files remain
        self.assertTrue(leftovers[0].startswith(vf.EMBEDDINGS_PREFIX))
        # An unreferenced generation is collected after a later commit once it is past the grace period.
        with patch.object(vf, "GENERATION_GRACE_SECONDS", -1.0):
            vf.write_features(self.dir, self.features, self.embeddings)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), self.old_files)

    # Kiran, #19 P2: another writer commits the same content-named generation before our commit fails.
    def test_a_failed_writer_does_not_delete_a_generation_another_writer_committed(self):
        def other_writer_commits():
            vf.write_features(self.dir, self.features, self.new_embeddings)  # identical embeddings

        with self.fail_json_commit(before_failure=other_writer_commits):
            with self.assertRaises(OSError):
                vf.write_features(self.dir, self.features, self.new_embeddings)
        np.testing.assert_allclose(vf.load_embeddings(self.dir), self.new_embeddings, atol=1e-3)
        self.assertIsNotNone(vf.load_current_features(self.dir, FeatureCacheTests.MANIFEST))

    def test_collection_never_removes_the_generation_the_committed_record_names(self):
        record = vf.load_features(self.dir)
        with patch.object(vf, "GENERATION_GRACE_SECONDS", -1.0):
            # A writer whose own generation differs must not collect the one now committed on disk.
            real = vf._read_features
            with patch.object(vf, "_read_features", side_effect=lambda d: (record, None)):
                vf.write_features(self.dir, self.features, self.new_embeddings)
            self.assertTrue((self.dir / record["embeddings"]["file"]).exists())
            self.assertIs(vf._read_features, real)

    # Kiran, #19 P2: an existing digest-named file with the wrong bytes must be repaired, not adopted.
    def test_a_corrupt_file_under_the_expected_name_is_repaired_by_the_next_write(self):
        record = vf.load_features(self.dir)
        generation = self.dir / record["embeddings"]["file"]
        generation.write_bytes(b"corrupt")
        self.assertIsNone(vf.load_embeddings(self.dir))
        vf.write_features(self.dir, self.features, self.embeddings)
        np.testing.assert_array_equal(vf.load_embeddings(self.dir), self.old_embeddings)

    def test_failure_while_writing_the_embeddings_leaves_the_old_pair_intact(self):
        with patch.object(Path, "write_bytes", side_effect=OSError("quota")):
            with self.assertRaises(OSError):
                vf.write_features(self.dir, self.features, self.new_embeddings)
        self.assertEqual((self.dir / vf.FEATURES_NAME).read_bytes(), self.old_json)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), self.old_files)

    def test_successful_publish_swaps_the_pair_and_keeps_recent_generations(self):
        vf.write_features(self.dir, self.features, self.new_embeddings)
        np.testing.assert_allclose(vf.load_embeddings(self.dir), self.new_embeddings, atol=1e-3)
        generations = [p for p in self.dir.iterdir() if p.name.startswith(vf.EMBEDDINGS_PREFIX)]
        self.assertEqual(len(generations), 2)  # the previous one is kept for the grace period

    def test_old_generations_are_removed_after_the_grace_period(self):
        with patch.object(vf, "GENERATION_GRACE_SECONDS", -1.0):
            vf.write_features(self.dir, self.features, self.new_embeddings)
        generations = [p.name for p in self.dir.iterdir() if p.name.startswith(vf.EMBEDDINGS_PREFIX)]
        self.assertEqual(generations, [vf.load_features(self.dir)["embeddings"]["file"]])

    def test_concurrent_writers_use_distinct_staging_names(self):
        names = {vf._unique_tmp(self.dir, "x").name for _ in range(50)}
        self.assertEqual(len(names), 50)

    def test_a_record_cannot_point_at_a_missing_changed_or_foreign_file(self):
        record = json.loads(self.old_json)
        generation = self.dir / record["embeddings"]["file"]
        cases = {
            "digest mismatch": lambda: np.save(generation.open("wb"), self.embeddings[::-1] + 1.0),
            "unreadable bytes": lambda: generation.write_bytes(b"corrupt"),
            "missing generation": lambda: generation.unlink(),
        }
        for name, break_it in cases.items():
            with self.subTest(defect=name):
                vf.write_features(self.dir, self.features, self.embeddings)
                break_it()
                self.assertIsNone(vf.load_embeddings(self.dir))
        for name, reference in (
            ("path traversal", {**record["embeddings"], "file": "../outside.npy"}),
            ("wrong prefix", {**record["embeddings"], "file": "other.npy"}),
            ("no reference", None),
            ("not an object", "x"),
        ):
            with self.subTest(defect=name):
                forged = {key: value for key, value in record.items() if key != "embeddings"}
                if reference is not None:
                    forged["embeddings"] = reference
                self.assertIsNone(vf.load_embeddings(self.dir, forged))


class EmbeddingValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.features = {**make_features([0.0, 0.5, 1.0], [0.0, 0.5, 0.1]), "embeddings": None}

    def stage(self, array):
        """Publish a (possibly invalid) array behind a record with a correct digest."""
        import hashlib
        import io as _io
        buffer = _io.BytesIO()
        np.save(buffer, array)
        data = buffer.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        name = f"{vf.EMBEDDINGS_PREFIX}.{digest[:16]}.npy"
        (self.dir / name).write_bytes(data)
        return {**self.features, "embeddings": {"file": name, "sha256": digest}}

    def stage_bytes(self, data):
        import hashlib
        digest = hashlib.sha256(data).hexdigest()
        name = f"{vf.EMBEDDINGS_PREFIX}.{digest[:16]}.npy"
        (self.dir / name).write_bytes(data)
        return {**self.features, "embeddings": {"file": name, "sha256": digest}}

    # Kiran, #19 P2: a correctly hashed file can still be something np.load does not decode to an array.
    def test_empty_truncated_and_archive_payloads_are_unusable_not_exceptions(self):
        import io as _io
        archive = _io.BytesIO()
        np.savez(archive, matrix=np.eye(3))
        good = _io.BytesIO()
        np.save(good, np.eye(3, dtype=np.float32))
        for name, data in (
            ("empty", b""),
            ("npz archive", archive.getvalue()),
            ("truncated npy", good.getvalue()[:-20]),
            ("garbage", b"\x93NUMPY nonsense"),
            ("zip header only", b"PK\x03\x04"),
        ):
            with self.subTest(payload=name):
                self.assertIsNone(vf.load_embeddings(self.dir, self.stage_bytes(data)))

    def test_only_a_finite_matrix_with_one_row_per_frame_loads(self):
        self.assertIsNotNone(vf.load_embeddings(self.dir, self.stage(np.eye(3, dtype=np.float32))))
        for name, array in (
            ("1-D", np.ones(3, dtype=np.float32)),
            ("nan", np.full((3, 2), np.nan, dtype=np.float32)),
            ("inf", np.full((3, 2), np.inf, dtype=np.float32)),
            ("wrong rows", np.ones((2, 2), dtype=np.float32)),
            ("no columns", np.ones((3, 0), dtype=np.float32)),
            ("strings", np.array([["a"], ["b"], ["c"]])),
        ):
            with self.subTest(embeddings=name):
                self.assertIsNone(vf.load_embeddings(self.dir, self.stage(array)))

    def test_redundancy_diagnostic_never_raises(self):
        times = [0.0, 0.5, 1.0]
        for bad in (np.ones(3), np.full((3, 2), np.nan), np.ones((2, 2)), "text", [[1, "a"]] * 3):
            with self.subTest(bad=repr(bad)[:20]):
                self.assertIsNone(vf.mean_pairwise_similarity(bad, times, [0.0, 1.0]))


def run_build_moments(root, videos, audio_cache=None):
    """Run build_moments --selector V over caches under root; returns (rows, report)."""
    root = Path(root)
    splits = root / "splits.json"
    splits.write_text(json.dumps({"k_rule": {"rate_per_minute": 3}}))
    output = root / "moments.jsonl"
    with (
        patch.object(sys, "argv", ["build_moments.py", "--selector", "V", "--splits", str(splits),
                                  "--cache-dir", str(root), "--output", str(output)]),
        patch.object(build_moments, "load_targets", return_value=videos),
        patch.object(build_moments, "frame_window", return_value=[]),
        redirect_stdout(io.StringIO()),
        fx.patched_audio_helpers(),
    ):
        build_moments.main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    return rows, json.loads(output.with_suffix(".report.json").read_text())


class CacheScriptTests(unittest.TestCase):
    """Hardening of scripts/cache_visual_features.py (see its module docstring)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make_cache(self, video_id, frames=4, manifest_hash="m1"):
        times = [i * 0.5 for i in range(frames)]
        paths = fx.make_media_cache(
            self.root, video_id, audio=False, manifest_hash=manifest_hash, frame_times=times or (0.0,))
        if not frames:
            manifest = {**paths["manifest"], "frames": []}
            (paths["cache"] / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return paths["cache"]

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
        self.assertIsNotNone(vf.load_embeddings(good))
        self.assertFalse((bad / vf.FEATURES_NAME).exists())
        self.assertEqual((self.report()["computed"], self.report()["reused"]), (1, 0))
        self.assertIn("bad", self.report()["failures"])
        code, _, _ = self.run_script()
        self.assertEqual((code, self.report()["reused"], self.report()["computed"]), (1, 1, 0))

    def test_features_without_their_embeddings_are_recomputed_not_reused(self):
        good = self.make_cache("good")
        self.run_script()
        record = json.loads((good / vf.FEATURES_NAME).read_text())
        (good / record["embeddings"]["file"]).unlink()
        code, _, _ = self.run_script()
        self.assertEqual((code, self.report()["computed"], self.report()["reused"]), (0, 1, 0))
        self.assertIsNotNone(vf.load_embeddings(good))

    # Kiran, #19 P2: recomputing identical embeddings over a corrupt generation must repair it.
    def test_a_corrupt_embeddings_generation_is_repaired_by_rerunning(self):
        cache = self.make_cache("good")
        self.run_script()
        record = json.loads((cache / vf.FEATURES_NAME).read_text())
        (cache / record["embeddings"]["file"]).write_bytes(b"corrupt")
        self.assertIsNone(vf.load_embeddings(cache))
        for extra in ((), ("--overwrite",)):
            with self.subTest(extra=extra):
                (cache / record["embeddings"]["file"]).write_bytes(b"corrupt")
                code, _, _ = self.run_script(*extra)
                self.assertEqual((code, self.report()["computed"]), (0, 1))  # recomputed, not "reused"
                self.assertIsNotNone(vf.load_current_features(cache))
                self.assertIsNotNone(vf.load_embeddings(cache))

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
        self.assertIsNotNone(vf.load_embeddings(cache))

    def test_unresolvable_frame_paths_do_not_delete_current_features(self):
        cache = self.make_cache("good")
        self.run_script()
        manifest = json.loads((cache / "manifest.json").read_text())
        for frame in manifest["frames"]:
            frame["path"] = f"data/cache/good/frames/{frame['index']}.jpg"  # relative to repo root
        (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        code, out, _ = self.run_script("--overwrite")
        self.assertEqual(code, 1)
        self.assertIn("do not resolve from this working directory", out)
        self.assertIn("kept current features", out)
        self.assertIsNotNone(vf.load_current_features(cache))

    def test_stale_features_are_removed_when_recompute_fails(self):
        cache = self.make_cache("good")
        self.run_script()
        manifest = json.loads((cache / "manifest.json").read_text())
        manifest["config_hash"] = "re-extracted"
        Path(manifest["frames"][1]["path"]).unlink()
        (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        code, out, _ = self.run_script()
        self.assertEqual(code, 1)
        self.assertIn("removed stale features", out)
        self.assertFalse((cache / vf.FEATURES_NAME).exists())
        self.assertEqual(list(cache.glob(f"{vf.EMBEDDINGS_PREFIX}.*.npy")), [])

    def test_missing_dependency_aborts_the_run_and_touches_nothing(self):
        first = self.make_cache("a")
        self.make_cache("b")
        self.run_script()
        before = (first / vf.FEATURES_NAME).read_bytes()
        report_before = (self.root / cache_visual_features.REPORT_NAME).read_text()
        code, out, built = self.run_script(
            "--overwrite", embedder_error=ModuleNotFoundError("No module named 'open_clip'"))
        self.assertEqual(code, 2)
        self.assertEqual(built, 1)
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

        def reextract_midway(paths):
            manifest = json.loads((cache / "manifest.json").read_text())
            manifest["config_hash"] = "changed-by-another-job"
            (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            return FeatureCacheTests.fake_embed(paths)

        code, out, _ = self.run_script("--overwrite", embed=reextract_midway)
        self.assertEqual(code, 1)
        self.assertIn("changed while features were being computed", out)
        self.assertFalse((cache / vf.FEATURES_NAME).exists())

    # Kiran #19 P3: the producer must check media freshness, not only the manifest hash.
    def test_replaced_source_is_a_failure_not_a_reuse_or_a_recompute(self):
        cache = self.make_cache("good")
        self.run_script()
        (self.root / "sources" / "good.mp4").write_bytes(b"a different, longer replacement video")
        for extra in ((), ("--overwrite",)):
            with self.subTest(extra=extra):
                code, out, built = self.run_script(*extra)
                self.assertEqual(code, 1)
                self.assertIn("media_cache_stale", out)
                self.assertEqual(built, 0)  # the embedder is never even loaded
                self.assertEqual((self.report()["reused"], self.report()["computed"]), (0, 0))
                self.assertFalse((cache / vf.FEATURES_NAME).exists())  # no longer describes this video


class BuildMomentsVisualTests(unittest.TestCase):
    """build_moments --selector V against real (synthetic) media caches."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make_cache(self, video_id, features=None, audio=False):
        if features is None:
            return fx.make_media_cache(self.root, video_id, audio=audio)
        return fx.make_visual_cache(self.root, video_id, features, audio=audio)

    def run_v(self, videos):
        rows, report = run_build_moments(self.root, videos)
        return rows, report

    def test_v_moments_come_from_visual_features_and_missing_features_are_reported(self):
        self.make_cache("good", spike_features())
        self.make_cache("nofeatures")
        rows, report = self.run_v([("good", 60, []), ("nofeatures", 60, [])])
        self.assertEqual([r["time"] for r in rows], [10.0, 30.0, 50.0])
        self.assertTrue(all(r["selector"] == "V" and r["video_id"] == "good" for r in rows))
        self.assertTrue(all(r["word_limit"] is None for r in rows))
        by_id = {v["video_id"]: v for v in report["videos"]}
        self.assertEqual(by_id["good"]["status"], "ok")
        self.assertEqual(by_id["nofeatures"]["status"], "missing_visual_features")
        self.assertEqual(by_id["nofeatures"]["selection_shortfall"], 0)

    def test_stale_features_on_disk_are_treated_as_missing(self):
        paths = self.make_cache("vid", spike_features())
        rows, report = self.run_v([("vid", 60, [])])
        self.assertEqual(len(rows), 3)
        manifest = {**paths["manifest"], "config_hash": "m2"}
        (paths["cache"] / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        rows, report = self.run_v([("vid", 60, [])])
        self.assertEqual(rows, [])
        self.assertEqual(report["videos"][0]["status"], "missing_visual_features")

    def test_registered_selector_signature_is_unchanged(self):
        self.assertEqual(build_moments.SELECTORS["V"](60.0, 3, None), [])
        self.assertEqual(build_moments.SELECTORS["V"](60.0, 3, spike_features()), [10.0, 30.0, 50.0])

    # Kiran #19 P2: the word budget must come from validated audio, never a replaced WAV.
    def test_word_budget_uses_validated_audio_only(self):
        paths = self.make_cache("vid", spike_features(), audio=True)
        fx.write_audio_features(paths["cache"], paths["manifest"], speech=[(11.0, 12.0)], gaps=[(12.25, 29.0)])
        rows, _ = self.run_v([("vid", 60, [])])
        self.assertTrue(all(r["word_limit"] is not None for r in rows))
        paths["wav"].write_bytes(b"RIFF....a completely different, longer recording")  # features now stale
        rows, report = self.run_v([("vid", 60, [])])
        self.assertEqual(len(rows), 3)  # V still produces its moments ...
        self.assertTrue(all(r["word_limit"] is None for r in rows))  # ... without trusting stale speech data
        self.assertEqual(report["videos"][0]["status"], "ok")

    def test_malformed_audio_json_does_not_abort_a_v_batch(self):
        bad = self.make_cache("bad", spike_features(), audio=True)
        fx.write_audio_features(bad["cache"], bad["manifest"], raw_speech=[[1.0]])
        self.make_cache("good", spike_features())
        rows, report = self.run_v([("bad", 60, []), ("good", 60, [])])
        self.assertEqual({r["video_id"] for r in rows}, {"bad", "good"})
        self.assertTrue(all(r["word_limit"] is None for r in rows))
        self.assertEqual({v["status"] for v in report["videos"]}, {"ok"})


if __name__ == "__main__":
    unittest.main()
