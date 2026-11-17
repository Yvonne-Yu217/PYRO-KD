import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.datasets.mtg_ad import parse_row
from src.evaluation.stats import bootstrap_mean_ci, paired_bootstrap_diff
from src.utils import cache_checks as cc
from src.utils.selection import oracle_timestamps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_fixtures as fx  # noqa: E402
import evaluate_selectors as ev  # noqa: E402


def make_video(video_id="v1", duration=100.0, ad_times=(10.0, 50.0, 90.0)):
    return parse_row({
        "video_id": video_id, "split": "validation", "category": "Vlog",
        "duration_seconds": duration,
        "gt_ad_segments": [{"start_seconds": t, "text": f"ad {t}"} for t in ad_times],
        "transcript_segments": [], "visual_info": None,
    })


def make_features(speech, gaps):
    return {
        "speech": [list(s) for s in speech],
        "gaps": [{"start": a, "end": b, "rms": 0.0, "onset": 0.0} for a, b in gaps],
    }


class OracleTests(unittest.TestCase):
    def test_uses_all_human_times_when_within_budget(self):
        self.assertEqual(oracle_timestamps([9, 1, 5], 5), [1.0, 5.0, 9.0])

    def test_caps_at_k_evenly_spaced_including_ends(self):
        self.assertEqual(oracle_timestamps(range(1, 11), 4), [1.0, 4.0, 7.0, 10.0])
        self.assertEqual(oracle_timestamps(range(1, 11), 1), [6.0])
        chosen = oracle_timestamps(range(100), 17)
        self.assertEqual(len(chosen), 17)
        self.assertEqual(len(set(chosen)), 17)

    def test_rejects_zero_budget(self):
        with self.assertRaises(ValueError):
            oracle_timestamps([1, 2], 0)


class StatsTests(unittest.TestCase):
    def test_bootstrap_is_deterministic_and_brackets_the_mean(self):
        values = [0.1, 0.2, 0.3, 0.4, 0.5, None]
        first = bootstrap_mean_ci(values, 2000, seed=3)
        self.assertEqual(first, bootstrap_mean_ci(values, 2000, seed=3))
        self.assertEqual(first["n"], 5)
        self.assertAlmostEqual(first["mean"], 0.3)
        self.assertLessEqual(first["ci_low"], 0.3)
        self.assertGreaterEqual(first["ci_high"], 0.3)

    def test_constant_values_have_zero_width_and_empty_is_none(self):
        constant = bootstrap_mean_ci([0.5] * 8, 500)
        self.assertEqual((constant["ci_low"], constant["ci_high"]), (0.5, 0.5))
        self.assertIsNone(bootstrap_mean_ci([None], 100)["mean"])

    def test_paired_diff_drops_incomplete_pairs_and_checks_alignment(self):
        result = paired_bootstrap_diff([0.5, 0.6, None, 0.7], [0.1, 0.2, 0.9, 0.1], 1000)
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean_diff"], (0.4 + 0.4 + 0.6) / 3)
        self.assertGreater(result["ci_low"], 0.0)
        with self.assertRaises(ValueError):
            paired_bootstrap_diff([0.1], [0.1, 0.2])


class EvaluateVideoTests(unittest.TestCase):
    def test_uniform_scores_against_human_times(self):
        row = ev.evaluate_video("U", make_video(), 3, None, 2.0)
        self.assertEqual(row["times"], [25.0, 50.0, 75.0])
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (1, 2, 2))
        self.assertAlmostEqual(row["f1"], 2 / 6)
        self.assertEqual(row["f1_by_tolerance"]["0.0"], 2 / 6)  # 50.0 matches exactly
        self.assertIsNone(row["in_speech_rate"])

    def test_oracle_has_perfect_precision_and_budget_limited_recall(self):
        video = make_video(ad_times=range(10, 100, 10))
        row = ev.evaluate_video("O", video, 3, None, 2.0)
        self.assertEqual(row["precision"], 1.0)
        self.assertAlmostEqual(row["recall"], 3 / 9)
        full = ev.evaluate_video("O", make_video(), 5, None, 2.0)
        self.assertEqual((full["f1"], full["shortfall"]), (1.0, 2))

    def test_audio_selector_needs_features_and_avoids_speech(self):
        video = make_video()
        self.assertIsNone(ev.evaluate_video("A", video, 3, None, 2.0))
        features = make_features(speech=[(0.0, 40.0)], gaps=[(40.25, 100.0)])
        row = ev.evaluate_video("A", video, 3, features, 2.0)
        self.assertEqual(len(row["times"]), 3)
        self.assertTrue(all(t >= 40.0 for t in row["times"]))
        self.assertEqual(row["in_speech_rate"], 0.0)

    def test_in_speech_rate(self):
        features = make_features(speech=[(20.0, 30.0)], gaps=[])
        self.assertEqual(ev.in_speech_rate([25.0, 50.0], features, 100.0), 0.5)
        self.assertIsNone(ev.in_speech_rate([25.0], None, 100.0))
        self.assertIsNone(ev.in_speech_rate([], features, 100.0))


class SummaryTests(unittest.TestCase):
    def rows(self):
        video_a = make_video("a", ad_times=(10, 50, 90))
        video_b = make_video("b", ad_times=(25, 60))
        return [ev.evaluate_video("U", v, 3, None, 2.0) for v in (video_a, video_b)]

    def test_macro_and_micro_scores(self):
        rows = self.rows()
        summary = ev.summarize(rows, 500, 0)
        self.assertEqual(summary["n_videos"], 2)
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        self.assertAlmostEqual(summary["micro_f1"], 2 * tp / (2 * tp + fp + fn))
        self.assertAlmostEqual(
            summary["macro_f1"]["mean"], (rows[0]["f1"] + rows[1]["f1"]) / 2)
        self.assertEqual(summary["total_gt"], 5)
        self.assertEqual(set(summary["macro_f1_by_tolerance"]), {"0.0", "1.0", "2.0", "3.0", "5.0"})

    def test_compare_uses_only_shared_videos(self):
        video_a, video_b = make_video("a"), make_video("b")
        u = [ev.evaluate_video("U", v, 3, None, 2.0) for v in (video_a, video_b)]
        o = [ev.evaluate_video("O", video_a, 3, None, 2.0)]
        comparison = ev.compare({"U": u, "O": o}, 200, 0)
        self.assertEqual(len(comparison), 1)
        self.assertEqual((comparison[0]["selector"], comparison[0]["baseline"]), ("O", "U"))
        self.assertEqual(comparison[0]["n_shared_videos"], 1)
        self.assertGreater(comparison[0]["mean_diff"], 0.0)


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_selectors(self, videos, selectors, **kwargs):
        with fx.patched_audio_helpers():
            result = ev.run(videos, 3.48, selectors, self.root, 2.0, 200, 0, **kwargs)
        result["set"] = "dev"
        json.dumps(result)  # the whole result must be JSON serializable
        return result

    def valid_audio_cache(self, video_id):
        paths = fx.make_media_cache(self.root, video_id)
        fx.write_audio_features(
            paths["cache"], paths["manifest"], speech=[(0.0, 40.0)], gaps=[(40.25, 100.0)])
        return paths

    def test_end_to_end_counts_videos_without_usable_audio_features(self):
        videos = [make_video("a"), make_video("b")]
        self.valid_audio_cache("a")
        result = self.run_selectors(videos, ["U", "O", "A"])
        self.assertEqual(result["skipped"], {"A": 1})
        self.assertEqual(result["skipped_reasons"], {"A": {"no_manifest": 1}})
        self.assertEqual(result["unavailable"], [{"video_id": "b", "selector": "A", "reason": "no_manifest"}])
        self.assertEqual(result["audio_features"], {"usable": 1, "unavailable": {"no_manifest": 1}})
        self.assertEqual(result["selectors"]["A"]["n_videos"], 1)
        self.assertEqual(result["selectors"]["U"]["n_videos"], 2)
        self.assertEqual(result["selectors"]["A"]["in_speech_rate"], 0.0)
        report = ev.format_report(result)
        self.assertIn("macro F1", report)
        self.assertIn("A: skipped 1 video(s) without usable audio features (no_manifest x1)", report)
        self.assertIn("O - U", report)

    # Finding P2: U/O must not depend on an unrelated optional audio cache.
    def test_corrupt_audio_json_does_not_stop_u_and_o_scoring(self):
        videos = [make_video("a")]
        paths = self.valid_audio_cache("a")
        (paths["cache"] / "audio_features.json").write_text("{broken", encoding="utf-8")
        result = self.run_selectors(videos, ["U", "O"])
        self.assertEqual(result["selectors"]["U"]["n_videos"], 1)
        self.assertEqual(result["selectors"]["O"]["n_videos"], 1)
        self.assertIsNone(result["selectors"]["U"]["in_speech_rate"])  # diagnostic unavailable
        self.assertEqual(result["audio_features"]["unavailable"], {"unreadable_audio_features": 1})
        with_a = self.run_selectors(videos, ["U", "A"])
        self.assertEqual(with_a["skipped_reasons"], {"A": {"unreadable_audio_features": 1}})
        self.assertEqual(with_a["selectors"]["U"]["n_videos"], 1)

    # Review finding P2 (second round): a stamped, current cache whose payload would crash a consumer.
    def test_malformed_speech_and_gap_entries_do_not_stop_u_and_o_and_skip_a(self):
        for name, payload in (
            ("speech pair", {"raw_speech": [[1.0]]}),
            ("empty gap record", {"raw_gaps": [{}]}),
        ):
            with self.subTest(payload=name):
                root = Path(tempfile.mkdtemp(dir=self.root))
                paths = fx.make_media_cache(root, "a")
                fx.write_audio_features(paths["cache"], paths["manifest"], **payload)
                with fx.patched_audio_helpers():
                    result = ev.run([make_video("a")], 3.48, ["U", "O", "A"], root, 2.0, 100, 0)
                self.assertEqual(result["selectors"]["U"]["n_videos"], 1)
                self.assertEqual(result["selectors"]["O"]["n_videos"], 1)
                self.assertNotIn("A", result["selectors"])
                self.assertEqual(result["skipped_reasons"], {"A": {"malformed_audio_features": 1}})
                self.assertIsNone(result["selectors"]["U"]["in_speech_rate"])  # diagnostic unavailable

    # Finding P1, scenario 1: a rebuild failed and left a complete cache for an older source.
    def test_stale_media_cache_is_never_scored_even_with_features_present(self):
        videos = [make_video("a")]
        paths = self.valid_audio_cache("a")
        paths["source"].write_bytes(b"the replacement source video, which failed to preprocess")
        result = self.run_selectors(videos, ["U", "A"])
        self.assertNotIn("A", result["selectors"])
        self.assertEqual(result["skipped_reasons"], {"A": {"media_cache_stale": 1}})
        self.assertIsNone(result["selectors"]["U"]["in_speech_rate"])  # not even as a diagnostic

    # Finding P1, scenario 2: the WAV was replaced while the manifest stayed current.
    def test_replaced_wav_makes_features_stale_and_unscored(self):
        videos = [make_video("a")]
        paths = self.valid_audio_cache("a")
        paths["wav"].write_bytes(b"RIFF....a replacement recording that is longer")
        result = self.run_selectors(videos, ["A"])
        self.assertEqual(result["skipped_reasons"], {"A": {"stale_audio_features": 1}})
        self.assertEqual(result["selectors"], {})

    def test_features_are_not_trusted_without_the_fingerprint_helpers(self):
        videos = [make_video("a")]
        self.valid_audio_cache("a")
        with patch.object(cc, "features_config_hash", None), patch.object(cc, "features_are_current", None):
            result = ev.run(videos, 3.48, ["U", "A"], self.root, 2.0, 200, 0)
        self.assertEqual(result["skipped_reasons"], {"A": {"unverifiable_audio_features": 1}})

    def test_incomplete_media_cache_is_reported(self):
        videos = [make_video("a")]
        paths = self.valid_audio_cache("a")
        paths["frame"].unlink()
        result = self.run_selectors(videos, ["A"])
        self.assertEqual(result["skipped_reasons"], {"A": {"media_cache_incomplete": 1}})


class BoundsPolicyTests(unittest.TestCase):
    """AD starts beyond the declared duration: explicit policy, never silent."""

    def video(self, duration=10.0, times=(1.0, 12.0)):
        return make_video("v", duration, times)

    def test_ground_truth_times_policies_and_slack(self):
        video = self.video(times=(1.0, 10.9, 11.5))
        self.assertEqual(ev.ground_truth_times(video, "exclude"), ([1.0, 10.9], [11.5]))
        self.assertEqual(ev.ground_truth_times(video, "keep"), ([1.0, 10.9, 11.5], [11.5]))
        self.assertEqual(ev.ground_truth_times(video, "exclude", slack=2.0)[1], [])
        with self.assertRaises(ValueError):
            ev.ground_truth_times(video, "clamp")

    def test_default_excludes_impossible_starts_from_gt_for_every_selector(self):
        video = self.video()
        oracle = ev.evaluate_video("O", video, 2, None, 2.0)
        self.assertEqual(oracle["times"], [1.0])  # not credited with the 12.0 outside the clip
        self.assertEqual((oracle["num_gt"], oracle["num_gt_scored"], oracle["num_gt_out_of_bounds"]), (2, 1, 1))
        self.assertEqual(oracle["out_of_bounds_gt_times"], [12.0])
        self.assertEqual(oracle["num_selected_out_of_bounds"], 0)
        uniform = ev.evaluate_video("U", video, 2, None, 2.0)
        self.assertEqual(uniform["num_gt_scored"], 1)  # U is scored against the same ground truth
        self.assertEqual(uniform["fn"] + uniform["tp"], 1)

    def test_keep_policy_is_the_unrestricted_upper_bound_and_is_labelled(self):
        video = self.video()
        oracle = ev.evaluate_video("O", video, 2, None, 2.0, bounds="keep")
        self.assertEqual(oracle["times"], [1.0, 12.0])
        self.assertEqual(oracle["num_selected_out_of_bounds"], 1)
        self.assertEqual(oracle["f1"], 1.0)
        result = ev.run([video], 3.48, ["O"], Path("/nonexistent"), 2.0, 100, 0, bounds="keep")
        self.assertTrue(result["bounds"]["unrestricted_annotation_upper_bound"])
        self.assertEqual(result["bounds"]["selected_out_of_bounds"], {"O": 1})
        result["set"] = "dev"
        self.assertIn("INCLUDED: unrestricted annotation upper bound", ev.format_report(result))

    def test_run_reports_affected_videos_and_counts_without_changing_k(self):
        videos = [make_video("ok", 100.0, (10.0, 50.0)), make_video("bad", 10.0, (1.0, 12.0, 15.0))]
        result = ev.run(videos, 3.48, ["U", "O"], Path("/nonexistent"), 2.0, 100, 0)
        result["set"] = "dev"
        bounds = result["bounds"]
        self.assertEqual(bounds["policy"], "exclude")
        self.assertEqual([v["video_id"] for v in bounds["videos"]], ["bad"])
        self.assertEqual(bounds["videos"][0]["out_of_bounds_gt_times"], [12.0, 15.0])
        self.assertEqual(bounds["total_gt_out_of_bounds"], 2)
        self.assertEqual(result["selectors"]["O"]["total_gt"], 3)  # scored
        self.assertEqual(result["selectors"]["O"]["total_gt_annotated"], 5)  # annotated
        k_bad = next(r["k"] for r in result["videos"] if r["video_id"] == "bad" and r["selector"] == "U")
        self.assertEqual(k_bad, 1)  # K still comes from the declared duration
        report = ev.format_report(result)
        self.assertIn("2 in 1 video(s) (bad) EXCLUDED from the ground truth for every selector", report)

    def test_no_anomalies_is_stated(self):
        result = ev.run([make_video("ok")], 3.48, ["U"], Path("/nonexistent"), 2.0, 100, 0)
        result["set"] = "dev"
        self.assertIn("Out-of-duration AD starts: none", ev.format_report(result))


if __name__ == "__main__":
    unittest.main()
