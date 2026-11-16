import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from src.datasets.mtg_ad import parse_row
from src.evaluation.stats import bootstrap_mean_ci, paired_bootstrap_diff
from src.utils import visual_features as vf
from src.utils.selection import oracle_timestamps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
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


def make_visual(spikes, duration=100.0):
    times = [i * 0.5 for i in range(int(duration / 0.5))]
    novelty = [0.05] * len(times)
    for t, value in spikes:
        novelty[times.index(t)] = value
    return {"frame_times": times, "novelty": novelty, "shots": []}


class VisualSelectorEvalTests(unittest.TestCase):
    def test_v_needs_visual_features_and_scores_its_peaks(self):
        video = make_video(ad_times=(10.0, 50.0, 90.0))
        self.assertIsNone(ev.evaluate_video("V", video, 3, None, 2.0))
        visual = make_visual([(10.0, 1.0), (50.0, 0.9), (90.0, 0.8)])
        row = ev.evaluate_video("V", video, 3, None, 2.0, visual=visual)
        self.assertEqual(row["times"], [10.0, 50.0, 90.0])
        self.assertEqual(row["f1"], 1.0)
        self.assertIsNone(row["visual_redundancy"])  # no embeddings supplied

    def test_redundancy_is_reported_when_embeddings_exist(self):
        video = make_video(ad_times=(10.0, 50.0))
        visual = make_visual([(10.0, 1.0), (50.0, 0.9)])
        embeddings = np.tile([1.0, 0.0], (len(visual["frame_times"]), 1))
        row = ev.evaluate_video("V", video, 2, None, 2.0, visual=visual, embeddings=embeddings)
        self.assertAlmostEqual(row["visual_redundancy"], 1.0)

    def test_run_loads_visual_cache_and_counts_skips(self):
        videos = [make_video("a"), make_video("b")]
        visual = make_visual([(10.0, 1.0), (50.0, 0.9), (90.0, 0.8)])
        embeddings = np.eye(len(visual["frame_times"]))[:, :4]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "a").mkdir()
            vf.write_features(cache / "a", visual, embeddings)
            result = ev.run(videos, 3.48, ["U", "V"], cache, 2.0, 200, 0)
        result["set"] = "dev"
        self.assertEqual(result["skipped"], {"V": 1})
        self.assertEqual(result["selectors"]["V"]["n_videos"], 1)
        self.assertEqual(result["selectors"]["V"]["n_with_embeddings"], 1)
        self.assertIn("V: skipped 1 video(s) without visual features", ev.format_report(result))
        json.dumps(result)


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
    def test_end_to_end_counts_videos_without_audio_features(self):
        videos = [make_video("a"), make_video("b")]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / "a").mkdir()
            (cache / "a" / "audio_features.json").write_text(json.dumps(
                make_features(speech=[(0.0, 40.0)], gaps=[(40.25, 100.0)])))
            result = ev.run(videos, 3.48, ["U", "O", "A"], cache, 2.0, 200, 0)
        result["set"] = "dev"
        self.assertEqual(result["skipped"], {"A": 1})
        self.assertEqual(result["selectors"]["A"]["n_videos"], 1)
        self.assertEqual(result["selectors"]["U"]["n_videos"], 2)
        self.assertEqual(result["selectors"]["A"]["in_speech_rate"], 0.0)
        report = ev.format_report(result)
        self.assertIn("macro F1", report)
        self.assertIn("A: skipped 1 video(s) without audio features", report)
        self.assertIn("O - U", report)
        json.dumps(result)  # the whole result must be JSON serializable


if __name__ == "__main__":
    unittest.main()
