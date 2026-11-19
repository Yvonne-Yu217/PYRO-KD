"""Placement metrics, significance tests, audio diagnostics, QA scoring, the
five-criterion judge rubric and the CLIP text similarity wrapper."""

import json
from pathlib import Path
import random
import tempfile
import unittest

from src.evaluation import audio_diagnostics as ad
from src.evaluation import clip_text, judge_rubric, placement, qa, significance

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None


class PlacementTests(unittest.TestCase):
    def test_rendered_duration_and_overlap(self):
        self.assertAlmostEqual(placement.rendered_seconds(9), 3.0)  # 9 / (2.5 * 1.2)
        out = placement.speech_overlap([(0.0, 9), (10.0, 3)], [(2.0, 4.0)])
        self.assertAlmostEqual(out["overlap_seconds"], 1.0)
        self.assertEqual(out["ads_with_overlap"], 1)
        self.assertEqual(out["per_ad_overlap_seconds"], [1.0, 0.0])

    def test_budget_placements_never_overlap(self):
        speech = [(5.0, 9.0), (12.0, 20.0)]
        placements = placement.budget_placements([0.0, 9.5, 11.9, 30.0], speech, 40.0)
        self.assertEqual(placement.speech_overlap(placements, speech)["overlap_seconds"], 0.0)

    def test_silence_utilization(self):
        out = placement.silence_utilization([(0.0, 9), (6.0, 3)], [(5.0, 9.0)], 20.0)
        self.assertAlmostEqual(out["per_ad"][0], 3.0 / 5.0)
        self.assertIsNone(out["per_ad"][1])
        self.assertEqual(out["n_in_speech"], 1)

    def test_diagnostics_detect_clustering(self):
        clustered = [118.75 + 2 * i for i in range(17)]
        diag = placement.placement_diagnostics(clustered, 300.0, [(0, 118.5), (151, 300)], [(118.75, 150.75)])
        self.assertEqual(diag["busiest_gap_share"], 1.0)
        self.assertGreater(diag["densest_decile_share"], 0.9)  # 16 of 17 picks within 30 s
        self.assertLess(diag["span_fraction"], 0.2)
        spread = placement.placement_diagnostics([30, 90, 150, 210, 270], 300.0, None, None)
        self.assertAlmostEqual(spread["densest_decile_share"], 0.2)
        self.assertIsNone(spread["picks_in_speech"])

    def test_gt_in_gap_and_svg(self):
        self.assertEqual(placement.gt_in_gap_fraction([1.0, 6.0], [(5.0, 9.0)], 20.0), 0.5)
        svg = placement.timeline_svg([{"label": "v <A>", "duration": 10, "speech": [(1, 2)], "gt": [3], "times": [4]}])
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("v &lt;A&gt;", svg)


class SignificanceTests(unittest.TestCase):
    def test_exact_small_sample_matches_scipy(self):
        a = [0.5, 0.7, 0.2, 0.9, 0.4, 0.66, 0.31]
        b = [0.4, 0.5, 0.25, 0.6, 0.38, 0.6, 0.1]
        ours = significance.wilcoxon_signed_rank(a, b)
        self.assertEqual(ours["method"], "exact")
        if scipy_stats is not None:
            ref = scipy_stats.wilcoxon(a, b)
            self.assertAlmostEqual(ours["p_value"], ref.pvalue, places=10)

    def test_normal_approximation_with_ties_close_to_scipy(self):
        rng = random.Random(0)
        a = [round(rng.random(), 1) for _ in range(40)]
        b = [round(rng.random() * 0.9, 1) for _ in range(40)]
        ours = significance.wilcoxon_signed_rank(a, b)
        self.assertTrue(ours["method"].startswith("normal"))
        if scipy_stats is not None:
            ref = scipy_stats.wilcoxon(a, b, correction=True, method="approx")
            self.assertAlmostEqual(ours["p_value"], ref.pvalue, places=6)

    def test_degenerate_and_none_pairs(self):
        out = significance.wilcoxon_signed_rank([1, 2, None], [1, 2, 3])
        self.assertEqual(out["n_pairs"], 2)
        self.assertIsNone(out["p_value"])
        with self.assertRaises(ValueError):
            significance.wilcoxon_signed_rank([1], [1, 2])

    def test_holm_and_mcnemar(self):
        self.assertEqual(significance.holm([0.01, None, 0.04, 0.03]), [0.03, None, 0.06, 0.06])
        m = significance.mcnemar_exact([True] * 8 + [False] * 2, [False] * 8 + [True] * 2)
        self.assertEqual((m["only_a"], m["only_b"]), (8, 2))
        if scipy_stats is not None:
            self.assertAlmostEqual(m["p_value"], scipy_stats.binomtest(2, 10, 0.5).pvalue, places=10)


class AudioDiagnosticTests(unittest.TestCase):
    def test_hand_labels_and_detector_pr(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            path.write_text("\n".join([
                json.dumps({"video_id": "v", "time": 10.0, "label": "Music"}),
                json.dumps({"video_id": "v", "time": 30.0, "superclass": "impulsive"}),
                "{bad json",
                json.dumps({"video_id": "v", "time": -1, "label": "Music"}),
            ]))
            labels, problems = ad.load_hand_labels(path)
        self.assertEqual(len(labels["v"]), 2)
        self.assertEqual(len(problems), 2)
        predicted = {"v": [{"time": 10.4, "superclass": "music"}, {"time": 50.0, "superclass": "music"}]}
        out = ad.detector_precision_recall(predicted, labels, tolerance=1.0)
        self.assertEqual(out["per_class"]["music"]["precision"], 0.5)
        self.assertEqual(out["per_class"]["music"]["recall"], 1.0)
        self.assertEqual(out["per_class"]["impulsive"]["recall"], 0.0)
        self.assertEqual(out["pooled"]["tp"], 1)

    def test_trigger_suppress_lift(self):
        videos = [{"video_id": "v", "duration": 100.0, "gt_times": [10.0, 50.0, 90.0, 70.0],
                   "events": [{"time": 10.5, "superclass": "music"}, {"time": 50.0, "superclass": "music"},
                              {"time": 30.0, "superclass": "impulsive"}]}]
        out = ad.trigger_suppress(videos, window=2.0)
        music = out["classes"]["music"]
        self.assertEqual(music["hit_rate"], 0.5)
        self.assertGreater(music["lift"], 5)
        self.assertEqual(out["classes"]["impulsive"]["hit_rate"], 0.0)

    def test_silent_coverage(self):
        grid = [i * 0.5 for i in range(201)]
        change = [1.0 if 19 <= g <= 21 else 0.0 for g in grid]
        silent = ad.silent_gt_times([20.0, 60.0], grid, change, 0.2, 2.0)
        self.assertEqual(silent, [60.0])
        cov = ad.silent_visual_coverage({"U": [10.0], "V": [60.5]}, silent, [20.0, 60.0])
        self.assertEqual(cov["silent_fraction"], 0.5)
        self.assertEqual(cov["recall_on_silent"], {"U": 0.0, "V": 1.0})


class QATests(unittest.TestCase):
    def test_parse_choice(self):
        opts = ["a cat", "a dog", "a bird"]
        self.assertEqual(qa.parse_choice("B", opts), 1)
        self.assertEqual(qa.parse_choice("(C)", opts), 2)
        self.assertEqual(qa.parse_choice("The answer is A.", opts), 0)
        self.assertEqual(qa.parse_choice("a dog", opts), 1)
        self.assertIsNone(qa.parse_choice("E", opts))
        self.assertIsNone(qa.parse_choice("A man walks his dog", opts))
        self.assertIsNone(qa.parse_choice(None, opts))

    def test_scores_bins_and_paired(self):
        rows_on = [{"question_id": i, "prediction": 0, "answer": 0 if i < 8 else 1, "duration": 100 * i} for i in range(10)]
        rows_off = [{"question_id": i, "prediction": 0 if i < 5 else None, "answer": 0 if i < 8 else 1,
                     "duration": 100 * i} for i in range(10)]
        self.assertAlmostEqual(qa.score_items(rows_on)["accuracy"]["mean"], 0.8)
        self.assertEqual(qa.score_items(rows_off)["n_unparsed"], 5)
        bins = qa.accuracy_by_length(rows_on, edges=(500,))
        self.assertEqual([b["n"] for b in bins], [5, 5])
        cmp = qa.paired_condition_comparison(rows_on, rows_off)
        self.assertEqual(cmp["n_shared"], 10)
        self.assertEqual(cmp["mcnemar"]["only_a"], 3)


class JudgeTests(unittest.TestCase):
    def test_valid_reply(self):
        reply = 'Sure: {"objectivity": 4, "concision_gap_fit": 5, "non_redundancy": 3, "specificity": 4, "grounding": 4.5, "rationale": "ok"}'
        prompts = []
        out = judge_rubric.judge_description("A man opens a door.", lambda p: (prompts.append(p), reply)[1],
                                             word_limit=6, audio_timeline="[0:01 door slam]")
        self.assertAlmostEqual(out["mean_score"], 4.1)
        self.assertIn("[0:01 door slam]", prompts[0])
        self.assertFalse(out["visual_accuracy_verified"])

    def test_invalid_replies(self):
        for reply in ("no json", '{"objectivity": 7}', '{"objectivity": 4, "concision_gap_fit": 4, "non_redundancy": 4, '
                      '"specificity": 4, "grounding": true, "rationale": "x"}'):
            with self.assertRaises(judge_rubric.JudgeError):
                judge_rubric.judge_description("x", lambda p, r=reply: r)
        with self.assertRaises(judge_rubric.JudgeError):
            judge_rubric.judge_description("x", lambda p: (_ for _ in ()).throw(RuntimeError("cap")))

    def test_custom_criteria(self):
        out = judge_rubric.judge_description("x", lambda p: '{"clarity": 2, "rationale": "r"}', criteria={"clarity": "c"})
        self.assertEqual(out["criteria"], ["clarity"])


class ClipTextTests(unittest.TestCase):
    def test_pair_similarities(self):
        enc = {"a": [1.0, 0.0], "b": [0.0, 2.0], "c": [3.0, 0.0]}
        sims = clip_text.pair_similarities(["a", "b"], ["c", "c"], lambda texts: [enc[t] for t in texts])
        self.assertEqual(sims, [1.0, 0.0])
        with self.assertRaises(ValueError):
            clip_text.pair_similarities(["a"], [], lambda t: [])


if __name__ == "__main__":
    unittest.main()
