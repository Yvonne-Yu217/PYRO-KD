"""Offline BERTScore tests with deterministic scorers and no model downloads."""

from dataclasses import FrozenInstanceError
import json
import math
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from src.evaluation.bertscore import GeneratedADItem, compute_bertscore
from src.evaluation.metrics import TemporalMatch, match_timestamps
from src.evaluation.parsing import ADItem


class BERTScoreTests(unittest.TestCase):
    def setUp(self) -> None:
        # Deliberately unsorted, with unmatched descriptions on both sides.
        self.generated = [
            GeneratedADItem(20.5, "A dog runs."),
            GeneratedADItem(100.0, "Unmatched generated text."),
            GeneratedADItem(10.5, "A woman waves."),
        ]
        self.gt = [
            ADItem(10.0, "A woman waves at the camera."),
            ADItem(200.0, "Unmatched professional text."),
            ADItem(20.0, "A dog runs across a field."),
        ]
        self.matches = match_timestamps(
            [item.timestamp for item in self.generated],
            [item.timestamp for item in self.gt],
        )

    def test_matched_indices_select_texts_and_exclude_unmatched_entries(self) -> None:
        scorer = Mock(return_value=(([0.8, 0.6], [0.5, 0.9], [0.7, 0.5]), "mock-config"))
        result = compute_bertscore(self.generated, self.gt, self.matches, scorer=scorer)

        scorer.assert_called_once_with(
            ["A dog runs.", "A woman waves."],
            ["A dog runs across a field.", "A woman waves at the camera."],
            lang="en",
            model_type=None,
            return_hash=True,
            idf=False,
            rescale_with_baseline=False,
            use_fast_tokenizer=False,
        )
        self.assertEqual(result["num_scored_pairs"], 2)
        self.assertAlmostEqual(result["precision_mean"], 0.7)
        self.assertAlmostEqual(result["recall_mean"], 0.7)
        self.assertAlmostEqual(result["f1_mean"], 0.6)
        self.assertEqual(
            result["pairs"],
            [
                {"prediction_index": 0, "ground_truth_index": 2,
                 "precision": 0.8, "recall": 0.5, "f1": 0.7},
                {"prediction_index": 2, "ground_truth_index": 0,
                 "precision": 0.6, "recall": 0.9, "f1": 0.5},
            ],
        )
        # Means are over scored pairs, without temporal-recall scaling.
        self.assertEqual(result["configuration_hash"], "mock-config")
        self.assertEqual(result["lang"], "en")
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_match_order_is_preserved(self) -> None:
        scorer = Mock(return_value=(([0.1, 0.2], [0.3, 0.4], [0.5, 0.6]), "config"))
        result = compute_bertscore(
            self.generated, self.gt, list(reversed(self.matches)), scorer=scorer,
        )
        self.assertEqual(scorer.call_args.args[0], ["A woman waves.", "A dog runs."])
        self.assertEqual([pair["prediction_index"] for pair in result["pairs"]], [2, 0])
        self.assertEqual(result["pairs"][0]["f1"], 0.5)

    def test_one_pair_and_explicit_configuration(self) -> None:
        scorer = Mock(return_value=(([0.75], [0.5], [0.6]), "explicit-model-config"))
        result = compute_bertscore(
            self.generated, self.gt, self.matches[:1],
            lang="de", model_type="bert-base-multilingual-cased", scorer=scorer,
        )
        self.assertEqual(result["num_scored_pairs"], 1)
        self.assertEqual(result["precision_mean"], 0.75)
        self.assertEqual(result["recall_mean"], 0.5)
        self.assertEqual(result["f1_mean"], 0.6)
        self.assertEqual(result["model_type"], "bert-base-multilingual-cased")
        self.assertEqual(result["lang"], "de")
        self.assertEqual(scorer.call_args.kwargs["model_type"], result["model_type"])
        self.assertEqual(scorer.call_args.kwargs["lang"], "de")

    def test_zero_matches_never_calls_scorer(self) -> None:
        scorer = Mock(side_effect=AssertionError("must not score empty matches"))
        for generated, gt in ((self.generated, self.gt), ([], [])):
            with self.subTest(empty_inputs=not generated):
                result = compute_bertscore(generated, gt, [], scorer=scorer)
                self.assertEqual(result["num_scored_pairs"], 0)
                for key in ("precision_mean", "recall_mean", "f1_mean"):
                    self.assertIsNone(result[key])
                self.assertEqual(result["pairs"], [])
                self.assertIsNone(result["configuration_hash"])
        scorer.assert_not_called()

    def test_zero_matches_does_not_import_bert_score(self) -> None:
        with patch.dict(sys.modules, {"bert_score": None, "bert_score.utils": None}):
            result = compute_bertscore(self.generated, self.gt, [])
        self.assertEqual(result["num_scored_pairs"], 0)
        self.assertIsNone(result["model_type"])

    def test_default_path_calls_official_api_and_records_resolved_model(self) -> None:
        package = ModuleType("bert_score")
        package.score = Mock(  # type: ignore[attr-defined]
            return_value=(([0.8, 0.6], [0.5, 0.9], [0.7, 0.5]), "official-config")
        )
        utils = ModuleType("bert_score.utils")
        utils.lang2model = {"en": "mock-english-model"}  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"bert_score": package, "bert_score.utils": utils}):
            result = compute_bertscore(self.generated, self.gt, self.matches)
        self.assertEqual(result["model_type"], "mock-english-model")
        self.assertEqual(result["configuration_hash"], "official-config")
        package.score.assert_called_once_with(  # type: ignore[attr-defined]
            ["A dog runs.", "A woman waves."],
            ["A dog runs across a field.", "A woman waves at the camera."],
            lang="en", model_type="mock-english-model", return_hash=True,
            idf=False, rescale_with_baseline=False, use_fast_tokenizer=False,
        )
        with patch.dict(sys.modules, {"bert_score": package, "bert_score.utils": utils}):
            explicit = compute_bertscore(
                self.generated, self.gt, self.matches, model_type="explicit-model",
            )
        self.assertEqual(explicit["model_type"], "explicit-model")
        self.assertEqual(package.score.call_args.kwargs["model_type"], "explicit-model")  # type: ignore[attr-defined]

    def test_invalid_or_reused_indices_fail_before_scoring(self) -> None:
        scorer = Mock()
        invalid = (
            ([TemporalMatch(-1, 0, 10.5, 10, 0.5)], "prediction_index"),
            ([TemporalMatch(3, 0, 10.5, 10, 0.5)], "prediction_index"),
            ([TemporalMatch(2, -1, 10.5, 10, 0.5)], "ground_truth_index"),
            ([TemporalMatch(2, 3, 10.5, 10, 0.5)], "ground_truth_index"),
            ([self.matches[0], self.matches[0]], "one-to-one"),
            ([self.matches[0], TemporalMatch(0, 0, 20.5, 10, 10.5)], "one-to-one"),
            ([self.matches[0], TemporalMatch(2, 2, 10.5, 20, 9.5)], "one-to-one"),
        )
        for matches, message in invalid:
            with self.subTest(matches=matches):
                with self.assertRaisesRegex(ValueError, message):
                    compute_bertscore(self.generated, self.gt, matches, scorer=scorer)
        scorer.assert_not_called()

    def test_stale_matches_fail_before_scoring(self) -> None:
        scorer = Mock()
        for generated, gt in (
            (list(reversed(self.generated)), self.gt),
            (self.generated, list(reversed(self.gt))),
        ):
            with self.subTest(generated=generated, gt=gt):
                with self.assertRaisesRegex(ValueError, "timestamps disagree"):
                    compute_bertscore(generated, gt, self.matches, scorer=scorer)
        scorer.assert_not_called()

    def test_missing_matched_text_is_rejected_but_unmatched_text_is_ignored(self) -> None:
        scorer = Mock(return_value=(([0.8], [0.5], [0.6]), "config"))
        for candidate, reference in (("", "Reference."), ("Candidate.", " \n")):
            with self.subTest(candidate=candidate, reference=reference):
                with self.assertRaisesRegex(ValueError, "text must be a nonempty string"):
                    compute_bertscore(
                        [GeneratedADItem(10, candidate)], [ADItem(10, reference)],
                        [TemporalMatch(0, 0, 10, 10, 0)], scorer=scorer,
                    )
        scorer.assert_not_called()
        result = compute_bertscore(
            [GeneratedADItem(10, "Candidate."), GeneratedADItem(100, "")],
            [ADItem(10, "Reference."), ADItem(200, "")],
            [TemporalMatch(0, 0, 10, 10, 0)], scorer=scorer,
        )
        self.assertEqual(result["num_scored_pairs"], 1)

    def test_invalid_scorer_output_is_not_silently_truncated(self) -> None:
        for values in (
            ([0.8], [0.5, 0.9], [0.7, 0.5]),
            ([0.8, 0.6], [math.nan, 0.9], [0.7, 0.5]),
            ([0.8, 0.6], [0.5, 0.9], [math.inf, 0.5]),
        ):
            with self.subTest(values=values):
                scorer = Mock(return_value=(values, "config"))
                with self.assertRaisesRegex(ValueError, "BERTScore"):
                    compute_bertscore(self.generated, self.gt, self.matches, scorer=scorer)

    def test_scorer_failure_propagates_without_fabricated_scores(self) -> None:
        scorer = Mock(side_effect=RuntimeError("Model unavailable"))
        with self.assertRaisesRegex(RuntimeError, "Model unavailable"):
            compute_bertscore(self.generated, self.gt, self.matches, scorer=scorer)

    def test_generated_item_is_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.generated[0].text = "Changed."  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
