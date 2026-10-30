"""Offline cosine evaluation tests; no models or third-party imports needed."""

import json
import math
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

from src.evaluation.bertscore import GeneratedADItem
from src.evaluation.cosine_similarity import (
    DEFAULT_MODEL_NAME,
    compute_cosine_similarity,
)
from src.evaluation.metrics import TemporalMatch, match_timestamps
from src.evaluation.parsing import ADItem


class CosineSimilarityTests(unittest.TestCase):
    def setUp(self) -> None:
        # Original order deliberately differs from timestamp order.
        self.generated = [
            GeneratedADItem(20.5, "A dog runs."),
            GeneratedADItem(100.0, "Unmatched generated text."),
            GeneratedADItem(10.5, "A woman waves."),
            GeneratedADItem(30.5, "A sign appears."),
        ]
        self.gt = [
            ADItem(10.0, "A woman waves at the camera."),
            ADItem(200.0, "Unmatched professional text."),
            ADItem(30.0, "An OPEN sign appears."),
            ADItem(20.0, "A dog runs across a field."),
        ]
        self.matches = match_timestamps(
            [item.timestamp for item in self.generated],
            [item.timestamp for item in self.gt],
        )

    def test_original_indices_select_only_matched_texts(self) -> None:
        # Candidate rows followed by reference rows: similarities 1, 0, 0.6.
        encoder = Mock()
        encoder.encode.return_value = [
            [1, 0], [1, 0], [1, 0],
            [1, 0], [0, 1], [0.6, 0.8],
        ]
        result = compute_cosine_similarity(
            self.generated, self.gt, self.matches, encoder=encoder,
        )
        encoder.encode.assert_called_once_with(
            [
                "A dog runs.", "A woman waves.", "A sign appears.",
                "A dog runs across a field.", "A woman waves at the camera.",
                "An OPEN sign appears.",
            ],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        self.assertEqual(result["num_scored_pairs"], 3)
        self.assertEqual(
            result["pairs"],
            [
                {"prediction_index": 0, "ground_truth_index": 3, "cosine_similarity": 1.0},
                {"prediction_index": 2, "ground_truth_index": 0, "cosine_similarity": 0.0},
                {"prediction_index": 3, "ground_truth_index": 2, "cosine_similarity": 0.6},
            ],
        )
        self.assertAlmostEqual(result["cosine_similarity_mean"], 1.6 / 3)
        self.assertAlmostEqual(result["cosine_similarity_median"], 0.6)
        self.assertEqual(result["model_name"], DEFAULT_MODEL_NAME)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_single_pair_identical_normalized_vectors(self) -> None:
        encoder = Mock()
        encoder.encode.return_value = [[0.6, 0.8], [0.6, 0.8]]
        result = compute_cosine_similarity(
            self.generated, self.gt, self.matches[:1], encoder=encoder,
        )
        self.assertEqual(result["num_scored_pairs"], 1)
        self.assertAlmostEqual(result["pairs"][0]["cosine_similarity"], 1.0)
        self.assertAlmostEqual(result["cosine_similarity_mean"], 1.0)
        self.assertAlmostEqual(result["cosine_similarity_median"], 1.0)

    def test_single_pair_orthogonal_normalized_vectors(self) -> None:
        encoder = Mock()
        encoder.encode.return_value = [[1, 0], [0, 1]]
        result = compute_cosine_similarity(
            self.generated, self.gt, self.matches[:1], encoder=encoder,
        )
        self.assertEqual(result["pairs"][0]["cosine_similarity"], 0.0)
        self.assertEqual(result["cosine_similarity_mean"], 0.0)
        self.assertEqual(result["cosine_similarity_median"], 0.0)

    def test_even_count_median_and_negative_similarity(self) -> None:
        encoder = Mock()
        encoder.encode.return_value = [[1, 0], [1, 0], [-1, 0], [0.6, 0.8]]
        result = compute_cosine_similarity(
            self.generated, self.gt, self.matches[:2], encoder=encoder,
        )
        self.assertEqual(result["pairs"][0]["cosine_similarity"], -1.0)
        self.assertAlmostEqual(result["cosine_similarity_mean"], -0.2)
        self.assertAlmostEqual(result["cosine_similarity_median"], -0.2)

    def test_injected_unnormalized_vectors_are_still_cosine(self) -> None:
        encoder = Mock()
        encoder.encode.return_value = [[3, 4], [6, 8]]
        result = compute_cosine_similarity(
            self.generated, self.gt, self.matches[:1], encoder=encoder,
        )
        self.assertAlmostEqual(result["cosine_similarity_mean"], 1.0)

    def test_match_order_is_preserved(self) -> None:
        encoder = Mock()
        encoder.encode.return_value = [[1, 0]] * 6
        result = compute_cosine_similarity(
            self.generated, self.gt, list(reversed(self.matches)), encoder=encoder,
        )
        self.assertEqual(
            encoder.encode.call_args.args[0][:3],
            ["A sign appears.", "A woman waves.", "A dog runs."],
        )
        self.assertEqual([pair["prediction_index"] for pair in result["pairs"]], [3, 2, 0])

    def test_zero_matches_does_not_import_or_instantiate_model(self) -> None:
        package = ModuleType("sentence_transformers")
        constructor = Mock(side_effect=AssertionError("must not instantiate a model"))
        package.SentenceTransformer = constructor  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"sentence_transformers": package}):
            for generated, gt in ((self.generated, self.gt), ([], [])):
                result = compute_cosine_similarity(generated, gt, [])
                self.assertEqual(result, {
                    "num_scored_pairs": 0,
                    "cosine_similarity_mean": None,
                    "cosine_similarity_median": None,
                    "pairs": [],
                    "model_name": DEFAULT_MODEL_NAME,
                })
        constructor.assert_not_called()
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            result = compute_cosine_similarity([], [], [])
        self.assertIsNone(result["cosine_similarity_mean"])

    def test_zero_matches_does_not_call_injected_encoder(self) -> None:
        encoder = Mock()
        result = compute_cosine_similarity(self.generated, self.gt, [], encoder=encoder)
        encoder.encode.assert_not_called()
        self.assertEqual(result["num_scored_pairs"], 0)
        self.assertIsNone(result["cosine_similarity_mean"])
        self.assertIsNone(result["cosine_similarity_median"])
        self.assertEqual(result["pairs"], [])

    def test_default_model_and_explicit_model_are_loaded_lazily(self) -> None:
        encoder = Mock()
        encoder.encode.return_value = [[1, 0], [1, 0]]
        constructor = Mock(return_value=encoder)
        package = ModuleType("sentence_transformers")
        package.SentenceTransformer = constructor  # type: ignore[attr-defined]
        for model_name in (DEFAULT_MODEL_NAME, "test/custom-model"):
            with self.subTest(model_name=model_name):
                with patch.dict(sys.modules, {"sentence_transformers": package}):
                    result = compute_cosine_similarity(
                        self.generated, self.gt, self.matches[:1], model_name=model_name,
                    )
                constructor.assert_called_with(model_name)
                self.assertEqual(result["model_name"], model_name)
                self.assertEqual(result["cosine_similarity_mean"], 1.0)

    def test_invalid_matches_fail_before_encoding(self) -> None:
        encoder = Mock()
        cases = (
            ([TemporalMatch(-1, 0, 10.5, 10, 0.5)], "prediction_index"),
            ([TemporalMatch(4, 0, 10.5, 10, 0.5)], "prediction_index"),
            ([TemporalMatch(2, -1, 10.5, 10, 0.5)], "ground_truth_index"),
            ([TemporalMatch(2, 4, 10.5, 10, 0.5)], "ground_truth_index"),
            ([self.matches[0], self.matches[0]], "one-to-one"),
            ([self.matches[0], TemporalMatch(2, 3, 10.5, 20, 9.5)], "one-to-one"),
            ([TemporalMatch(0, 0, 10.5, 10, 0.5)], "timestamps disagree"),
        )
        for matches, message in cases:
            with self.subTest(matches=matches):
                with self.assertRaisesRegex(ValueError, message):
                    compute_cosine_similarity(self.generated, self.gt, matches, encoder=encoder)
        encoder.encode.assert_not_called()

    def test_only_matched_text_must_be_nonempty(self) -> None:
        encoder = Mock()
        encoder.encode.return_value = [[1, 0], [1, 0]]
        match = TemporalMatch(0, 0, 10, 10, 0)
        for candidate, reference in (("", "Reference."), ("Candidate.", " \n")):
            with self.subTest(candidate=candidate, reference=reference):
                with self.assertRaisesRegex(ValueError, "text must be a nonempty string"):
                    compute_cosine_similarity(
                        [GeneratedADItem(10, candidate)], [ADItem(10, reference)],
                        [match], encoder=encoder,
                    )
        encoder.encode.assert_not_called()
        result = compute_cosine_similarity(
            [GeneratedADItem(10, "Candidate."), GeneratedADItem(100, "")],
            [ADItem(10, "Reference."), ADItem(200, "")], [match], encoder=encoder,
        )
        self.assertEqual(result["num_scored_pairs"], 1)

    def test_invalid_embeddings_do_not_produce_fabricated_scores(self) -> None:
        invalid = (
            ([[1, 0]], "one vector"),
            ([[1, 0], [1]], "same dimension"),
            ([[], []], "nonempty"),
            ([[0, 0], [1, 0]], "nonzero norm"),
            ([[math.nan, 0], [1, 0]], "finite numbers"),
            ([[1, 0], [math.inf, 0]], "finite numbers"),
        )
        for embeddings, message in invalid:
            with self.subTest(embeddings=embeddings):
                encoder = Mock()
                encoder.encode.return_value = embeddings
                with self.assertRaisesRegex(ValueError, message):
                    compute_cosine_similarity(
                        self.generated, self.gt, self.matches[:1], encoder=encoder,
                    )

    def test_encoder_failure_propagates(self) -> None:
        encoder = Mock()
        encoder.encode.side_effect = RuntimeError("Encoder unavailable")
        with self.assertRaisesRegex(RuntimeError, "Encoder unavailable"):
            compute_cosine_similarity(self.generated, self.gt, self.matches, encoder=encoder)


if __name__ == "__main__":
    unittest.main()
