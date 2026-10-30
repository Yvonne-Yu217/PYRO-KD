"""Offline tests: python -m unittest discover -s tests -v."""

from dataclasses import FrozenInstanceError
from itertools import product
import json
import math
import unittest

from src.evaluation.metrics import (
    TemporalMatch,
    match_timestamps,
    temporal_precision_recall_f1,
    temporal_tolerance_sweep,
)
from src.evaluation.parsing import ADItem, parse_mtg_ad


class ParsingTests(unittest.TestCase):
    def test_unicode_line_separators(self) -> None:
        raw = (
            "\n00:00:10.043-STANDARD\u2028She closes her eyes.\n"
            "00:00:21.671-STANDARD\u2029She models her outfit.\n\n"
        )
        self.assertEqual(
            parse_mtg_ad(raw),
            [
                ADItem(10.043, "She closes her eyes."),
                ADItem(21.671, "She models her outfit."),
            ],
        )

    def test_supported_timestamp_formats(self) -> None:
        cases = {
            "1:02:03.250": 3723.25,
            "00:00:10.043": 10.043,
            "1:05.250": 65.25,
            "1:05": 65.0,
            "0:17:460": 17.46,
            "0:17:005": 17.005,
            "0:17.5": 17.5,
            "0:17,460": 17.46,
            "1:02:03,250": 3723.25,
            "75:05.000": 4505.0,
            "0:17.1234": 17.1234,
        }
        for timestamp, expected in cases.items():
            with self.subTest(timestamp=timestamp):
                item, = parse_mtg_ad(f"{timestamp}-STANDARD A sign appears.")
                self.assertAlmostEqual(item.timestamp, expected)
                self.assertEqual(item.text, "A sign appears.")

    def test_clean_text_and_untagged_lines(self) -> None:
        self.assertEqual(
            parse_mtg_ad(
                '  0:02.000 - STANDARD  A red-blue sign says "OPEN".  \r\n'
                "\t\n1:05 A woman waves.\n"
            ),
            [
                ADItem(2.0, 'A red-blue sign says "OPEN".'),
                ADItem(65.0, "A woman waves."),
            ],
        )

    def test_preserves_order_and_duplicate_gt_entries(self) -> None:
        self.assertEqual(
            parse_mtg_ad("0:20-STANDARD Later.\n0:10-STANDARD First.\n0:10-STANDARD Second."),
            [ADItem(20.0, "Later."), ADItem(10.0, "First."), ADItem(10.0, "Second.")],
        )

    def test_empty_input(self) -> None:
        self.assertEqual(parse_mtg_ad(" \n\r\n\t"), [])

    def test_ambiguous_three_part_timestamp_requires_explicit_format(self) -> None:
        # MTG-AD interprets these as M:SS:fraction; they can also mean H:MM:SS.
        # PYRO-KD requires .000 (clock) or three final digits (milliseconds).
        for timestamp in ("00:00:10", "1:02:03", "0:17:46"):
            with self.subTest(timestamp=timestamp):
                with self.assertRaisesRegex(ValueError, r"line 1: ambiguous.*H:MM:SS.000"):
                    parse_mtg_ad(f"{timestamp}-STANDARD Description.")
        self.assertEqual(parse_mtg_ad("1:02:03.000-STANDARD Clock."), [ADItem(3723.0, "Clock.")])

    def test_malformed_entries_include_line_number(self) -> None:
        malformed = (
            "bad-STANDARD Description.",
            "00:60:01.000-STANDARD Description.",
            "1:60.000-STANDARD Description.",
            "0:60:000-STANDARD Description.",
            "-0:10.000-STANDARD Description.",
            "0:10.3.4-STANDARD Description.",
            "0:10.000-STANDARD",
            "0:10.000",
            "0:10.000-UNKNOWN Description.",
            "0:10.000-0:15.000-STANDARD Description.",
        )
        for entry in malformed:
            with self.subTest(entry=entry):
                with self.assertRaisesRegex(ValueError, r"Invalid MTG-AD entry on line 3"):
                    parse_mtg_ad("0:01.000-STANDARD Valid.\n\n" + entry)

    def test_ad_item_is_immutable_and_has_no_end_time(self) -> None:
        item = ADItem(10.0, "A woman waves.")
        self.assertFalse(hasattr(item, "end"))
        with self.assertRaises(FrozenInstanceError):
            item.timestamp = 20.0  # type: ignore[misc]


class TemporalMetricsTests(unittest.TestCase):
    def test_perfect_match(self) -> None:
        result = temporal_precision_recall_f1([10, 20, 30], [10, 20, 30])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (3, 0, 0))
        self.assertEqual((result["precision"], result["recall"], result["f1"]), (1.0, 1.0, 1.0))
        self.assertEqual(result["mean_temporal_error_seconds"], 0.0)
        self.assertEqual(result["tolerance_seconds"], 2.0)

    def test_within_outside_and_inclusive_boundary(self) -> None:
        for prediction, expected_tp in ((11.9, 1), (12.1, 0), (12.0, 1), (8.0, 1), (7.9, 0)):
            with self.subTest(prediction=prediction):
                result = temporal_precision_recall_f1([prediction], [10], 2.0)
                self.assertEqual(result["tp"], expected_tp)
                if expected_tp:
                    self.assertAlmostEqual(result["mean_temporal_error_seconds"], abs(prediction - 10))
                else:
                    self.assertIsNone(result["mean_temporal_error_seconds"])

    def test_duplicate_predictions_choose_lowest_error(self) -> None:
        result = temporal_precision_recall_f1([9.5, 10.2], [10.0])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 0))
        self.assertEqual(result["matches"][0]["prediction_index"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertAlmostEqual(result["mean_temporal_error_seconds"], 0.2)

    def test_ambiguous_matching_prioritizes_cardinality(self) -> None:
        # Nearest-neighbor would use 1 -> 1.5 and strand 2.5. Two pairs win
        # even though their total error (2 s) exceeds the single pair's 0.5 s.
        matches = match_timestamps([1, 2.5], [0, 1.5], 1.0)
        self.assertEqual([(m.prediction_index, m.ground_truth_index) for m in matches], [(0, 0), (1, 1)])
        self.assertEqual(sum(m.error_seconds for m in matches), 2.0)

    def test_ambiguous_matching_minimizes_error_at_maximum_cardinality(self) -> None:
        matches = match_timestamps([1, 2.5], [0, 1.5, 3], 1.0)
        self.assertEqual([(m.prediction_index, m.ground_truth_index) for m in matches], [(0, 1), (1, 2)])
        self.assertEqual(sum(m.error_seconds for m in matches), 1.0)

    def test_unsorted_inputs_keep_original_indices_and_are_unchanged(self) -> None:
        predictions, gt = [30, 10, 20], [21, 31, 11]
        self.assertEqual(
            match_timestamps(predictions, gt),
            [
                TemporalMatch(0, 1, 30.0, 31.0, 1.0),
                TemporalMatch(1, 2, 10.0, 11.0, 1.0),
                TemporalMatch(2, 0, 20.0, 21.0, 1.0),
            ],
        )
        self.assertEqual(predictions, [30, 10, 20])
        self.assertEqual(gt, [21, 31, 11])

    def test_exact_duplicates_and_ties_are_deterministic(self) -> None:
        for _ in range(3):
            self.assertEqual(match_timestamps([10, 10], [10], 0), [TemporalMatch(0, 0, 10, 10, 0)])
            self.assertEqual(match_timestamps([10], [10, 10], 0), [TemporalMatch(0, 0, 10, 10, 0)])
            self.assertEqual(match_timestamps([11, 9], [10], 1), [TemporalMatch(1, 0, 9, 10, 1)])
            matches = match_timestamps([10, 10], [10, 10], 0)
            self.assertEqual([(m.prediction_index, m.ground_truth_index) for m in matches], [(0, 0), (1, 1)])

    def test_empty_and_no_match_cases(self) -> None:
        cases = (
            ([], [10], (0, 0, 1), (0.0, 0.0, 0.0)),
            ([10], [], (0, 1, 0), (0.0, 0.0, 0.0)),
            ([], [], (0, 0, 0), (None, None, None)),
            ([50], [10], (0, 1, 1), (0.0, 0.0, 0.0)),
        )
        for predictions, gt, counts, scores in cases:
            with self.subTest(predictions=predictions, gt=gt):
                result = temporal_precision_recall_f1(predictions, gt)
                self.assertEqual(tuple(result[k] for k in ("tp", "fp", "fn")), counts)
                self.assertEqual(tuple(result[k] for k in ("precision", "recall", "f1")), scores)
                self.assertIsNone(result["mean_temporal_error_seconds"])
                self.assertEqual(result["matches"], [])

    def test_zero_tolerance_only_matches_identical_times(self) -> None:
        self.assertEqual(match_timestamps([10.001], [10], 0), [])
        self.assertEqual(len(match_timestamps([0, 10], [0, 10], 0)), 2)

    def test_tolerance_sweep_is_independent_and_preserves_order(self) -> None:
        predictions, gt = [11.9, 24], [10, 20]
        results = temporal_tolerance_sweep(predictions, gt)
        self.assertEqual([r["tolerance_seconds"] for r in results], [0.0, 1.0, 2.0, 3.0, 5.0])
        self.assertEqual([r["tp"] for r in results], [0, 0, 1, 1, 2])
        for result in results:
            self.assertEqual(result, temporal_precision_recall_f1(predictions, gt, result["tolerance_seconds"]))
        custom = temporal_tolerance_sweep(predictions, gt, (5, 0, 5))
        self.assertEqual([r["tp"] for r in custom], [2, 0, 2])
        custom[0]["matches"].clear()
        self.assertEqual(len(custom[2]["matches"]), 2)
        self.assertEqual(temporal_tolerance_sweep(predictions, gt, ()), [])

    def test_invalid_times_and_tolerances(self) -> None:
        invalid_values = (-1, math.nan, math.inf, -math.inf, True, "10", None)
        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, r"predictions\[0\]"):
                    match_timestamps([invalid], [10])  # type: ignore[list-item]
                with self.assertRaisesRegex(ValueError, r"ground_truth\[0\]"):
                    match_timestamps([10], [invalid])  # type: ignore[list-item]
                with self.assertRaisesRegex(ValueError, "tolerance_seconds"):
                    temporal_precision_recall_f1([], [], invalid)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "tolerance_seconds"):
            temporal_tolerance_sweep([10], [10], (0, -1))

    def test_match_is_immutable_and_metrics_are_json_serializable(self) -> None:
        match, = match_timestamps([10], [10])
        with self.assertRaises(FrozenInstanceError):
            match.error_seconds = 1  # type: ignore[misc]
        for predictions, gt in (([10], [11]), ([], [])):
            result = temporal_precision_recall_f1(predictions, gt)
            self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_parsed_gt_integrates_with_metrics(self) -> None:
        items = parse_mtg_ad("0:10.000-STANDARD\u2028A wave.\n0:20.000-STANDARD A sign.")
        result = temporal_precision_recall_f1([11.5, 30], [item.timestamp for item in items])
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertEqual(result["f1"], 0.5)
        self.assertEqual(result["mean_temporal_error_seconds"], 1.5)

    def test_matches_exhaustive_assignment_oracle(self) -> None:
        # Independently enumerate all partial assignments, including crossing
        # ones. Small integer inputs keep objective comparisons exact.
        def best_score(
            predictions: tuple[int, ...], gt: tuple[int, ...], tolerance: int,
        ) -> tuple[int, float]:
            def visit(i: int, used: frozenset[int]) -> tuple[int, float]:
                if i == len(predictions):
                    return (0, 0.0)
                best = visit(i + 1, used)
                for j, timestamp in enumerate(gt):
                    error = abs(predictions[i] - timestamp)
                    if j not in used and error <= tolerance:
                        count, negative_error = visit(i + 1, used | {j})
                        best = max(best, (count + 1, negative_error - error))
                return best

            return visit(0, frozenset())

        sequences = [sequence for size in range(4) for sequence in product(range(3), repeat=size)]
        for predictions, gt, tolerance in product(sequences, sequences, range(3)):
            with self.subTest(predictions=predictions, gt=gt, tolerance=tolerance):
                matches = match_timestamps(predictions, gt, tolerance)
                score = (len(matches), -sum(m.error_seconds for m in matches))
                self.assertEqual(score, best_score(predictions, gt, tolerance))
                self.assertEqual(len({m.prediction_index for m in matches}), len(matches))
                self.assertEqual(len({m.ground_truth_index for m in matches}), len(matches))
                for match in matches:
                    self.assertEqual(match.prediction_time, predictions[match.prediction_index])
                    self.assertEqual(match.ground_truth_time, gt[match.ground_truth_index])
                    self.assertLessEqual(match.error_seconds, tolerance)


if __name__ == "__main__":
    unittest.main()
