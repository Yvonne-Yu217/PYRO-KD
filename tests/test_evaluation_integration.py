"""Offline integration tests using synthetic MTGVideo and generated JSONL."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import evaluate_ad as evaluation
from src.datasets.mtg_ad import MTGVideo
from src.evaluation import bertscore, cosine_similarity, metrics
from src.evaluation.parsing import ADItem


def video(video_id: str, times: tuple[float, ...]) -> MTGVideo:
    return MTGVideo(
        video_id=video_id, split="validation", category="synthetic", duration_seconds=300,
        ad=[ADItem(time, f"GT {video_id} {time:g}") for time in times],
        transcript=[], visual_info=None,
    )


def row(video_id: str, time: float, text: str = "A description.", **metadata) -> dict:
    return {"video_id": video_id, "time": time, "output": text, **metadata}


class EvaluationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name, "generated.jsonl")
        self.videos = [video("a", (10, 20)), video("b", (30,)), video("unused", (50,))]
        self.loader = patch.object(evaluation.mtg_ad, "load_split", return_value=self.videos).start()
        self.addCleanup(patch.stopall)
        self.bert = Mock(side_effect=lambda candidates, references, **kwargs: (
            ([0.8] * len(candidates), [0.6] * len(candidates), [0.7] * len(candidates)),
            "mock-configuration",
        ))
        self.encoder = Mock()
        self.encoder.encode.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]

    def write_rows(self, rows: list[dict]) -> None:
        self.path.write_text("\n".join(json.dumps(item) for item in rows) + "\n", encoding="utf-8")

    def evaluate(self, rows: list[dict], **kwargs) -> dict:
        self.write_rows(rows)
        return evaluation.evaluate_ad(
            self.path, Path(self.tmp.name, "dataset"), "validation",
            bert_scorer=self.bert, encoder=self.encoder, **kwargs,
        )

    def test_successful_rows_group_by_video_and_preserve_original_order(self) -> None:
        result = self.evaluate([row("b", 31, "B."), row("a", 21, "A later."), row("a", 11, "A first.")])
        self.assertEqual([item["video_id"] for item in result["videos"]], ["a", "b"])
        self.assertEqual([item["num_generated"] for item in result["videos"]], [2, 1])
        self.assertEqual(
            [item["text"] for item in result["videos"][0]["generated_items"]],
            ["A later.", "A first."],
        )
        self.assertEqual(result["coverage"], {
            "num_split_videos": 3, "num_evaluated_videos": 2,
            "split_video_ids_without_input": ["unused"],
        })
        self.loader.assert_called_once_with(Path(self.tmp.name, "dataset"), "validation")

    def test_generation_errors_are_skipped_counted_and_preserved(self) -> None:
        failed = {"video_id": "a", "time": 20, "id": "retry", "error": "OOM", "prior": "P0"}
        # The same ID may fail and then succeed on a later generation attempt.
        result = self.evaluate([failed, row("a", 10, id="retry", prior="P0")])
        self.assertEqual(result["num_input_rows"], 2)
        self.assertEqual(result["num_successful_rows"], 1)
        self.assertEqual(result["num_generation_errors"], 1)
        self.assertEqual(result["generation_errors"], [failed])
        self.assertEqual(result["status"], "partial")
        per_video = result["videos"][0]
        self.assertEqual(per_video["num_generation_errors"], 1)
        self.assertEqual(per_video["generation_errors"], [failed])
        self.assertEqual((per_video["temporal"]["tp"], per_video["temporal"]["fn"]), (1, 1))
        self.assertEqual(self.bert.call_args.args[0], ["A description."])

    def test_any_error_key_marks_a_failed_row_even_with_output(self) -> None:
        result = self.evaluate([row("a", 10, error=None)])
        self.assertEqual(result["num_successful_rows"], 0)
        self.assertEqual(result["num_generation_errors"], 1)
        self.bert.assert_not_called()
        self.encoder.encode.assert_not_called()

    def test_same_match_objects_and_gt_instances_drive_both_semantic_functions(self) -> None:
        produced = []

        def capture_matches(*args, **kwargs):
            matches = metrics.match_timestamps(*args, **kwargs)
            produced.append(matches)
            return matches

        with (
            patch.object(evaluation, "match_timestamps", side_effect=capture_matches) as matcher,
            patch.object(evaluation, "compute_bertscore", wraps=bertscore.compute_bertscore) as bert,
            patch.object(evaluation, "compute_cosine_similarity", wraps=cosine_similarity.compute_cosine_similarity) as cosine,
        ):
            result = self.evaluate([row("a", 21), row("a", 100, "Unmatched."), row("a", 11)])
        matcher.assert_called_once()
        self.assertIs(bert.call_args.args[2], produced[0])
        self.assertIs(cosine.call_args.args[2], produced[0])
        for actual, expected in zip(bert.call_args.args[2], produced[0], strict=True):
            self.assertIs(actual, expected)
        self.assertIs(bert.call_args.args[1], self.videos[0].ad)
        self.assertIs(cosine.call_args.args[1], self.videos[0].ad)
        self.assertEqual(result["videos"][0]["temporal"]["matches"], [asdict(m) for m in produced[0]])
        self.assertNotIn("Unmatched.", self.bert.call_args.args[0])
        self.assertNotIn("Unmatched.", self.encoder.encode.call_args.args[0])

    def test_micro_counts_and_pair_weighted_semantic_aggregation(self) -> None:
        self.bert.side_effect = [
            (([0.2, 0.4], [0.3, 0.5], [0.1, 0.3]), "mock-configuration"),
            (([1.0], [0.8], [0.9]), "mock-configuration"),
        ]
        self.encoder.encode.side_effect = [
            [[1, 0], [1, 0], [1, 0], [0, 1]],
            [[1, 0], [0.6, 0.8]],
        ]
        result = self.evaluate([row("a", 10.5), row("a", 20.5), row("a", 100), row("b", 30.5)])
        aggregate = result["aggregate"]
        temporal = aggregate["temporal"]
        self.assertEqual((temporal["tp"], temporal["fp"], temporal["fn"]), (3, 1, 0))
        self.assertEqual(temporal["precision"], 3 / 4)
        self.assertEqual(temporal["recall"], 1)
        self.assertAlmostEqual(temporal["f1"], 6 / 7)
        self.assertEqual(temporal["mean_temporal_error_seconds"], 0.5)
        self.assertEqual(aggregate["total_matched_pairs"], 3)
        self.assertEqual(aggregate["bertscore"]["num_scored_pairs"], 3)
        self.assertAlmostEqual(aggregate["bertscore"]["precision_mean"], 1.6 / 3)
        self.assertAlmostEqual(aggregate["bertscore"]["recall_mean"], 1.6 / 3)
        self.assertAlmostEqual(aggregate["bertscore"]["f1_mean"], 1.3 / 3)
        self.assertAlmostEqual(aggregate["cosine"]["cosine_similarity_mean"], 1.6 / 3)
        self.assertEqual(aggregate["cosine"]["cosine_similarity_median"], 0.6)
        for per_video in result["videos"]:
            generated = [item["timestamp"] for item in per_video["generated_items"]]
            gt = next(v.ad for v in self.videos if v.video_id == per_video["video_id"])
            self.assertEqual(per_video["temporal"], metrics.temporal_precision_recall_f1(
                generated, [item.timestamp for item in gt],
            ))

    def test_multiple_videos_sum_false_negatives(self) -> None:
        result = self.evaluate([row("a", 10), row("b", 200)])
        temporal = result["aggregate"]["temporal"]
        self.assertEqual((temporal["tp"], temporal["fp"], temporal["fn"]), (1, 1, 2))
        self.assertEqual(temporal["precision"], 0.5)
        self.assertEqual(temporal["recall"], 1 / 3)
        self.assertEqual(temporal["f1"], 0.4)

    def test_no_match_scores_are_none_and_no_scorer_is_called(self) -> None:
        result = self.evaluate([row("a", 100), row("b", 100)])
        self.assertEqual(result["aggregate"]["total_matched_pairs"], 0)
        self.assertEqual(result["aggregate"]["temporal"]["precision"], 0)
        self.assertIsNone(result["aggregate"]["temporal"]["mean_temporal_error_seconds"])
        for scope in [result["aggregate"], *result["videos"]]:
            self.assertEqual(scope["bertscore"]["num_scored_pairs"], 0)
            self.assertIsNone(scope["bertscore"]["precision_mean"])
            self.assertIsNone(scope["bertscore"]["recall_mean"])
            self.assertIsNone(scope["bertscore"]["f1_mean"])
            self.assertIsNone(scope["cosine"]["cosine_similarity_mean"])
            self.assertIsNone(scope["cosine"]["cosine_similarity_median"])
        self.bert.assert_not_called()
        self.encoder.encode.assert_not_called()

    def test_actual_zero_semantic_scores_remain_zero(self) -> None:
        self.bert.side_effect = None
        self.bert.return_value = (([0.0], [0.0], [0.0]), "mock-configuration")
        self.encoder.encode.side_effect = None
        self.encoder.encode.return_value = [[1, 0], [0, 1]]
        result = self.evaluate([row("b", 30)])
        self.assertEqual(result["aggregate"]["bertscore"]["f1_mean"], 0)
        self.assertEqual(result["aggregate"]["cosine"]["cosine_similarity_mean"], 0)
        self.assertEqual(result["aggregate"]["cosine"]["num_scored_pairs"], 1)

    def test_error_only_video_is_evaluated_as_no_successful_predictions(self) -> None:
        result = self.evaluate([{"video_id": "a", "time": 10, "error": "input error"}])
        per_video = result["videos"][0]
        self.assertEqual(per_video["status"], "no_successful_generations")
        self.assertEqual((per_video["temporal"]["tp"], per_video["temporal"]["fp"], per_video["temporal"]["fn"]), (0, 0, 2))
        self.assertIsNone(per_video["bertscore"]["f1_mean"])

    def test_empty_file_and_explicit_full_split_scope(self) -> None:
        result = self.evaluate([])
        self.assertEqual(result["status"], "empty_input")
        self.assertEqual(result["videos"], [])
        self.assertIsNone(result["aggregate"]["temporal"]["precision"])
        self.assertIsNone(result["aggregate"]["bertscore"]["f1_mean"])
        full = self.evaluate([], all_videos=True)
        self.assertEqual(len(full["videos"]), 3)
        self.assertEqual(full["aggregate"]["temporal"]["fn"], 4)
        self.assertEqual(full["aggregate"]["temporal"]["recall"], 0)
        self.bert.assert_not_called()
        self.encoder.encode.assert_not_called()

    def test_video_with_empty_gt(self) -> None:
        self.videos[0] = video("a", ())
        result = self.evaluate([row("a", 10)])
        self.assertEqual(result["videos"][0]["status"], "no_ground_truth")
        self.assertEqual(result["aggregate"]["temporal"]["fp"], 1)
        self.assertIsNone(result["aggregate"]["cosine"]["cosine_similarity_mean"])

    def test_configurable_inclusive_tolerance(self) -> None:
        at_boundary = self.evaluate([row("b", 33)], tolerance_seconds=3.0)
        outside = self.evaluate([row("b", 33)], tolerance_seconds=2.0)
        self.assertEqual(at_boundary["aggregate"]["total_matched_pairs"], 1)
        self.assertEqual(outside["aggregate"]["total_matched_pairs"], 0)
        self.assertEqual(at_boundary["aggregate"]["temporal"]["tolerance_seconds"], 3)

    def test_required_fields_fail_clearly_with_path_and_line(self) -> None:
        invalid = (
            {"time": 10, "output": "Text."}, row("", 10), row("a", None),
            row("a", "10"), row("a", True), row("a", -1), row("a", math.inf),
            {"video_id": "a", "time": 10}, row("a", 10, " "), row("a", 10, None),
            row("a", 10, ["Text."]), {"video_id": "a", "error": "Missing time."},
        )
        for entry in invalid:
            with self.subTest(entry=entry):
                with self.assertRaisesRegex(ValueError, r"generated.jsonl:1:"):
                    self.evaluate([entry])
        self.bert.assert_not_called()
        self.encoder.encode.assert_not_called()

    def test_malformed_json_and_nonobject_rows(self) -> None:
        for raw in ('{"video_id":', "[]", "null", "42"):
            with self.subTest(raw=raw):
                self.path.write_text("\n" + raw + "\n")
                with self.assertRaisesRegex(ValueError, r"generated.jsonl:2:"):
                    evaluation.read_generated_jsonl(self.path)
        self.path.write_text("\n" + json.dumps(row("a", 10)) + "\n\n")
        self.assertEqual(len(evaluation.read_generated_jsonl(self.path)), 1)

    def test_unknown_ids_including_failed_rows_fail_before_scoring(self) -> None:
        for entry in (row("unknown", 10), {"video_id": "unknown", "time": 10, "error": "OOM"}):
            with self.subTest(entry=entry):
                with self.assertRaisesRegex(ValueError, "absent.*validation.*unknown"):
                    self.evaluate([entry])
        self.bert.assert_not_called()
        self.encoder.encode.assert_not_called()

    def test_metadata_preserved_and_mixed_experiments_rejected(self) -> None:
        metadata = {"selector": "U", "model": "vlm", "model_id": "org/vlm", "prior": "P0", "quant": "none"}
        result = self.evaluate([row("a", 10, id="moment-1", custom={"seed": 7}, **metadata)])
        self.assertEqual(result["metadata"], metadata)
        per_item = result["videos"][0]["generated_items"][0]["metadata"]
        self.assertEqual(per_item["id"], "moment-1")
        self.assertEqual(per_item["custom"], {"seed": 7})
        for field in metadata:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"Mixed {field}"):
                    self.evaluate([row("a", 10, **metadata), row("b", 30, **{**metadata, field: "other"})])

    def test_duplicate_success_ids_fail_but_duplicate_timestamps_are_preserved(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate successful id"):
            self.evaluate([row("a", 10, id="same"), row("a", 20, id="same")])
        result = self.evaluate([row("a", 10, id="first"), row("a", 10, id="second")])
        self.assertEqual(result["aggregate"]["temporal"]["fp"], 1)

    def test_semantic_failures_are_reported_without_losing_temporal_results(self) -> None:
        self.bert.side_effect = RuntimeError("BERT model unavailable")
        result = self.evaluate([row("a", 10)])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["aggregate"]["total_matched_pairs"], 1)
        self.assertEqual(result["aggregate"]["bertscore"]["num_scored_pairs"], 0)
        self.assertIsNone(result["aggregate"]["bertscore"]["f1_mean"])
        self.assertEqual(result["aggregate"]["cosine"]["num_scored_pairs"], 1)
        self.assertIn("BERT model unavailable", result["semantic_errors"][0]["error"])
        self.assertEqual(result["videos"][0]["status"], "semantic_error")

    def test_pair_aggregation_excludes_failed_videos_only_for_failed_metric(self) -> None:
        self.bert.side_effect = [RuntimeError("Unavailable"), (([0.6], [0.7], [0.8]), "mock")]
        self.encoder.encode.side_effect = [RuntimeError("Encoder unavailable"), [[1, 0], [1, 0]]]
        result = self.evaluate([row("a", 10), row("a", 20), row("b", 30)])
        self.assertEqual(result["aggregate"]["total_matched_pairs"], 3)
        self.assertEqual(result["aggregate"]["bertscore"]["num_scored_pairs"], 1)
        self.assertEqual(result["aggregate"]["bertscore"]["f1_mean"], 0.8)
        self.assertEqual(result["aggregate"]["cosine"]["num_scored_pairs"], 1)
        self.assertEqual(len(result["semantic_errors"]), 2)

    def test_invalid_tolerance_and_duplicate_gt_ids_fail(self) -> None:
        for tolerance in (-1, math.inf, math.nan, True):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(ValueError, "tolerance_seconds"):
                    self.evaluate([], tolerance_seconds=tolerance)
        self.videos.append(self.videos[0])
        with self.assertRaisesRegex(ValueError, "duplicate video IDs"):
            self.evaluate([])

    def test_cli_saves_strict_json_report_and_passes_options(self) -> None:
        self.write_rows([row("a", 10)])
        output = Path(self.tmp.name, "reports", "evaluation.json")
        package_score = Mock(return_value=(([0.2], [0.3], [0.4]), "mock-config"))
        # Exercise the real CLI orchestration with injected scoring dependencies.
        original = evaluation.evaluate_ad

        def offline(*args, **kwargs):
            return original(*args, **kwargs, bert_scorer=package_score, encoder=self.encoder)

        with patch.object(evaluation, "evaluate_ad", side_effect=offline), redirect_stdout(io.StringIO()):
            code = evaluation.main([
                "--input", str(self.path), "--data-dir", self.tmp.name, "--split", "validation",
                "--output", str(output), "--tolerance", "1.5", "--lang", "en",
                "--bert-model", "test-bert", "--cosine-model", "test-cosine",
            ])
        self.assertEqual(code, 0)
        report = json.loads(output.read_text())
        self.assertEqual(report["aggregate"]["temporal"]["tolerance_seconds"], 1.5)
        self.assertEqual(report["videos"][0]["bertscore"]["model_type"], "test-bert")
        self.assertEqual(report["videos"][0]["cosine"]["model_name"], "test-cosine")
        self.assertEqual(report["evaluation_config"], {
            "tolerance_seconds": 1.5,
            "bertscore": {"lang": "en", "model_type": "test-bert"},
            "cosine": {"model_name": "test-cosine"},
        })
        self.assertNotIn("rubric", report)

    def test_cli_partial_exit_and_invalid_input_do_not_overwrite_source(self) -> None:
        self.write_rows([{"video_id": "a", "time": 10, "error": "OOM"}])
        output = Path(self.tmp.name, "evaluation.json")
        arguments = ["--input", str(self.path), "--split", "validation", "--output", str(output)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(evaluation.main(arguments), 1)
        self.assertEqual(json.loads(output.read_text())["num_generation_errors"], 1)
        original = self.path.read_text()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            evaluation.main(["--input", str(self.path), "--split", "validation", "--output", str(self.path)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(self.path.read_text(), original)


if __name__ == "__main__":
    unittest.main()
