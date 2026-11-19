"""A generation cut off at max_new_tokens is an error row: not scored, retried on requeue."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts import evaluate_ad as evaluation
from src.datasets.mtg_ad import MTGVideo
from src.evaluation.parsing import ADItem
from src.generation.hf_vlm import clean
from src.generation.outputs import final_rows, is_success, merge_meta, read_rows, settled_ids, split_rows, status_counts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_ad_vlm import done_ids, finalize_generation  # noqa: E402

SCRATCHPAD = (
    "to=selfThese 4 frames are from a video, in order, spanning about 2 seconds.\n"
    "Write one audio description ... Output only the description.\n\n"
    "Need one description summarizing what is happening across frames. Count words. Let's craft"
)


def make_row(base, result, temperature=0.0):
    return {"id": base, "video_id": "a", "time": float(base[-2:]), "temperature": temperature, **result}


def ok(moment_id, max_new_tokens=64, temperature=0.0):
    return make_row(moment_id, finalize_generation("A dog runs.", "r", 12, 1, 0.1, max_new_tokens), temperature)


def cut(moment_id, max_new_tokens=64, temperature=0.0):
    return make_row(moment_id, finalize_generation(SCRATCHPAD, "r", max_new_tokens, 1, 0.1, max_new_tokens), temperature)


def oom(moment_id):
    return {"id": moment_id, "video_id": "a", "time": float(moment_id[-2:]), "temperature": 0.0,
            "error": "generate: OutOfMemoryError()"}


class FinalizeGenerationTests(unittest.TestCase):
    def test_a_completed_generation_has_no_error(self):
        result = finalize_generation("A dog runs.", "<raw>", 12, 4, 1.23456, 64)
        self.assertNotIn("error", result)
        self.assertFalse(result["hit_token_limit"])
        self.assertEqual(result["output"], "A dog runs.")
        self.assertEqual(result["batch_latency_s"], 1.235)

    def test_reaching_the_limit_is_an_error_and_keeps_the_cut_off_text(self):
        result = finalize_generation(clean(SCRATCHPAD), "<raw scratchpad>", 512, 4, 9.0, 512)
        self.assertTrue(result["hit_token_limit"])
        self.assertTrue(result["error"].startswith("hit_token_limit"))
        self.assertIn("max_new_tokens=512", result["error"])
        self.assertTrue(result["output"].startswith("to=selfThese 4 frames"))  # kept for debugging
        self.assertEqual(result["raw_output"], "<raw scratchpad>")

    def test_the_boundary_is_inclusive(self):
        self.assertNotIn("error", finalize_generation("x", "x", 511, 1, 0.0, 512))
        self.assertIn("error", finalize_generation("x", "x", 512, 1, 0.0, 512))


class RequeueTests(unittest.TestCase):
    def test_truncated_rows_are_not_done_and_are_retried_after_a_requeue(self):
        good = make_row("a_U_11", finalize_generation("A dog runs.", "r", 12, 1, 0.1, 64))
        cut = make_row("a_U_21", finalize_generation(SCRATCHPAD, "r", 64, 1, 0.1, 64))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "out.jsonl")
            path.write_text("\n".join(json.dumps(r) for r in (good, cut)) + "\n", encoding="utf-8")
            self.assertEqual(done_ids(path), {"a_U_11"})


def video(video_id, times):
    return MTGVideo(
        video_id=video_id, split="validation", category="synthetic", duration_seconds=300,
        ad=[ADItem(t, f"GT {video_id} {t:g}") for t in times], transcript=[], visual_info=None,
    )


class EvaluationTests(unittest.TestCase):
    def test_truncated_rows_are_reported_as_errors_and_never_scored(self):
        good = make_row("a_U_11", finalize_generation("A dog runs.", "r", 12, 1, 0.1, 64))
        cut = make_row("a_U_21", finalize_generation(SCRATCHPAD, "r", 64, 1, 0.1, 64))
        bert = Mock(side_effect=lambda candidates, references, **kwargs: (
            ([0.8] * len(candidates), [0.6] * len(candidates), [0.7] * len(candidates)), "mock"))
        encoder = Mock()
        encoder.encode.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "generated.jsonl")
            path.write_text("\n".join(json.dumps(r) for r in (good, cut)) + "\n", encoding="utf-8")
            with patch.object(evaluation.mtg_ad, "load_split", return_value=[video("a", (10.0, 20.0))]):
                result = evaluation.evaluate_ad(
                    path, Path(tmp, "dataset"), "validation", bert_scorer=bert, encoder=encoder)
        self.assertEqual(result["num_generation_errors"], 1)
        self.assertEqual(result["num_successful_rows"], 1)
        self.assertTrue(result["generation_errors"][0]["error"].startswith("hit_token_limit"))
        candidates = [c for call in bert.call_args_list for c in call.args[0]]
        self.assertEqual(candidates, ["A dog runs."])  # the scratchpad never reached the scorer


def legacy_cut(moment_id):
    """A token-limit row as written before cut-offs were recorded as errors: no "error" field."""
    return {"id": moment_id, "video_id": "a", "time": float(moment_id[-2:]), "output": SCRATCHPAD,
            "raw_output": "r", "new_tokens": 64, "batch_size": 1, "hit_token_limit": True,
            "batch_latency_s": 0.1}


class LegacyTruncationTests(unittest.TestCase):
    """Historical rows with hit_token_limit=true and no error are failures in every reader."""

    def evaluate(self, rows):
        bert = Mock(side_effect=lambda candidates, references, **kwargs: (
            ([0.8] * len(candidates), [0.6] * len(candidates), [0.7] * len(candidates)), "mock"))
        encoder = Mock()
        encoder.encode.side_effect = lambda texts, **kwargs: [[1.0, 0.0] for _ in texts]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "generated.jsonl")
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            with patch.object(evaluation.mtg_ad, "load_split", return_value=[video("a", (10.0, 20.0))]):
                result = evaluation.evaluate_ad(
                    path, Path(tmp, "dataset"), "validation", bert_scorer=bert, encoder=encoder)
        return result, [c for call in bert.call_args_list for c in call.args[0]]

    def test_a_legacy_truncated_row_is_a_failure_and_never_scored(self):
        result, scored = self.evaluate([ok("a_U_11"), legacy_cut("a_U_21")])
        self.assertEqual((result["num_successful_rows"], result["num_generation_errors"]), (1, 1))
        self.assertEqual(scored, ["A dog runs."])

    def test_legacy_truncation_then_success_is_history_not_a_duplicate_success(self):
        result, scored = self.evaluate([legacy_cut("a_U_11"), ok("a_U_11")])
        self.assertEqual((result["num_successful_rows"], result["num_generation_errors"]), (1, 0))
        self.assertEqual(result["num_superseded_error_rows"], 1)
        self.assertEqual(scored, ["A dog runs."])

    def test_legacy_truncation_then_oom_leaves_the_real_failure_standing(self):
        result, scored = self.evaluate([legacy_cut("a_U_21"), oom("a_U_21")])
        self.assertEqual(result["generation_errors"], [oom("a_U_21")])
        self.assertEqual(result["superseded_error_rows"], [legacy_cut("a_U_21")])
        self.assertEqual(scored, [])

    def test_repeated_legacy_truncations_count_once(self):
        result, _ = self.evaluate([legacy_cut("a_U_21"), legacy_cut("a_U_21")])
        self.assertEqual((result["num_successful_rows"], result["num_generation_errors"]), (0, 1))
        self.assertEqual(result["num_superseded_error_rows"], 1)

    def test_a_current_success_with_hit_limit_false_is_still_a_success(self):
        self.assertTrue(is_success(ok("a_U_11")))
        self.assertFalse(is_success(legacy_cut("a_U_11")))
        self.assertFalse(is_success(cut("a_U_11")))

    def test_requeue_retries_legacy_rows_at_any_budget_and_does_not_call_them_done(self):
        rows = [legacy_cut("a_U_21")]
        for tokens in (32, 64, 128):
            self.assertEqual(settled_ids(rows, tokens, 0.0), (set(), set()))
        self.assertEqual(settled_ids(rows)[0], set())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "out.jsonl")
            path.write_text(json.dumps(legacy_cut("a_U_21")) + "\n", encoding="utf-8")
            self.assertEqual(done_ids(path), set())
            self.assertEqual(done_ids(path, 128, 0.0), set())

    def test_meta_counts_standing_legacy_truncations_as_errors(self):
        self.assertEqual(
            status_counts([legacy_cut("a_U_21")]), {"n_ok": 0, "n_error": 1, "n_hit_token_limit": 1})
        self.assertEqual(
            status_counts([legacy_cut("a_U_11"), ok("a_U_11")]),
            {"n_ok": 1, "n_error": 0, "n_hit_token_limit": 0})
        self.assertEqual(
            status_counts([legacy_cut("a_U_21"), oom("a_U_21")]),
            {"n_ok": 0, "n_error": 1, "n_hit_token_limit": 0})


class FinalRowsTests(unittest.TestCase):
    def test_a_success_supersedes_earlier_errors(self):
        rows = [oom("a_U_11"), cut("a_U_21"), ok("a_U_11"), oom("a_U_21")]
        final, superseded = split_rows(rows)
        # Moment 11 succeeded after its error; moment 21 never did, so its latest error stands.
        self.assertEqual([r["id"] for r in final], ["a_U_11", "a_U_21"])
        self.assertEqual(final[1], rows[3])
        self.assertEqual(superseded, [rows[0], rows[1]])
        self.assertEqual(status_counts(rows), {"n_ok": 1, "n_error": 1, "n_hit_token_limit": 0})
        self.assertEqual(status_counts([oom("a_U_11"), ok("a_U_11")])["n_error"], 0)

    def test_repeated_errors_count_once_and_the_latest_stands(self):
        first, second, third = cut("a_U_21"), cut("a_U_21"), oom("a_U_21")
        final, superseded = split_rows([first, ok("a_U_11"), second, third])
        self.assertEqual(final, [ok("a_U_11"), third])
        self.assertEqual(superseded, [first, second])
        counts = status_counts([first, second, third])
        self.assertEqual(counts, {"n_ok": 0, "n_error": 1, "n_hit_token_limit": 0})  # the standing error is OOM

    def test_token_limit_failures_are_counted_when_they_stand(self):
        rows = [ok("a_U_11"), cut("a_U_21"), cut("a_U_21"), oom("a_U_31")]
        self.assertEqual(status_counts(rows), {"n_ok": 1, "n_error": 2, "n_hit_token_limit": 1})

    def test_rows_without_ids_pass_through_and_order_is_preserved(self):
        loose = {"error": "input: bad frame"}
        final = final_rows([loose, ok("a_U_11"), loose, cut("a_U_21")])
        self.assertEqual(final, [loose, ok("a_U_11"), loose, cut("a_U_21")])

    def test_duplicate_successes_are_all_kept_for_the_evaluator_to_reject(self):
        final = final_rows([ok("a_U_11"), ok("a_U_11")])
        self.assertEqual(len(final), 2)

    def test_read_rows_skips_partial_lines_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "out.jsonl")
            self.assertEqual(read_rows(path), [])
            path.write_text(json.dumps(ok("a_U_11")) + '\n{"id": "a_U_2\n[1, 2]\n', encoding="utf-8")
            self.assertEqual([r["id"] for r in read_rows(path)], ["a_U_11"])


class RetryPolicyTests(unittest.TestCase):
    def settled(self, rows, tokens, temperature=0.0):
        return settled_ids(rows, tokens, temperature)

    def test_a_token_limit_failure_is_not_retried_with_the_same_settings(self):
        done, repeats = self.settled([cut("a_U_21", 64)], 64)
        self.assertEqual((done, repeats), ({"a_U_21"}, {"a_U_21"}))
        # Fewer tokens allowed now than when it failed: it would cut off again.
        self.assertEqual(self.settled([cut("a_U_21", 64)], 32)[0], {"a_U_21"})

    def test_raising_max_new_tokens_or_sampling_makes_it_retryable(self):
        self.assertEqual(self.settled([cut("a_U_21", 64)], 128), (set(), set()))
        self.assertEqual(self.settled([cut("a_U_21", 64, temperature=0.7)], 64, temperature=0.7), (set(), set()))
        self.assertEqual(self.settled([cut("a_U_21", 64)], 64, temperature=0.7), (set(), set()))

    def test_other_errors_and_legacy_rows_are_retried(self):
        self.assertEqual(self.settled([oom("a_U_21")], 64), (set(), set()))
        legacy = {"id": "a_U_21", "video_id": "a", "time": 21.0, "hit_token_limit": True,
                  "error": "hit_token_limit: old row without recorded settings"}
        self.assertEqual(self.settled([legacy], 64), (set(), set()))

    def test_successes_and_resolved_failures_are_settled(self):
        self.assertEqual(self.settled([ok("a_U_11")], 64)[0], {"a_U_11"})
        self.assertEqual(self.settled([oom("a_U_11"), ok("a_U_11")], 64)[0], {"a_U_11"})
        self.assertEqual(settled_ids([cut("a_U_21")], None)[0], set())  # no settings given: successes only

    def test_done_ids_default_still_means_successes_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "out.jsonl")
            path.write_text("\n".join(json.dumps(r) for r in (ok("a_U_11"), cut("a_U_21"))) + "\n", encoding="utf-8")
            self.assertEqual(done_ids(path), {"a_U_11"})
            self.assertEqual(done_ids(path, 64, 0.0), {"a_U_11", "a_U_21"})
            self.assertEqual(done_ids(path, 128, 0.0), {"a_U_11"})

    def test_a_requeue_does_not_append_another_error_for_the_same_cut_off(self):
        """Simulate attempts the way run_ad_vlm drives them: todo = moments not settled."""
        moments = ["a_U_11", "a_U_21", "a_U_31"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "out.jsonl")
            attempt_1 = [ok("a_U_11"), cut("a_U_21"), oom("a_U_31")]
            path.write_text("\n".join(json.dumps(r) for r in attempt_1) + "\n", encoding="utf-8")
            todo = [m for m in moments if m not in done_ids(path, 64, 0.0)]
            self.assertEqual(todo, ["a_U_31"])  # the cut-off moment is not run again
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(ok("a_U_31")) + "\n")  # the OOM moment succeeds on retry
            counts = status_counts(read_rows(path))
            self.assertEqual(counts, {"n_ok": 2, "n_error": 1, "n_hit_token_limit": 1})
            # With a larger budget the cut-off moment becomes eligible again.
            self.assertEqual([m for m in moments if m not in done_ids(path, 128, 0.0)], ["a_U_21"])


class MergeMetaTests(unittest.TestCase):
    def attempt(self, job, ok_rows, errors, load_s, generate_s, peak, **extra):
        return {
            "model_id": "m", "model_class": "C", "quant": "none", "n_gpus": 2, "args": {"max_new_tokens": 64},
            "gpu": "A100", "torch": "t", "transformers": "x", "host": f"b-{job}", "slurm_job_id": str(job),
            "n_ok": ok_rows, "n_error": errors, "n_hit_token_limit": 0, "load_s": load_s,
            "generate_s": generate_s, "peak_gpu_mem_gb": peak,
            "peak_gpu_mem_gb_per_device": None if peak is None else [peak / 2, peak / 2],
            **extra,
        }

    def test_first_attempt_is_recorded_as_is(self):
        counts = {"n_ok": 3, "n_error": 1, "n_hit_token_limit": 1}
        meta = merge_meta(None, self.attempt(1, 3, 1, 90.0, 12.0, 20.0), counts)
        self.assertEqual((meta["n_ok"], meta["n_error"], meta["n_hit_token_limit"]), (3, 1, 1))
        self.assertEqual((meta["load_s"], meta["generate_s"], meta["s_per_item"]), (90.0, 12.0, 4.0))
        self.assertEqual((meta["n_attempts"], len(meta["attempts"])), (1, 1))
        self.assertEqual(meta["slurm_job_id"], "1")

    def test_a_requeue_accumulates_instead_of_replacing(self):
        first = merge_meta(None, self.attempt(1, 3, 1, 90.0, 12.0, 20.0), {"n_ok": 3, "n_error": 1, "n_hit_token_limit": 0})
        # The second attempt ran only the one retried moment, which succeeded.
        second = merge_meta(first, self.attempt(2, 1, 0, 80.0, 6.0, 31.0), {"n_ok": 4, "n_error": 0, "n_hit_token_limit": 0})
        self.assertEqual((second["n_ok"], second["n_error"]), (4, 0))  # the file's state now, not this attempt's
        self.assertEqual((second["load_s"], second["generate_s"]), (170.0, 18.0))  # whole-run cost
        self.assertEqual(second["s_per_item"], 4.5)
        self.assertEqual(second["peak_gpu_mem_gb"], 31.0)  # the heaviest attempt
        self.assertEqual(second["peak_gpu_mem_gb_per_device"], [15.5, 15.5])
        self.assertEqual(second["n_attempts"], 2)
        self.assertEqual([a["slurm_job_id"] for a in second["attempts"]], ["1", "2"])
        self.assertEqual(second["slurm_job_id"], "2")  # latest attempt's identity at top level
        json.dumps(second)

    def test_a_meta_file_from_before_attempts_existed_becomes_the_first_attempt(self):
        legacy = {"model_id": "m", "model_class": "C", "quant": "none", "n_gpus": 2, "args": {},
                  "n_ok": 4, "n_error": 0, "load_s": 94.3, "generate_s": 6.1, "s_per_item": 1.5,
                  "peak_gpu_mem_gb": 9.45, "gpu": "A100", "host": "b-1", "slurm_job_id": "27224"}
        meta = merge_meta(legacy, self.attempt(2, 1, 0, 10.0, 2.0, 12.0), {"n_ok": 5, "n_error": 0, "n_hit_token_limit": 0})
        self.assertEqual(meta["n_attempts"], 2)
        self.assertEqual(meta["attempts"][0]["slurm_job_id"], "27224")
        self.assertEqual((meta["load_s"], meta["generate_s"]), (104.3, 8.1))

    def test_no_successes_gives_no_per_item_time(self):
        meta = merge_meta(None, self.attempt(1, 0, 2, 5.0, 1.0, None), {"n_ok": 0, "n_error": 2, "n_hit_token_limit": 2})
        self.assertIsNone(meta["s_per_item"])
        self.assertIsNone(meta["peak_gpu_mem_gb"])


if __name__ == "__main__":
    unittest.main()
