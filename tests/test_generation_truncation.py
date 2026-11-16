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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_ad_vlm import done_ids, finalize_generation  # noqa: E402

SCRATCHPAD = (
    "to=selfThese 4 frames are from a video, in order, spanning about 2 seconds.\n"
    "Write one audio description ... Output only the description.\n\n"
    "Need one description summarizing what is happening across frames. Count words. Let's craft"
)


def make_row(base, result):
    return {"id": base, "video_id": "a", "time": float(base[-2:]), **result}


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


if __name__ == "__main__":
    unittest.main()
