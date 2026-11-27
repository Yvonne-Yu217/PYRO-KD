"""Regressions for the review of the HPC results integration (#24)."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from src.audio import cues_io
from src.evaluation import clip_text
from src.experiments import generation_stage, io as xio, matrix
from src.generation import backends
from src.generation.outputs import merge_meta, settled_ids
from src.utils.media import _source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "tests")]
import build_moments  # noqa: E402
import cache_fixtures as fx  # noqa: E402
from run_ad_vlm import finalize_generation  # noqa: E402


def over_budget(moment="a", temperature=0.0, tokens=5, max_new=64):
    return {"id": moment, "temperature": temperature,
            **finalize_generation("one two three", "r", tokens, 1, 0.1, max_new, 2)}


class DeterministicFailureTests(unittest.TestCase):
    def test_content_failures_at_temperature_zero_are_settled(self):
        rows = [
            over_budget("ad"),
            {"id": "flag", "temperature": 0.0, "max_new_tokens": 64, "over_budget": True, "hit_token_limit": False,
             "error": "word_budget: generated output exceeds the declared limit"},
            {"id": "qa", "temperature": 0.0, "max_new_tokens": 16, "hit_token_limit": False,
             "error": "invalid_answer: expected a single option letter"},
        ]
        for budget in (16, 512):  # a complete greedy decode does not change with the token budget
            with self.subTest(budget=budget):
                self.assertEqual(settled_ids(rows, budget, 0.0), ({"ad", "flag", "qa"},) * 2)
        self.assertEqual(settled_ids(rows)[0], set())  # without settings only successes count

    def test_transient_sampled_and_truncated_failures_stay_retryable(self):
        cases = {
            "sampled row": over_budget(temperature=0.7),
            "backend error": {"id": "a", "temperature": 0.0, "max_new_tokens": 64, "error": "generate: OOM"},
            "failed repair": {**over_budget(), "repair_error": "repair: RuntimeError()"},
            "cut off, larger budget": over_budget(tokens=32, max_new=32),
        }
        for name, row in cases.items():
            with self.subTest(case=name):
                self.assertEqual(settled_ids([row], 64, 0.0)[0], set())
        self.assertEqual(settled_ids([over_budget()], 64, 0.7)[0], set())  # now sampling

    def test_pipeline_flag_policy_does_not_call_the_model_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, run, exp = pipeline_case(Path(tmp), timeline=False, status="ok",
                                          generation={"word_limit_policy": "flag"})
            backend = CountingBackend("one two three four five six seven eight nine ten eleven")
            for _ in range(3):
                out = generation_stage.run_generation(cfg, run, exp, lambda spec: backend)
            self.assertEqual(backend.calls, 1)
            self.assertEqual(len(out.read_text().splitlines()), 1)


class CountingBackend:
    name = model_id = "counting"
    supports_audio = False

    def __init__(self, reply="A door."):
        self.reply, self.calls = reply, 0

    def generate(self, requests):
        self.calls += 1
        return [backends.GenResult(r.id, self.reply) for r in requests]


def pipeline_case(tmp, timeline, status, generation=None):
    source, frame = tmp / "v.mp4", tmp / "f.jpg"
    source.write_bytes(b"video")
    frame.write_bytes(b"jpg")
    run = matrix.GenerationRun(matrix._selection("U"), "P0", "m", json.dumps({"backend": "echo"}), timeline, "ad")
    moment = {"id": "v_U_000", "video_id": "v", "selector": "U", "time": 5.0, "frames": [str(frame)],
              "frame_times": [5.0], "word_limit": 9, "transcript": "hello there", "audio_timeline": "",
              "timeline_status": status, "media_fingerprint": _source_fingerprint(source)}
    exp = tmp / "exp"
    path = generation_stage.moments_path(exp, run.selection)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(moment) + "\n")
    return {"generation": generation or {}}, run, exp


class TimelineContextTests(unittest.TestCase):
    def generate(self, status, timeline=True):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, run, exp = pipeline_case(Path(tmp), timeline, status)
            backend = CountingBackend()
            out = generation_stage.run_generation(cfg, run, exp, lambda spec: backend)
            return json.loads(out.read_text().splitlines()[0]), backend.calls

    def test_a_core_timeline_run_without_cues_is_an_error_not_a_silent_no_audio_run(self):
        row, calls = self.generate("stale_audio_cues")
        self.assertEqual(row["error"], "unavailable_audio_timeline")
        self.assertEqual(calls, 0)

    def test_available_timeline_and_timeline_off_runs_still_generate(self):
        for status, timeline in (("ok", True), ("stale_audio_cues", False)):
            with self.subTest(status=status, timeline=timeline):
                row, calls = self.generate(status, timeline)
                self.assertNotIn("error", row)
                self.assertEqual(calls, 1)

    def test_qa_timeline_without_cues_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run = matrix.GenerationRun(matrix._selection("U"), "P0", "m", json.dumps({"backend": "echo"}), True, "qa")
            path = tmp / "selections" / f"{run.selection.run_id}.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"video_id": "v", "status": "ok", "times": [1.0], "duration": 9.0, "k": 1}) + "\n")
            item = SimpleNamespace(question_id="q", question="What?", options=["a", "b"], answer=0)
            backend = CountingBackend("A")
            frame = SimpleNamespace(path="f.jpg", timestamp=1.0)
            context = SimpleNamespace(audio_cues=None)
            with patch.object(generation_stage, "_current_media", return_value=({"frames": []}, None)), \
                    patch.object(generation_stage, "frame_window", return_value=[frame]), \
                    patch.object(generation_stage, "load_context", return_value=(context, {"audio_cues": "no_audio_cues"})):
                out = generation_stage.run_qa({"paths": {"cache_root": str(tmp)}}, run,
                                              [SimpleNamespace(video_id="v", qa=[item])], tmp, lambda spec: backend)
            row = json.loads(out.read_text().splitlines()[0])
            self.assertEqual(row["error"], "unavailable_audio_timeline: no_audio_cues")
            self.assertEqual(backend.calls, 0)


class ExperimentGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = {"paths": {"output_root": self.tmp.name}, "experiment": {"name": "g"},
                    "selection": {"num_frames": 4}}

    def test_only_output_code_is_hashed_with_repo_relative_paths(self):
        sources = xio.output_sources(ROOT)
        for included in ("scripts/run_experiment.py", "src/selection/two_stage.py", "src/generation/prompts.py",
                         "src/evaluation/qa.py", "src/evaluation/parsing.py"):
            self.assertIn(included, sources)
        for excluded in ("scripts/plot_hpc_dev_pilot.py", "src/evaluation/clip_text.py",
                         "src/evaluation/metrics.py", "src/experiments/report.py"):
            self.assertNotIn(excluded, sources)
        self.assertFalse(any(Path(name).is_absolute() for name in sources))

    def test_a_code_change_blocks_writing_stages_but_not_read_only_ones(self):
        with patch.object(xio, "output_sources", return_value={"src/a.py": "1"}):
            xio.experiment_dir(self.cfg)
            xio.experiment_dir(self.cfg)
        with patch.object(xio, "output_sources", return_value={"src/a.py": "2"}):
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                xio.experiment_dir(self.cfg)
            xio.experiment_dir(self.cfg, writes_outputs=False)

    def test_a_configuration_change_blocks_read_only_stages_too(self):
        xio.experiment_dir(self.cfg)
        changed = {**self.cfg, "selection": {"num_frames": 8}}
        for writes in (True, False):
            with self.subTest(writes=writes), self.assertRaisesRegex(ValueError, "configuration changed"):
                xio.experiment_dir(changed, writes_outputs=writes)

    def test_read_only_stages_accept_older_guards_and_never_create_one(self):
        path = xio.experiment_dir(self.cfg, writes_outputs=False)
        guard = path / "config.identity.json"
        self.assertFalse(guard.exists())
        old = {**self.cfg, "source_and_input_hashes": {"/old/clone/src/a.py": "1"}}
        guard.write_text(json.dumps(old))
        xio.experiment_dir(self.cfg, writes_outputs=False)
        with self.assertRaises(ValueError):
            xio.experiment_dir(self.cfg)


class MemoryNoveltyTests(unittest.TestCase):
    def test_one_video_with_bad_embeddings_is_reported_and_the_rest_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            visual = {"frame_times": [i * 0.5 for i in range(120)], "novelty": [0.1] * 120, "shots": []}
            good = np.random.default_rng(0).random((120, 4)) + 0.1
            fx.make_visual_cache(tmp, "bad", visual, good + 1.0)
            for path in (tmp / "bad").glob("visual_embeddings.*.npy"):
                path.write_bytes(b"corrupt")
            fx.make_visual_cache(tmp, "zero", visual, np.zeros((120, 4)))
            fx.make_visual_cache(tmp, "good", visual, good)
            splits = tmp / "splits.json"
            splits.write_text(json.dumps({"k_rule": {"rate_per_minute": 3}}))
            out = tmp / "m.jsonl"
            argv = ["build_moments.py", "--selector", "V", "--visual-novelty", "memory", "--splits", str(splits),
                    "--cache-dir", str(tmp), "--output", str(out)]
            targets = [("bad", 60, []), ("zero", 60, []), ("good", 60, [])]
            with patch.object(sys, "argv", argv), patch.object(build_moments, "load_targets", return_value=targets), \
                    patch.object(build_moments, "frame_window", return_value=[]), redirect_stdout(io.StringIO()):
                build_moments.main()
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            report = json.loads(out.with_suffix(".report.json").read_text())
            self.assertEqual({r["video_id"] for r in rows}, {"good"})
            self.assertEqual(len(rows), 3)
            status = {v["video_id"]: v["status"] for v in report["videos"]}
            self.assertEqual(status, {"bad": "missing_visual_embeddings", "zero": "missing_visual_embeddings",
                                      "good": "ok"})


class MinorFindingTests(unittest.TestCase):
    def test_malformed_bridge_sources_is_a_reason_not_an_exception(self):
        base = {"version": 1, "duration": 1.0, "streams": {}, "events": [], "speech_cues": [], "asr": None}
        self.assertIsNone(cues_io.malformed_cues({**base, "bridge_sources": {"/a": "hash"}}))
        for bad in ([], "x", None, {"/a": 1}):
            with self.subTest(bridge_sources=bad), tempfile.TemporaryDirectory() as tmp:
                record = {**base, "bridge_sources": bad}
                (Path(tmp) / cues_io.CUES_NAME).write_text(json.dumps(record))
                self.assertEqual(cues_io.read_cues(tmp), (None, cues_io.MALFORMED_CUES))

    def test_meta_keeps_provenance_for_every_attempt(self):
        counts = {"n_ok": 1, "n_error": 0, "n_hit_token_limit": 0}
        first = merge_meta(None, {"model_id": "m", "provenance": {"run": 1}}, counts)
        second = merge_meta(first, {"model_id": "m", "provenance": {"run": 2}}, counts)
        self.assertEqual(second["provenance"], {"run": 2})
        self.assertEqual([a["provenance"] for a in second["attempts"]], [{"run": 1}, {"run": 2}])

    def test_missing_openclip_weights_name_the_prefetch_command(self):
        hub = MagicMock()
        hub.hf_hub_download.side_effect = OSError("not cached")
        with patch.dict(sys.modules, {"open_clip": MagicMock(), "huggingface_hub": hub}):
            with self.assertRaisesRegex(RuntimeError, "hf download laion/CLIP-ViT-B-32"):
                clip_text.openclip_text_encoder(device="cpu")


if __name__ == "__main__":
    unittest.main()
