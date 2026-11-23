"""Regressions for the review of the experiment pipeline (#22).

Each class pins one finding: dry-run output isolation, run IDs ignoring cost/throughput settings, the
audio-only A_cue, retrying selections whose inputs were missing, API failure isolation and the shared
spend ledger, QA failure isolation, nested audio-cue validation, the locked status file, and the
AV-disagreement term in memory mode.
"""

import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.audio import cues_io
from src.experiments import config as cfgmod
from src.experiments import generation_stage, io, matrix
from src.experiments.data import clear_context_cache
from src.generation import backends
from src.selection import registry, two_stage
from src.selection.cues import CueBundle, make_grid
from src.utils import audio_features as af
from src.utils import visual_features as vf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiment  # noqa: E402
import test_experiment_integration as integ  # noqa: E402


def write_dataset(root: Path, video_id="talky"):
    duration, speech, scenes, gt = integ.VIDEOS[video_id]
    data, cache_root, out = root / "mtg", root / "cache", root / "out"
    cache, manifest = integ.make_cache(cache_root, data, video_id, duration, speech, scenes)
    data.mkdir(exist_ok=True)
    row = {"video_id": video_id, "split": "validation", "category": "Vlog", "duration_seconds": duration,
           "gt_ad_segments": [{"start_seconds": t, "text": f"A person does thing {i}."} for i, t in enumerate(gt)],
           "transcript_segments": [], "visual_info": None}
    (data / "validation.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (data / "test.jsonl").write_text("", encoding="utf-8")
    return data, cache_root, out, cache, manifest, speech, scenes


def cache_features(cache, manifest, speech, scenes):
    """Write the VAD features, audio cues and visual features a selector needs."""
    with patch.object(af, "detect_speech", lambda w, sr, cfg, sp=speech: list(sp)):
        features = af.compute_audio_features(cache / "audio_16k.wav")
    af.write_features(cache, features)
    cues_io.write_cues(cache, cues_io.compute_audio_cues(cache / "audio_16k.wav", features, integ.fake_backends()))
    visual, embeddings = vf.compute_visual_features(manifest, integ.colour_embedder, lambda p, th, ms, s=scenes: list(s))
    vf.write_features(cache, visual, embeddings)


def write_config(root: Path, data, cache_root, out, selectors, reply="A person walks in.", extra=""):
    path = root / "exp.toml"
    path.write_text(f"""
[experiment]
name = "review"
seed = 0
[dataset]
name = "mtg-ad"
root = "{data}"
split = "validation"
subset = "all"
[paths]
cache_root = "{cache_root}"
output_root = "{out}"
[k_rule]
rate_per_minute = 6.0
[matrix]
selectors = {json.dumps(selectors)}
priors = ["P0"]
models = ["m1"]
[models.m1]
backend = "echo"
reply = "{reply}"
[evaluation]
resamples = 50
text_metrics = []
{extra}
""")
    return path


class DryRunIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        data, cache_root, out, cache, manifest, speech, scenes = write_dataset(self.root)
        cache_features(cache, manifest, speech, scenes)
        self.config = write_config(self.root, data, cache_root, out, ["U"], reply="A real model reply.")
        self.exp = out / "review"
        clear_context_cache()

    def stage(self, name, *extra):
        return run_experiment.main(["--config", str(self.config), "--stage", name, *extra])

    def test_dry_run_rows_never_reach_the_real_runs_files_or_resume_state(self):
        self.assertEqual(self.stage("select"), 0)
        self.assertEqual(self.stage("moments"), 0)
        self.assertEqual(self.stage("generate", "--dry-run"), 0)
        self.assertEqual(list((self.exp / "generations").glob("*.jsonl")), [])  # nothing at the real path
        dry = list((self.exp / "generations" / "dry_run").glob("*.jsonl"))
        self.assertEqual(len(dry), 1)
        self.assertIn("DRY RUN", dry[0].read_text())
        # The real run must generate every moment itself instead of treating the dry rows as done.
        self.assertEqual(self.stage("generate"), 0)
        real = list((self.exp / "generations").glob("*.jsonl"))
        self.assertEqual(len(real), 1)
        rows = [json.loads(line) for line in real[0].read_text().splitlines()]
        self.assertTrue(rows)
        self.assertTrue(all(r["model_output"] == "A real model reply." for r in rows))
        self.assertNotIn("DRY RUN", real[0].read_text())
        status = json.loads((self.exp / "status.json").read_text())
        self.assertIn("generate_dry_run", status)
        self.assertIn("generate", status)

    def test_evaluate_generation_ignores_dry_run_output(self):
        self.stage("select")
        self.stage("moments")
        self.stage("generate", "--dry-run")
        self.assertEqual(self.stage("evaluate-generation"), 0)
        self.assertEqual(list((self.exp / "eval").glob("generation_*.json")), [])  # no real run to evaluate


class RunIdTests(unittest.TestCase):
    def run_id(self, **model):
        sel = matrix._selection("U")
        return matrix.GenerationRun(sel, "P0", "m", json.dumps({"backend": "openai", "model": "x", **model}, sort_keys=True),
                                    False).run_id

    def test_cost_and_throughput_settings_do_not_fork_the_run(self):
        base = self.run_id()
        for change in ({"spend_cap_usd": 50.0}, {"prices": {"input_per_million": 1, "output_per_million": 2}},
                       {"ledger_path": "/x/ledger.json"}, {"timeout_s": 5}, {"max_retries": 9},
                       {"request_batch": 2}, {"batch_size": 8}, {"api_key_env": "OTHER_KEY"}):
            with self.subTest(change=change):
                self.assertEqual(self.run_id(**change), base)

    def test_result_affecting_settings_still_fork_it(self):
        base = self.run_id()
        for change in ({"model": "y"}, {"quant": "4bit"}, {"reply": "different"}, {"base_url": "http://elsewhere"}):
            with self.subTest(change=change):
                self.assertNotEqual(self.run_id(**change), base)


class AudioOnlyACueTests(unittest.TestCase):
    def context(self, with_visual):
        times = [i * 0.5 for i in range(80)]
        visual = {"frame_times": times, "novelty": [0.0, 0.9] * 40, "shots": [10.0, 20.0]}
        embeddings = np.array([[1.0, 0.0] if i % 2 else [0.0, 1.0] for i in range(80)], dtype=np.float32)
        features = {"speech": [[0.0, 5.0], [30.0, 35.0]], "gaps": [{"start": 5.0, "end": 30.0, "rms": 0.0, "onset": 0.0}],
                    "config_hash": "h"}
        return registry.SelectionContext(
            "v", 40.0, audio_features=features, visual_features=visual if with_visual else None,
            embeddings=embeddings if with_visual else None)

    def test_which_specs_read_visual_features(self):
        expected = {"U": False, "O": False, "A": False, "V": True, "AV": True, "A_cue": False, "V_mem": True}
        for name, uses in expected.items():
            with self.subTest(selector=name):
                self.assertEqual(registry.spec_uses_visual(name), uses)

    def test_a_cue_ignores_a_visual_cache_entirely(self):
        with_visual = registry.run_selector("A_cue", self.context(True), 4)
        without = registry.run_selector("A_cue", self.context(False), 4)
        self.assertEqual(with_visual.times, without.times)
        # The bundle it ran on had no CLIP embeddings or visual novelty, so MMR used the temporal kernel.
        self.assertIn("embeddings", with_visual.diagnostics["streams_missing"])
        self.assertIsNone(self.context(True).bundle(use_visual=False).vn)
        self.assertIsNone(self.context(True).bundle(use_visual=False).embeddings)
        self.assertIsNotNone(self.context(True).bundle().embeddings)

    def test_visual_selectors_still_get_the_visual_bundle(self):
        sel = registry.run_selector("AV", self.context(True), 4)
        self.assertNotIn("embeddings", sel.diagnostics["streams_missing"])


class RetryMissingInputsTests(unittest.TestCase):
    def test_selections_with_missing_inputs_are_recomputed_once_caches_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, cache_root, out, cache, manifest, speech, scenes = write_dataset(root)
            config = write_config(root, data, cache_root, out, ["U", "AV"])
            exp = out / "review"

            def stage(name):
                clear_context_cache()
                return run_experiment.main(["--config", str(config), "--stage", name])

            self.assertEqual(stage("select"), 0)
            av_path = next((exp / "selections").glob("AV-*.jsonl"))
            first = [json.loads(line) for line in av_path.read_text().splitlines()]
            self.assertEqual(first[-1]["status"], "missing_visual_features")
            stage("select")  # still missing: an unchanged result is not appended again
            self.assertEqual(av_path.read_text().count("\n"), 1)
            cache_features(cache, manifest, speech, scenes)  # the caches are built later
            stage("select")
            rows = [json.loads(line) for line in av_path.read_text().splitlines()]
            self.assertEqual(rows[-1]["status"], "ok")
            self.assertTrue(rows[-1]["times"])
            self.assertEqual(len(rows), 2)
            stage("select")  # settled: nothing recomputed or appended
            self.assertEqual(av_path.read_text().count("\n"), 2)

    def test_recompute_policy(self):
        from src.experiments.selection_stage import _recompute

        self.assertFalse(_recompute("ok", True))
        self.assertTrue(_recompute("missing_vad", False))
        self.assertTrue(_recompute("missing_visual_features", False))
        self.assertTrue(_recompute("ok_degraded", False))
        self.assertFalse(_recompute("error", False))
        self.assertTrue(_recompute("error", True))


def spec(tmp, **extra):
    return {"spend_cap_usd": 5.0, "ledger_path": f"{tmp}/ledger.json",
            "prices": {"input_per_million": 1.0, "output_per_million": 4.0, "image_tokens": 1000}, "max_retries": 0, **extra}


class ApiFailureIsolationTests(unittest.TestCase):
    def backend(self, tmp, transport, **extra):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}):
            return backends.OpenAIBackend("m", spec(tmp, **extra), transport=transport)

    def ok_reply(self):
        return {"choices": [{"message": {"content": "A dog."}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3}}

    def test_connection_errors_are_recorded_per_request_and_the_batch_continues(self):
        for name, exc in (("reset", ConnectionResetError("reset")), ("incomplete", http.client.IncompleteRead(b"x")),
                          ("os error", OSError("socket")), ("bad usage", ValueError("int()"))):
            with self.subTest(failure=name), tempfile.TemporaryDirectory() as tmp:
                calls = []

                def transport(url, payload, headers, timeout, exc=exc):
                    calls.append(1)
                    if len(calls) == 1:
                        raise exc
                    return self.ok_reply()

                results = self.backend(tmp, transport).generate(
                    [backends.GenRequest(str(i), "describe", [], None, 8) for i in range(3)])
                self.assertEqual(len(results), 3)  # no exception escaped generate()
                self.assertTrue(results[0].error.startswith("request:"))
                self.assertTrue(all(r.error is None for r in results[1:]))

    def test_a_lost_reply_is_charged_conservatively_but_a_pre_send_failure_is_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            def lost(url, payload, headers, timeout):
                raise ConnectionResetError("reset after the request was accepted")

            result = self.backend(tmp, lost).generate([backends.GenRequest("a", "describe", [], None, 8)])[0]
            self.assertIsNotNone(result.error)
            spent = backends.SpendLedger(f"{tmp}/ledger.json", 5.0).spent()
            self.assertGreater(spent, 0.0)  # it may have been billed: the worst-case reservation counts

        with tempfile.TemporaryDirectory() as tmp:
            sent = []
            result = self.backend(tmp, lambda *a: sent.append(1) or self.ok_reply()).generate(
                [backends.GenRequest("a", "describe", ["/no/such/frame.jpg"], None, 8)])[0]
            self.assertIn("frame unreadable", result.error)
            self.assertEqual(sent, [])  # nothing was sent, and it was not retried
            self.assertEqual(backends.SpendLedger(f"{tmp}/ledger.json", 5.0).spent(), 0.0)


class SharedLedgerTests(unittest.TestCase):
    def test_a_relative_or_missing_ledger_path_is_refused(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=False):
            os.environ.pop("PYRO_API_LEDGER", None)
            prices = {"input_per_million": 1, "output_per_million": 1}
            for ledger in (None, "outputs/api_spend_ledger.json", "ledger.json"):
                with self.subTest(ledger=ledger):
                    cfg = {"spend_cap_usd": 1, "prices": prices, **({"ledger_path": ledger} if ledger else {})}
                    with self.assertRaises(ValueError):
                        backends.OpenAIBackend("m", cfg)
            with tempfile.TemporaryDirectory() as tmp:
                backends.OpenAIBackend("m", {"spend_cap_usd": 1, "prices": prices, "ledger_path": f"{tmp}/l.json"})

    def test_the_runner_gives_every_api_job_the_experiment_ledger(self):
        seen = []
        with patch.object(backends, "build_backend", side_effect=lambda spec: seen.append(spec) or object()):
            factory = run_experiment._backend_factory(False, Path("/exp/api_spend_ledger.json"))
            factory({"backend": "openai", "model": "x"})
            factory({"backend": "gemini", "model": "y", "ledger_path": "/custom/ledger.json"})
            factory({"backend": "hf_vlm", "model": "z"})
        self.assertEqual(seen[0]["ledger_path"], "/exp/api_spend_ledger.json")
        self.assertEqual(seen[1]["ledger_path"], "/custom/ledger.json")  # an explicit path wins
        self.assertNotIn("ledger_path", seen[2])  # local models have no ledger


class QaFailureIsolationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cache_root = self.root / "cache"
        duration, speech, scenes, _ = integ.VIDEOS["talky"]
        integ.make_cache(self.cache_root, self.root / "mtg", "talky", duration, speech, scenes)
        self.exp = self.root / "exp"
        self.cfg = {"paths": {"cache_root": str(self.cache_root)}, "qa": {"max_frames": 2}}
        self.run = matrix.GenerationRun(matrix._selection("U"), "P0", "m", json.dumps({"backend": "echo"}), False, "qa")
        rows = [{"video_id": v, "status": "ok", "times": [10.0, 20.0], "duration": 40.0, "k": 2} for v in ("ghost", "talky")]
        path = self.exp / "selections" / f"{self.run.selection.run_id}.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        item = lambda q: SimpleNamespace(question_id=q, question="What?", options=["a", "b"], answer=1)
        self.records = [SimpleNamespace(video_id="ghost", qa=[item("ghost_q")]),
                        SimpleNamespace(video_id="talky", qa=[item("talky_q1"), item("talky_q2")])]

    def rows(self, dry_run=False):
        path = generation_stage.generations_path(self.exp, self.run, dry_run)
        return {r["question_id"]: r for r in map(json.loads, path.read_text().splitlines())}

    def test_a_video_without_a_cache_is_recorded_and_the_rest_are_still_answered(self):
        factory = lambda spec: backends.EchoBackend("echo", {"reply": "B"})
        generation_stage.run_qa(self.cfg, self.run, self.records, self.exp, factory)
        rows = self.rows()
        self.assertEqual(rows["ghost_q"]["status"], "no_frames")
        self.assertEqual(rows["ghost_q"]["reason"], "no_manifest")
        self.assertEqual(rows["talky_q1"]["status"], "answered")
        self.assertEqual(rows["talky_q2"]["prediction"], 1)

    def test_a_backend_exception_is_one_error_row_not_an_aborted_stage(self):
        class Flaky:
            name, model_id, supports_audio = "flaky", "flaky", False

            def __init__(self):
                self.calls = 0

            def generate(self, requests):
                self.calls += 1
                if self.calls == 1:
                    raise ConnectionResetError("boom")
                return [backends.GenResult(r.id, "B") for r in requests]

        backend = Flaky()
        generation_stage.run_qa(self.cfg, self.run, self.records[1:], self.exp, lambda spec: backend)
        rows = self.rows()
        self.assertEqual(rows["talky_q1"]["status"], "error")
        self.assertIn("ConnectionResetError", rows["talky_q1"]["error"])
        self.assertEqual(rows["talky_q2"]["status"], "answered")
        # The failed question is retried on the next run; the answered one is not asked again.
        generation_stage.run_qa(self.cfg, self.run, self.records[1:], self.exp, lambda spec: backend)
        self.assertEqual(backend.calls, 3)

    def test_dry_run_qa_uses_its_own_file(self):
        factory = lambda spec: backends.EchoBackend("echo", {"reply": "B"})
        generation_stage.run_qa(self.cfg, self.run, self.records, self.exp, factory, dry_run=True)
        self.assertFalse(generation_stage.generations_path(self.exp, self.run).exists())
        self.assertTrue(generation_stage.generations_path(self.exp, self.run, True).is_file())
        result = generation_stage.evaluate_qa(self.cfg, [self.run], self.exp)
        self.assertEqual(result["runs"][self.run.run_id]["n"], 0)  # the real file does not exist


class CueValidationTests(unittest.TestCase):
    def record(self, **changes):
        record = {"version": 1, "duration": 30.0, "streams": {"asr": "ok"}, "tagger": None, "clap": None, "acoustic": None,
                  "events": [{"time": 5.0, "strength": 0.8, "label": "Music", "superclass": "music"}],
                  "speech_cues": [{"time": 6.0, "strength": 1.0, "kind": "deictic", "text": "look at this"}],
                  "asr": {"source": "dataset", "segments": [{"start": 1.0, "end": 3.0, "text": "look at this", "words": [
                      {"start": 1.0, "end": 1.4, "word": "look"}]}]}}
        record.update(changes)
        return record

    def test_a_well_formed_record_passes(self):
        self.assertIsNone(cues_io.malformed_cues(self.record()))

    def test_rows_missing_fields_that_consumers_index_are_rejected(self):
        good_event = {"time": 5.0, "strength": 0.8, "label": "Music", "superclass": "music"}
        good_cue = {"time": 6.0, "strength": 1.0, "kind": "deictic"}
        cases = {
            "speech cue without kind": {"speech_cues": [{"time": 6.0, "strength": 1.0}]},
            "speech cue with null kind": {"speech_cues": [{**good_cue, "kind": None}]},
            "event without superclass": {"events": [{k: v for k, v in good_event.items() if k != "superclass"}]},
            "event without label": {"events": [{k: v for k, v in good_event.items() if k != "label"}]},
            "asr segment without start": {"asr": {"segments": [{"end": 2.0, "text": "x"}]}},
            "asr segment reversed": {"asr": {"segments": [{"start": 3.0, "end": 2.0, "text": "x"}]}},
            "asr segment text not a string": {"asr": {"segments": [{"start": 1.0, "end": 2.0, "text": 5}]}},
            "asr segment not an object": {"asr": {"segments": ["look at this"]}},
            "asr word without start": {"asr": {"segments": [{"start": 1.0, "end": 2.0, "text": "x", "words": [{"end": 1.5}]}]}},
            "asr words not a list": {"asr": {"segments": [{"start": 1.0, "end": 2.0, "text": "x", "words": "look"}]}},
            "asr segment time too large for a float": {"asr": {"segments": [{"start": 1.0, "end": 10 ** 400, "text": "x"}]}},
            "event time too large for a float": {"events": [{"time": 10 ** 400, "strength": 1.0, "label": "Music", "superclass": "music"}]},
        }
        for name, change in cases.items():
            with self.subTest(defect=name):
                self.assertIsNotNone(cues_io.malformed_cues(self.record(**change)))

    def test_a_stamped_but_malformed_file_is_unavailable_not_an_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = self.record(speech_cues=[{"time": 6.0, "strength": 1.0}])
            (Path(tmp) / cues_io.CUES_NAME).write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(cues_io.read_cues(tmp), (None, cues_io.MALFORMED_CUES))


class StatusLockTests(unittest.TestCase):
    def test_concurrent_status_updates_are_not_lost(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            real_write = io.write_json
            at_pause, other_done = threading.Event(), threading.Event()
            armed = [True]

            def slow_write(path, data):
                if threading.current_thread().name == "first" and armed[0]:
                    armed[0] = False
                    at_pause.set()
                    other_done.wait(0.5)  # a second job runs now; with the lock it must wait instead
                return real_write(path, data)

            def update(name, done=None):
                io.update_status(exp, "select", name, {"n": 1})
                if done:
                    done.set()

            with patch.object(io, "write_json", slow_write):
                first = threading.Thread(target=update, args=("run-a",), name="first")
                first.start()
                self.assertTrue(at_pause.wait(10))
                second = threading.Thread(target=update, args=("run-b", other_done), name="second")
                second.start()
                first.join(30)
                second.join(30)
            self.assertFalse(first.is_alive() or second.is_alive(), "deadlocked")
            status = json.loads((exp / "status.json").read_text())
            self.assertEqual(sorted(status["select"]), ["run-a", "run-b"])


class MemoryModeDisagreementTests(unittest.TestCase):
    def bundle(self):
        n = 41
        grid = make_grid(20.0, 0.5)
        vn = [0.1] * n
        vn[11], vn[30] = 1.0, 0.9  # the visual peak beside the first audio event, and on the second
        music = [0.0] * n
        music[10], music[30] = 0.9, 0.5
        frame_times = list(grid)
        embeddings = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (n, 1))
        return CueBundle(duration=20.0, hop=0.5, grid=grid, vn=vn, frame_times=frame_times, embeddings=embeddings,
                         nse={"music": music})

    def options(self, **extra):
        return {"placement": "event", "require_vad": False, "candidate_streams": ["nse"], "fill": "none",
                "weights": {"vn": 1.0, "nse": 0.5, "sc": 0.0, "gf": 0.0, "op": 0.0, "av_disagreement": 1.0}, **extra}

    def test_memory_mode_with_no_memory_weight_ranks_like_local_mode(self):
        local = two_stage.select(self.bundle(), 1, self.options(novelty_mode="local"))
        memory = two_stage.select(self.bundle(), 1, self.options(novelty_mode="memory", memory_blend=0.0))
        self.assertEqual(local.times, [5.0])
        # blend 0 leaves VN unchanged, so the disagreement term must be unchanged too; it was recomputed
        # at the event's audio-change bin instead of the bin where VN was read, which penalized this pick.
        self.assertEqual(memory.times, local.times)

    def test_candidates_record_the_audio_change_their_disagreement_used(self):
        candidates, _ = two_stage.generate_candidates(self.bundle(), two_stage.merge_config(self.options()))
        first = next(c for c in candidates if c.event_time == 5.0)
        self.assertEqual(first.audio_change, 0.0)  # read at the VN peak bin (5.5 s), not at the event bin
        self.assertAlmostEqual(first.avd, first.vn * (1.0 - first.audio_change))


if __name__ == "__main__":
    unittest.main()
