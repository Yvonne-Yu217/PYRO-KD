import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from src.datasets.adapters import ADEvent, QAItem, VideoRecord
from src.experiments import generation_stage as stage, io, matrix
from src.experiments.config import ConfigError
from src.experiments.selection_stage import selections_path
from src.generation.backends import EchoBackend
from src.utils.cache_checks import MEDIA_STALE
from src.utils.media import _source_fingerprint


class GenerationInputTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.exp = self.root / "exp"
        self.cache_root = self.root / "cache"
        self.cfg = {
            "paths": {"cache_root": str(self.cache_root)},
            "matrix": {"selectors": ["U"], "priors": ["P0"], "models": ["vl"]},
            "models": {"vl": {"backend": "echo"}, "native": {"backend": "echo"}},
            "ablations": {"timeline": {"selector": "U", "model": "vl", "audio_native_model": "native"}},
            "evaluation": {"resamples": 10},
        }
        self.run = matrix.core_generation_runs(self.cfg)[0]
        self.video = self.make_video("original")

    def make_video(self, video_id, *, audio=True):
        source = self.root / f"{video_id}.mp4"
        source.write_bytes(b"synthetic source video")
        cache = self.cache_root / video_id
        cache.mkdir(parents=True)
        frame = cache / "000001.jpg"
        frame.write_bytes(b"synthetic frame")
        wav = cache / "audio_16k.wav"
        if audio:
            wav.write_bytes(b"synthetic audio")
        io.write_json(cache / "manifest.json", {
            "source": str(source), "settings": {"source_fingerprint": _source_fingerprint(source)},
            "media_info": {"duration": 20.0, "has_audio": audio}, "audio_path": str(wav) if audio else None,
            "frames": [{"index": 0, "timestamp": 5.0, "path": str(frame)}],
        })
        return VideoRecord("synthetic", video_id, "validation", 20.0, source,
                           ad=[ADEvent(5.0, None, "A lamp lights up.")],
                           transcript=[{"start": 4.0, "end": 6.0, "text": "nearby transcript marker"}],
                           qa=[QAItem(f"{video_id}_q", video_id, "What happens?", ("nothing", "A lamp lights up"), 1)])

    def select(self, video, t=5.0):
        io.append_jsonl(selections_path(self.exp, self.run.selection), [
            {"video_id": video.video_id, "status": "ok", "times": [t], "duration": 20.0, "k": 1},
        ])

    def factory(self, reply="A lamp lights up."):
        backend = EchoBackend(config={"reply": reply})
        backend.generate = Mock(wraps=backend.generate)
        return Mock(return_value=backend), backend

    def test_moments_and_qa_skip_stale_complete_media_and_continue(self):
        healthy = self.make_video("healthy")
        for video in (self.video, healthy):
            self.select(video)
        stale_cache = self.cache_root / self.video.video_id
        snapshot = {p.name: p.read_bytes() for p in stale_cache.iterdir()}
        # A failed rebuild leaves the previous complete cache at the same path.
        self.video.video_path.write_bytes(b"different replacement video bytes")
        videos = [self.video, healthy]
        path = stage.build_moments(self.cfg, self.run.selection, videos, self.exp)
        self.assertEqual([row["video_id"] for row in io.read_jsonl(path)], [healthy.video_id])
        report = json.loads(path.with_suffix(".report.json").read_text(encoding="utf-8"))
        self.assertEqual([(row["status"], row["written"]) for row in report["videos"]], [(MEDIA_STALE, 0), ("ok", 1)])
        factory, backend = self.factory("B")
        qa = matrix.GenerationRun(self.run.selection, "P0", "vl", self.run.model_json, False, "qa")
        rows = io.read_jsonl(stage.run_qa(self.cfg, qa, videos, self.exp, factory))
        self.assertEqual([(row["status"], row.get("reason")) for row in rows], [("no_frames", MEDIA_STALE), ("answered", None)])
        self.assertEqual(backend.generate.call_count, 1)
        self.assertEqual({p.name: p.read_bytes() for p in stale_cache.iterdir()}, snapshot)

    def test_old_prepared_moments_do_not_send_frames_after_source_change(self):
        self.select(self.video)
        stage.build_moments(self.cfg, self.run.selection, [self.video], self.exp)
        self.video.video_path.write_bytes(b"different replacement video bytes")
        factory = Mock(side_effect=AssertionError("stale frames must not be sent"))
        with self.assertRaisesRegex(ConfigError, "Media is stale or unverifiable"):
            stage.run_generation(self.cfg, self.run, self.exp, factory)
        factory.assert_not_called()

    def test_changed_timestamp_cannot_reuse_or_score_previous_success(self):
        self.select(self.video)
        videos = [self.video]
        stage.build_moments(self.cfg, self.run.selection, videos, self.exp)
        factory, _ = self.factory()
        output = stage.run_generation(self.cfg, self.run, self.exp, factory)
        original = output.read_bytes()
        forbidden = Mock(side_effect=AssertionError("unchanged runs should resume without a backend"))
        stage.run_generation(self.cfg, self.run, self.exp, forbidden)
        forbidden.assert_not_called()
        self.assertEqual(stage.evaluate_generation(self.cfg, self.run, videos, self.exp, {})["macro_f1"]["mean"], 1.0)

        self.select(self.video, t=12.0)
        path = stage.build_moments(self.cfg, self.run.selection, videos, self.exp)
        self.assertEqual(io.read_jsonl(path)[0]["id"], io.read_jsonl(output)[0]["id"])
        with self.assertRaisesRegex(ConfigError, "Generation inputs changed"):
            stage.run_generation(self.cfg, self.run, self.exp, forbidden)
        scorer = Mock()
        with self.assertRaisesRegex(ConfigError, "Generation inputs changed"):
            stage.evaluate_generation(self.cfg, self.run, videos, self.exp, {"mock": (scorer, None)})
        forbidden.assert_not_called()
        scorer.assert_not_called()
        self.assertEqual(output.read_bytes(), original)

    def test_changed_frame_bytes_at_same_path_invalidate_previous_success(self):
        self.select(self.video)
        stage.build_moments(self.cfg, self.run.selection, [self.video], self.exp)
        factory, _ = self.factory()
        stage.run_generation(self.cfg, self.run, self.exp, factory)
        (self.cache_root / self.video.video_id / "000001.jpg").write_bytes(b"replacement frame")
        with self.assertRaisesRegex(ConfigError, "Generation inputs changed"):
            stage.run_generation(self.cfg, self.run, self.exp, factory)
        self.assertEqual(factory.call_count, 1)

    def test_unverified_legacy_success_is_refused_but_unverified_failure_is_retryable(self):
        self.select(self.video)
        path = stage.build_moments(self.cfg, self.run.selection, [self.video], self.exp)
        moment = io.read_jsonl(path)[0]
        output = stage.generations_path(self.exp, self.run)
        io.append_jsonl(output, [{"id": moment["id"], "output": "legacy success"}])
        factory, _ = self.factory()
        with self.assertRaisesRegex(ConfigError, "cannot be verified"):
            stage.run_generation(self.cfg, self.run, self.exp, factory)
        output.unlink()
        io.append_jsonl(output, [{"id": moment["id"], "error": "token limit", "hit_token_limit": True,
                                  "max_new_tokens": 64, "temperature": 0}])
        stage.run_generation(self.cfg, self.run, self.exp, factory)
        self.assertEqual(len(io.read_jsonl(output)), 2)
        self.assertIn("input_fingerprint", io.read_jsonl(output)[1])

    def test_ablation_requests_have_separate_context_and_output_ids(self):
        self.select(self.video)
        path = stage.build_moments(self.cfg, self.run.selection, [self.video], self.exp)
        moment = io.read_jsonl(path)[0]
        moment["audio_timeline"] = "cached timeline marker"
        moment["timeline_status"] = "ok"
        path.unlink()
        io.append_jsonl(path, [moment])
        runs = [self.run, *matrix.ablation5_runs(self.cfg)]
        self.assertEqual(len({run.run_id for run in runs}), len(runs))
        self.assertEqual([run.audio_context for run in runs], [None, "none", "timeline", "native"])
        for run, transcript, timeline, audio in zip(runs, [True, False, False, False],
                                                     [False, False, True, False], [True, False, False, True]):
            with self.subTest(mode=run.audio_context):
                factory, backend = self.factory()
                output = stage.run_generation(self.cfg, run, self.exp, factory)
                (request,) = backend.generate.call_args.args[0]
                self.assertEqual("nearby transcript marker" in request.prompt, transcript)
                self.assertEqual("cached timeline marker" in request.prompt, timeline)
                self.assertEqual(request.audio, moment["audio_clip"] if audio else None)
                self.assertEqual(io.read_jsonl(output)[0]["audio_context"], run.audio_context)

    def test_native_ablation_records_unavailable_audio_without_sending_an_image_only_request(self):
        video = self.make_video("silent", audio=False)
        self.select(video)
        self.select(self.video)
        stage.build_moments(self.cfg, self.run.selection, [video, self.video], self.exp)
        native = matrix.ablation5_runs(self.cfg)[-1]
        factory, backend = self.factory()
        output = stage.run_generation(self.cfg, native, self.exp, factory)
        (request,) = backend.generate.call_args.args[0]
        self.assertTrue(request.audio)
        rows = {row["video_id"]: row for row in io.read_jsonl(output)}
        self.assertEqual(rows[video.video_id]["error"], "no_native_audio")
        self.assertNotIn("error", rows[self.video.video_id])

    def test_missing_timeline_is_reported_without_running_the_no_context_condition(self):
        self.select(self.video)
        stage.build_moments(self.cfg, self.run.selection, [self.video], self.exp)
        timeline = matrix.ablation5_runs(self.cfg)[1]
        factory = Mock(side_effect=AssertionError("missing timeline should not instantiate a model"))
        output = stage.run_generation(self.cfg, timeline, self.exp, factory)
        (row,) = io.read_jsonl(output)
        self.assertEqual(row["error"], "unavailable_audio_timeline")
        self.assertEqual(row["timeline_status"], "no_vad_features")
        factory.assert_not_called()
