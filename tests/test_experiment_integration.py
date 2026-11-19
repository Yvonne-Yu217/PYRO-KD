"""End-to-end experiment pipeline on synthetic data (no ffmpeg, no MLLM, no network).

Media caches are written exactly as preprocess_video lays them out (real JPEG
frames at 2 fps, a real 16 kHz WAV, a manifest with the source fingerprint),
so every cache validator runs for real. Audio features come from the real
compute_audio_features (real Silero VAD on the speech-free clip; the speech
segments of the talky clip are injected because synthetic tones contain no
speech). Visual features use a colour-mean embedder in place of OpenCLIP. The
generation backend is the deterministic echo backend.
"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.audio import cues_io, speech_cues
from src.datasets.adapters import CanonicalAdapter, MTGADAdapter
from src.experiments import config as cfgmod
from src.experiments import matrix
from src.experiments.config import ConfigError
from src.generation import backends
from src.utils import audio_features as af
from src.utils import visual_features as vf
from src.utils.media import _source_fingerprint, config_hash

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_experiment  # noqa: E402

SR = 16000
VIDEOS = {
    # video_id: (duration, speech segments or None for no audio, scene changes, GT AD starts)
    "talky": (60.0, [(0.0, 12.0), (15.0, 30.0), (34.0, 50.0)], [13.0, 31.0, 52.0], [13.5, 31.0, 52.0, 56.0]),
    "silent": (40.0, [], [10.0, 25.0], [10.0, 25.0, 35.0]),
    "noaudio": (30.0, None, [15.0], [5.0, 15.0]),
}


def make_cache(root: Path, data: Path, video_id: str, duration: float, speech, scenes):
    from PIL import Image
    import soundfile as sf

    source = data / "videos" / f"{video_id}.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"synthetic " + video_id.encode())
    cache = root / video_id
    (cache / "frames").mkdir(parents=True)
    frames = []
    palette = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30)]
    for i in range(int(duration * 2)):
        t = i / 2.0
        colour = palette[sum(t >= s for s in scenes) % len(palette)]
        path = cache / "frames" / f"{i + 1:06d}.jpg"
        Image.new("RGB", (32, 18), colour).save(path)
        frames.append({"index": i, "timestamp": round(t, 6), "path": str(path)})
    audio_path = None
    if speech is not None:
        audio_path = cache / "audio_16k.wav"
        tt = np.arange(int(duration * SR)) / SR
        wave = np.zeros_like(tt)
        for s, e in speech:
            wave[(tt >= s) & (tt < e)] = 0.2 * np.sin(2 * np.pi * 220 * tt[(tt >= s) & (tt < e)])
        music = (tt >= duration * 0.5) & (tt < duration * 0.5 + 4)
        wave[music] += 0.4 * np.sin(2 * np.pi * 880 * tt[music])
        sf.write(audio_path, wave.astype(np.float32), SR, subtype="PCM_16")
    settings = {"fps": 2.0, "sample_rate": SR, "max_side": 768, "manifest_version": 2,
                "source_fingerprint": _source_fingerprint(source)}
    manifest = {"video_id": video_id, "source": str(source.resolve()),
                "media_info": {"duration": duration, "has_audio": speech is not None},
                "settings": settings, "config_hash": config_hash(settings),
                "audio_path": str(audio_path) if audio_path else None, "frames": frames}
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache, manifest


def colour_embedder(paths):
    from PIL import Image

    return np.array([np.asarray(Image.open(p).convert("RGB"), dtype=np.float64).mean(axis=(0, 1)) for p in paths])


def fake_backends():
    def tagger(windows, sr):
        return np.array([[float(np.abs(w).max() > 0.3), 0.0] for w in windows], dtype=np.float32)

    def transcriber(chunk):
        return [{"start": 0.0, "end": 1.0, "text": "look at this", "words": [
            {"start": 0.1, "end": 0.3, "word": "look"}, {"start": 0.35, "end": 0.5, "word": "at"},
            {"start": 0.55, "end": 0.8, "word": "this"}]}]

    return cues_io.CueBackends(transcriber=transcriber, transcriber_name="fake", tagger=tagger,
                               tagger_labels=["Music", "Speech"], tagger_name="fake-ast",
                               text_encoder=speech_cues.lexical_encoder(), text_encoder_name="lexical")


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.data, cls.cache_root, cls.out = root / "mtg", root / "cache", root / "out"
        rows = []
        for vid, (duration, speech, scenes, gt) in VIDEOS.items():
            cache, manifest = make_cache(cls.cache_root, cls.data, vid, duration, speech, scenes)
            rows.append({"video_id": vid, "split": "validation", "category": "Vlog", "duration_seconds": duration,
                         "gt_ad_segments": [{"start_seconds": t, "text": f"A person does thing {i}."} for i, t in enumerate(gt)],
                         "transcript_segments": [{"start_seconds": s, "end_seconds": e, "text": "look at this over there"}
                                                 for s, e in (speech or [])], "visual_info": None})
            if speech is not None:
                if speech:
                    with patch.object(af, "detect_speech", lambda w, sr, cfg, sp=speech: list(sp)):
                        features = af.compute_audio_features(cache / "audio_16k.wav")
                else:
                    features = af.compute_audio_features(cache / "audio_16k.wav")  # real Silero VAD
                af.write_features(cache, features)
                cues_io.write_cues(cache, cues_io.compute_audio_cues(cache / "audio_16k.wav", features, fake_backends()))
            visual, embeddings = vf.compute_visual_features(manifest, colour_embedder, lambda p, th, ms, s=scenes: list(s))
            vf.write_features(cache, visual, embeddings)
        (cls.data / "validation.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (cls.data / "test.jsonl").write_text("", encoding="utf-8")
        cls.exemplars = root / "exemplars.json"
        cls.exemplars.write_text(json.dumps({"version": "test", "exemplars": [
            {"text": "A woman pours tea into a cup.", "video_id": "pool1", "time": 1.0},
            {"text": "Two dogs run across a field.", "video_id": "pool2", "time": 2.0},
            {"text": "He opens the red door.", "video_id": "pool3", "time": 3.0}]}))
        cls.config = root / "exp.toml"
        cls.config.write_text(f"""
[experiment]
name = "synthetic"
seed = 0
[dataset]
name = "mtg-ad"
root = "{cls.data}"
split = "validation"
subset = "all"
[paths]
cache_root = "{cls.cache_root}"
output_root = "{cls.out}"
[k_rule]
rate_per_minute = 6.0
[matrix]
selectors = ["U", "V", "A", "AV", "O"]
priors = ["P0", "P1"]
models = ["m1", "m2"]
[models.m1]
backend = "echo"
reply = "A person in a red shirt walks to the door and opens it slowly."
[models.m2]
backend = "echo"
[generation]
exemplars_path = "{cls.exemplars}"
[evaluation]
resamples = 200
text_metrics = ["bertscore"]
[ablations]
enabled = ["1", "2", "4", "8"]
[ablations.timeline]
selector = "AV"
model = "m1"
audio_native_model = "m2"
""")
        cls.exp_dir = cls.out / "synthetic"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_stage(self, stage, *extra):
        return run_experiment.main(["--config", str(self.config), "--stage", stage, *extra])

    def test_01_selector_only_never_builds_a_backend(self):
        with patch.object(backends, "build_backend", side_effect=AssertionError("selector-only must not build a model")):
            self.assertEqual(self.run_stage("selector-only"), 0)
        metrics = json.loads((self.exp_dir / "eval" / "selection_metrics.json").read_text())
        runs = metrics["runs"]
        for name in ("U", "V", "A", "AV", "O", "A_cue-both", "AV-suppress", "AV-event", "AV-local"):
            self.assertIn(name, runs)
        self.assertTrue(runs["O"]["oracle"])
        self.assertEqual(runs["O"]["summary"]["macro_f1"]["mean"], 1.0)
        self.assertEqual(runs["A"]["not_run"], {"missing_audio_features": 1})
        self.assertEqual(runs["AV"]["not_run"], {"missing_vad": 1})
        self.assertEqual(runs["U"]["summary"]["n_videos"], 3)
        self.assertEqual(metrics["common_video_ids"], ["silent", "talky"])  # A and AV cannot run without audio
        self.assertEqual(runs["U"]["summary_common"]["n_videos"], 2)
        av = {v["video_id"]: v for v in runs["AV"]["videos"]}
        self.assertEqual(av["talky"]["overlap_seconds"], 0.0)  # snapped to gaps with the full budget
        self.assertEqual(av["talky"]["zero_budget_picks"], 0)
        self.assertIsNotNone(av["silent"]["recall_on_silent_gt"])
        oracle_cmp = [c for c in metrics["comparisons"] if c["selector"] == "O"]
        self.assertTrue(all(c["oracle_upper_bound"] and "p_holm" not in c["wilcoxon"] for c in oracle_cmp))
        self.assertTrue((self.exp_dir / "tables" / "selection_summary.md").is_file())
        self.assertTrue((self.exp_dir / "plots" / "selection_timelines.svg").is_file())

    def test_02_selection_resume_and_retry(self):
        run = matrix.core_selections(cfgmod.load_config(self.config))[0]
        path = self.exp_dir / "selections" / f"{run.run_id}.jsonl"
        before = path.read_text().count("\n")
        self.assertEqual(self.run_stage("select"), 0)
        self.assertEqual(path.read_text().count("\n"), before)  # nothing recomputed

    def test_03_moments_generate_resume_evaluate_report(self):
        self.assertEqual(self.run_stage("moments"), 0)
        moments = [json.loads(l) for l in (self.exp_dir / "moments").glob("AV-*.jsonl").__next__().read_text().splitlines()]
        self.assertTrue(moments)
        for m in moments:
            self.assertEqual(len(m["frames"]), 4)
            self.assertGreater(m["word_limit"], 0)
        talky = [m for m in moments if m["video_id"] == "talky"]
        self.assertTrue(any(m["audio_timeline"] for m in talky))
        self.assertEqual(self.run_stage("generate"), 0)
        files = sorted((self.exp_dir / "generations").glob("ad-*.jsonl"))
        self.assertEqual(len(files), 20)
        sizes = {f: f.read_text().count("\n") for f in files}
        for f in files:
            for row in map(json.loads, f.read_text().splitlines()):
                self.assertLessEqual(row["words"], row["word_limit"])
                self.assertEqual(row["temperature"], 0.0)
        p1 = [f for f in files if "-P1-" in f.name][0]
        self.assertIn("Examples of professional audio descriptions", json.loads(p1.read_text().splitlines()[0])["prompt"])
        self.assertEqual(self.run_stage("generate"), 0)  # resume: nothing appended
        self.assertEqual({f: f.read_text().count("\n") for f in files}, sizes)
        self.assertEqual(self.run_stage("ablation5"), 0)
        tl = [f for f in (self.exp_dir / "generations").glob("ad-AV-P0-m1-tl1-*.jsonl")]
        self.assertTrue(tl and "Audio timeline" in tl[0].read_text())
        self.assertEqual(self.run_stage("evaluate-generation"), 0)
        evals = list((self.exp_dir / "eval").glob("generation_*.json"))
        self.assertGreaterEqual(len(evals), 20)
        one = json.loads(evals[0].read_text())
        self.assertIn(one["text_metrics"]["bertscore"]["status"].split(":")[0], ("ok", "unavailable", "no matched pairs"))
        self.assertEqual(self.run_stage("report"), 0)
        table = (self.exp_dir / "tables" / "generation_matrix.md").read_text()
        self.assertGreaterEqual(table.count("\n| "), 20)

    def test_04_p1_exemplar_leakage_is_refused(self):
        leaky = Path(self.tmp.name) / "leaky.json"
        leaky.write_text(json.dumps({"exemplars": [{"text": "x y z w", "video_id": "talky", "time": 0},
                                                   {"text": "x y z w", "video_id": "p2", "time": 0},
                                                   {"text": "x y z w", "video_id": "p3", "time": 0}]}))
        self.run_stage("moments")
        with self.assertRaises(ConfigError):
            run_experiment.main(["--config", str(self.config), "--stage", "generate", "--runs", "ad-U-P1-m1-*",
                                 "--set", f'generation.exemplars_path="{leaky}"',
                                 "--set", 'experiment.name="leak"'])

    def test_05_qa_with_canonical_dataset(self):
        canon = Path(self.tmp.name) / "qa.jsonl"
        canon.write_text("".join(json.dumps({
            "video_id": vid, "split": "test", "video_path": None, "duration": d,
            "qa": [{"question_id": f"{vid}_q", "question": "What happens?", "options": ["a", "b", "c"], "answer": 1}],
        }) + "\n" for vid, (d, *_rest) in VIDEOS.items()))
        cfg_path = Path(self.tmp.name) / "qa.toml"
        cfg_path.write_text(self.config.read_text().replace('[dataset]\nname = "mtg-ad"', f'[dataset]\nname = "canonical"\npath = "{canon}"')
                            .replace('split = "validation"', 'split = "test"').replace('name = "synthetic"', 'name = "qa"\ntask = "qa"\nallow_test = true')
                            .replace('reply = "A person in a red shirt walks to the door and opens it slowly."', 'reply = "B"')
                            .replace('selectors = ["U", "V", "A", "AV", "O"]', 'selectors = ["U", "AV"]'))
        self.assertEqual(run_experiment.main(["--config", str(cfg_path), "--stage", "select"]), 0)
        self.assertEqual(run_experiment.main(["--config", str(cfg_path), "--stage", "qa"]), 0)
        qa = json.loads((self.out / "qa" / "eval" / "qa_metrics.json").read_text())
        accs = {r["selector"] + str(r["timeline"]) + r["model"]: r for r in qa["runs"].values()}
        self.assertEqual(accs["UFalsem1"]["accuracy"]["mean"], 1.0)  # echo replies "B" = index 1
        self.assertEqual(accs["UFalsem2"]["n_unparsed"], 3)          # m2 echoes a sentence: unparsed, counted wrong
        self.assertTrue(qa["timeline_comparisons"])


    def test_06_tuning_is_dev_only_and_cross_validated(self):
        from src.experiments.data import load_records
        from src.experiments.tuning import tune

        cfg = cfgmod.load_config(self.config)
        result = tune(cfg, load_records(cfg), "AV", {"weights.nse": [-0.5, 0.5], "memory_blend": [0.5, 1.0]}, folds=2)
        self.assertEqual(len(result["all_points"]), 4)
        self.assertEqual(result["n_videos_scored"], 2)  # noaudio has no VAD, so AV cannot run there
        self.assertEqual(len(result["cv"]), 2)
        self.assertIsNotNone(result["cv_held_out_mean_f1"])
        test_cfg = cfgmod.apply_overrides(cfg, ['dataset.split="test"'])
        with self.assertRaises(ConfigError):
            tune(test_cfg, [], "AV")

    def test_07_cpu_cue_caching_script_resumes(self):
        import cache_audio_cues

        argv = ["--cache-dir", str(self.cache_root), "--no-asr", "--no-tagger", "--no-clap", "--text-encoder", "lexical",
                "--dataset-transcripts", str(self.data), "--output-name", "audio_cues.cpu.json"]
        self.assertEqual(cache_audio_cues.main(argv), 0)
        report = json.loads((self.cache_root / "audio_cues_report.json").read_text())
        self.assertEqual(sorted(report["computed"]), ["silent", "talky"])
        self.assertEqual(report["skipped"], {"noaudio": "no_audio_stream"})
        record = json.loads((self.cache_root / "talky" / "audio_cues.cpu.json").read_text())
        self.assertEqual(record["asr"]["source"], "dataset")
        self.assertTrue(any(c["kind"] == "deictic" for c in record["speech_cues"]))
        self.assertTrue(record["streams"]["acoustic"].startswith("ok"))
        self.assertEqual(cache_audio_cues.main(argv), 0)
        report = json.loads((self.cache_root / "audio_cues_report.json").read_text())
        self.assertEqual(sorted(report["current"]), ["silent", "talky"])  # resume: nothing recomputed

    def test_08_exemplars_and_precommit(self):
        import build_p1_exemplars
        import precommit_checks

        from src.generation.priors import MissingAsset

        splits = Path(self.tmp.name) / "splits.json"
        out = Path(self.tmp.name) / "ex.json"
        argv = ["--data-dir", str(self.data), "--splits", str(splits), "--output", str(out), "--max-words", "8"]
        splits.write_text(json.dumps({"dev": {"videos": [{"video_id": "talky"}]}, "test": {"video_ids": []}}))
        with self.assertRaises(MissingAsset):  # only 2 eligible videos left: refuse, never reuse or invent
            build_p1_exemplars.main(argv)
        splits.write_text(json.dumps({"dev": {"videos": []}, "test": {"video_ids": []}}))
        self.assertEqual(build_p1_exemplars.main(argv), 0)
        record = json.loads(out.read_text())
        self.assertEqual(len({e["video_id"] for e in record["exemplars"]}), 3)
        texts = {a for r in map(json.loads, (self.data / "validation.jsonl").read_text().splitlines())
                 for a in (g["text"] for g in r["gt_ad_segments"])}
        self.assertTrue(all(e["text"] in texts for e in record["exemplars"]))  # real annotation lines only
        report_path = Path(self.tmp.name) / "pre.json"
        self.assertEqual(precommit_checks.main(["--output", str(report_path)]), 0)
        statuses = {r["check"]: r["status"] for r in json.loads(report_path.read_text())}
        self.assertEqual(set(statuses.values()), {"blocked"})


class ConfigTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    BASE = '[experiment]\nname="x"\n[dataset]\nname="mtg-ad"\nroot="${ROOT:-/d}"\n[paths]\ncache_root="${CACHE}"\noutput_root="o"\n'

    def test_env_expansion_and_overrides(self):
        cfg = cfgmod.load_config(self.write(self.BASE), ["generation.max_new_tokens=96", 'matrix.priors=["P0"]'],
                                 env={"CACHE": "/scratch/c"})
        self.assertEqual(cfg["dataset"]["root"], "/d")
        self.assertEqual(cfg["paths"]["cache_root"], "/scratch/c")
        self.assertEqual(cfg["generation"]["max_new_tokens"], 96)
        self.assertEqual(cfg["matrix"]["priors"], ["P0"])
        with self.assertRaises(ConfigError):
            cfgmod.load_config(self.write(self.BASE), env={})

    def test_guards(self):
        env = {"CACHE": "c"}
        with self.assertRaises(ConfigError):
            cfgmod.load_config(self.write(self.BASE + '[models.api]\napi_key="sk-123"\n'), env=env)
        with self.assertRaises(ConfigError):
            cfgmod.load_config(self.write(self.BASE), ["generation.temperature=0.7"], env=env)
        with self.assertRaises(ConfigError):
            cfgmod.load_config(self.write(self.BASE), ['dataset.split="test"'], env=env)
        cfg = cfgmod.load_config(self.write(self.BASE), ['dataset.split="test"', "experiment.allow_test=true"], env=env)
        self.assertTrue(cfg["experiment"]["allow_test"])

    def test_matrix_shape_and_run_ids(self):
        cfg = cfgmod.load_config(self.write(self.BASE + '[matrix]\nmodels=["a","b"]\n[models.a]\nbackend="echo"\n[models.b]\nbackend="echo"\n'),
                                 env={"CACHE": "c"})
        runs = matrix.core_generation_runs(cfg)
        self.assertEqual(len(runs), 20)
        self.assertEqual(len({r.run_id for r in runs}), 20)
        cfg2 = cfgmod.apply_overrides(cfg, ["generation.max_new_tokens=99"])
        self.assertNotEqual(runs[0].run_id, matrix.core_generation_runs(cfg2)[0].run_id)
        a_cue = cfgmod.apply_overrides(cfg, ['matrix.a_selector="A_cue"'])
        self.assertIn("A_cue", [s.name for s in matrix.core_selections(a_cue)])
        for n in "12345678":
            matrix.ablation_selections(n, cfg)
        with self.assertRaises(ConfigError):
            matrix.ablation_selections("9", cfg)


class SpendCapTests(unittest.TestCase):
    def test_cap_is_enforced_before_sending(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            sent = []

            def transport(url, payload, headers, timeout):
                sent.append(payload)
                return {"choices": [{"message": {"content": "A dog."}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1000, "completion_tokens": 3}}

            spec = {"spend_cap_usd": 0.01, "ledger_path": f"{tmp}/ledger.json",
                    "prices": {"input_per_million": 1.0, "output_per_million": 4.0, "image_tokens": 1000}}
            backend = backends.OpenAIBackend("gpt-test", spec, transport=transport)
            results = backend.generate([backends.GenRequest(str(i), "describe", [], None, 64) for i in range(20)])
            ok = [r for r in results if r.error is None]
            capped = [r for r in results if r.error and r.error.startswith("spend_cap")]
            self.assertTrue(ok and capped)
            self.assertEqual(len(sent), len(ok))
            per_call = 1000 * 1.0 / 1e6 + 3 * 4.0 / 1e6
            spent = backends.SpendLedger(f"{tmp}/ledger.json", 0.01).spent()
            # At most one call's underestimate past the cap (before the ratio is learned).
            self.assertLessEqual(spent, 0.01 + per_call)
            self.assertLessEqual(len(ok), int(0.01 / per_call) + 1)

    def test_refuses_without_cap_prices_or_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "k"}, clear=False):
            os.environ.pop("PYRO_API_SPEND_CAP_USD", None)
            with self.assertRaises(ValueError):
                backends.OpenAIBackend("m", {"prices": {"input_per_million": 1, "output_per_million": 1}})
            with self.assertRaises(ValueError):
                backends.OpenAIBackend("m", {"spend_cap_usd": 1})
            with self.assertRaises(ValueError):
                backends.OpenAIBackend("m", {"spend_cap_usd": 1, "prices": {"input_per_million": 1, "output_per_million": 1},
                                             "temperature": 0.5})
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                backends.GeminiBackend("m", {"spend_cap_usd": 1, "prices": {"input_per_million": 1, "output_per_million": 1}})


class AdapterTests(unittest.TestCase):
    def test_mtg_validate_reports_counts_and_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "validation.jsonl").write_text(json.dumps({
                "video_id": "a", "split": "validation", "category": "Vlog", "duration_seconds": 10,
                "gt_ad_segments": [{"start_seconds": 12.5, "text": "x"}], "transcript_segments": [], "visual_info": None}) + "\n")
            (root / "test.jsonl").write_text("")
            (root / "videos").mkdir()
            (root / "videos" / "b.mp4").write_bytes(b"x")
            (root / "unavailable_videos.json").write_text(json.dumps(["a"]))
            (root / "transcripts").mkdir()
            (root / "transcripts" / "a.json").write_text("{}")
            report = MTGADAdapter(root).validate().to_dict()
        self.assertFalse(report["ok"])
        self.assertIn("validation: 1 rows, expected 332", report["errors"])
        self.assertEqual(report["counts"]["validation"]["ad_starts_beyond_duration"], 1)
        self.assertEqual(report["counts"]["listed_unavailable"], 1)
        self.assertTrue(any("not in any split" in w for w in report["warnings"]))
        self.assertTrue(any(u.startswith("transcripts/") for u in report["unvalidated"]))

    def test_canonical_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            path.write_text("\n".join([
                json.dumps({"video_id": "a", "split": "test", "qa": [{"question": "q", "options": ["x", "y"], "answer": 3}]}),
                json.dumps({"video_id": "b", "split": "test", "ad": [{"start": 5, "end": 2, "text": "t"}]}),
                json.dumps({"video_id": "c", "split": "test", "duration": 30.0}),
                json.dumps({"video_id": "c", "split": "test"}),
            ]))
            report = CanonicalAdapter(path).validate()
            self.assertEqual(len(report.errors), 3)
            with self.assertRaises(ValueError):
                CanonicalAdapter(path).load("test")


if __name__ == "__main__":
    unittest.main()
