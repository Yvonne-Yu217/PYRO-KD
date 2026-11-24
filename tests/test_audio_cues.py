"""Audio cue stack: NSE salience, speech cues, VAD-gated ASR plumbing, tagging
windows, the timeline, and audio_cues.json end to end with injected models.

Models are injected as plain functions (no Whisper/AST/CLAP downloads); the
logic around them (gating, offsets, onsets, change, hashing) runs for real.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cache_fixtures import make_media_cache
from src.audio import asr, cues_io, salience, speech_cues, tagging, timeline
from src.audio.taxonomy import superclass_of
from src.datasets.adapters import ADEvent, VideoRecord
from src.experiments import io, matrix, selection_stage
from src.utils import audio_features as af


class TaxonomyTests(unittest.TestCase):
    def test_known_and_keyword_and_override(self):
        self.assertEqual(superclass_of("Music"), "music")
        self.assertEqual(superclass_of("Applause"), "impulsive")
        self.assertEqual(superclass_of("Male speech, man speaking"), "speech")
        self.assertEqual(superclass_of("Electric guitar"), "music")
        self.assertEqual(superclass_of("Zither-ish thing"), "other")
        self.assertEqual(superclass_of("Music", {"Music": "other"}), "other")


class SalienceTests(unittest.TestCase):
    def test_class_onset_detects_rise_and_skips_speech(self):
        times = [float(i) for i in range(20)]
        music = [0.05] * 10 + [0.8] * 10
        speech = [0.0] * 5 + [0.9] * 15
        events = salience.class_onsets(times, np.array([music, speech]).T, ["Music", "Speech"])
        self.assertEqual([(e.time, e.label, e.superclass) for e in events], [(10.0, "Music", "music")])

    def test_onset_needs_min_probability(self):
        times = [float(i) for i in range(10)]
        low = [0.0] * 5 + [0.25] * 5
        self.assertEqual(salience.class_onsets(times, np.array([low]).T, ["Music"]), [])

    def test_bad_shape_raises(self):
        with self.assertRaises(ValueError):
            salience.class_onsets([0.0, 1.0], np.zeros((3, 1)), ["Music"])

    def test_embedding_change_peaks_at_transition(self):
        times = [float(i) for i in range(10)]
        emb = np.array([[1.0, 0.0]] * 5 + [[0.0, 1.0]] * 5)
        change = salience.embedding_change(times, emb, window_seconds=3.0)
        self.assertEqual(int(np.argmax(change)), 5)
        self.assertAlmostEqual(change[0], 0.0)

    def test_loudness_and_jumps(self):
        sr = 16000
        wave = np.concatenate([np.zeros(sr), 0.5 * np.ones(sr)]).astype(np.float32)
        times, levels = salience.loudness_db(wave, sr, 0.5)
        self.assertEqual(len(times), 4)
        self.assertEqual(levels[0], -100.0)
        jumps = salience.loudness_jumps(times, levels, 1.0)
        self.assertEqual(int(np.argmax(jumps)), 2)

    def test_resample_and_events_to_track_and_normalize(self):
        grid = [0.0, 0.5, 1.0, 1.5]
        self.assertEqual(salience.resample_to_grid([0.0, 0.1, 1.0], [1.0, 3.0, 2.0], grid, "max"), [3.0, 3.0, 2.0, 2.0])
        track = salience.events_to_track([{"time": 1.0, "strength": 0.8, "superclass": "music"}], grid, 1.0)
        self.assertEqual(track, [0.0, 0.4, 0.8, 0.4])
        self.assertEqual(salience.events_to_track([{"time": 1.0, "strength": 1.0, "superclass": "music"}], grid, 1.0,
                                                  superclass="impulsive"), [0.0] * 4)
        self.assertEqual(salience.normalize([0.0, 2.0, 4.0], 100.0), [0.0, 0.5, 1.0])
        self.assertEqual(salience.normalize([0.0, 0.0]), [0.0, 0.0])


class SpeechCueTests(unittest.TestCase):
    def test_deictic_with_word_timestamps(self):
        seg = {"start": 10.0, "end": 13.0, "text": "Okay look at this one",
               "words": [{"start": 10.0, "end": 10.3, "word": "Okay"}, {"start": 10.5, "end": 10.8, "word": "look"},
                         {"start": 10.9, "end": 11.0, "word": "at"}, {"start": 11.1, "end": 11.4, "word": "this"},
                         {"start": 11.5, "end": 11.8, "word": "one"}]}
        found = {(c.time, c.text) for c in speech_cues.deictic_cues([seg])}
        self.assertIn((10.5, "look at this"), found)
        self.assertIn((11.1, "this one"), found)

    def test_deictic_interpolated_without_words(self):
        cues = speech_cues.deictic_cues([{"start": 0.0, "end": 10.0, "text": "and over there is the lake"}])
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0].timing, "interpolated")
        self.assertTrue(0.0 < cues[0].time < 10.0)

    def test_malformed_segments_are_skipped(self):
        self.assertEqual(speech_cues.deictic_cues([{"start": "x", "end": 1, "text": "look at this"}, {"text": "hi"}]), [])

    def test_semantic_shift_with_lexical_encoder(self):
        segs = [{"start": float(i), "end": float(i) + 1, "text": t} for i, t in enumerate(
            ["pasta water boil salt", "pasta salt boil water", "boil the pasta water",
             "football goal match score", "match score goal football", "goal football score match"])]
        cues = speech_cues.semantic_shift_cues(segs, speech_cues.lexical_encoder(), window=2, threshold=0.3)
        self.assertEqual([c.time for c in cues], [3.0])

    def test_speaker_change_and_long_silence(self):
        turns = [{"start": 0, "end": 2, "speaker": "A"}, {"start": 2, "end": 4, "speaker": "A"},
                 {"start": 4, "end": 6, "speaker": "B"}]
        self.assertEqual([c.time for c in speech_cues.speaker_change_cues(turns)], [4.0])
        cues = speech_cues.long_silence_cues([(0.0, 5.0), (10.0, 30.0)], 8.0)
        self.assertEqual([(c.time, c.strength) for c in cues], [(10.0, 1.0)])
        self.assertEqual(speech_cues.long_silence_track([(10.0, 30.0)], [5.0, 15.0, 31.0], 8.0), [0.0, 1.0, 0.0])


class ASRTests(unittest.TestCase):
    def test_regions_pad_merge_split(self):
        regions = asr.vad_regions([(1.0, 2.0), (2.5, 3.0), (10.0, 80.0)], 100.0, pad=0.2, merge_gap=1.0, max_region=30.0)
        self.assertEqual(regions[0], (0.8, 3.2))
        self.assertEqual(len(regions), 4)  # 9.8-80.2 split in 3
        self.assertTrue(all(e - s <= 30.0 + 1e-9 for s, e in regions))

    def test_transcribe_regions_offsets_and_clips(self):
        sr = 16000
        wave = np.zeros(sr * 20, dtype=np.float32)
        calls = []

        def fake(chunk):
            calls.append(len(chunk))
            return [{"start": 0.1, "end": 99.0, "text": " hi there ",
                     "words": [{"start": 0.1, "end": 0.4, "word": "hi"}, {"start": 0.5, "end": 99.0, "word": "there"}]},
                    {"start": 0.0, "end": 1.0, "text": "  "}]

        segs = asr.transcribe_regions(wave, sr, [(5.0, 7.0), (12.0, 13.0)], fake)
        self.assertEqual(calls, [2 * sr, sr])
        self.assertEqual(segs[0]["start"], 5.1)
        self.assertEqual(segs[0]["end"], 7.0)  # clipped to its region
        self.assertEqual(segs[0]["words"][1]["end"], 7.0)
        self.assertEqual(len(segs), 2)  # blank segments dropped
        self.assertIsNone(asr.validate_segments(segs, 20.0))
        self.assertIsNotNone(asr.validate_segments([{"start": 5, "end": 30}], 20.0))

    def test_whisper_tags(self):
        events = asr.whisper_tag_events([{"start": 3.0, "end": 5.0, "text": "[MUSIC] (applause) ♪ [silence]"}])
        self.assertEqual([(e.label, e.superclass) for e in events],
                         [("Applause", "impulsive"), ("Music", "music"), ("Music", "music")])


class TaggingTests(unittest.TestCase):
    def test_windows_are_centred_and_padded(self):
        sr = 100
        wave = np.arange(300, dtype=np.float32)
        centres = tagging.window_centres(3.0, 1.0)
        self.assertEqual(centres, [0.0, 1.0, 2.0, 3.0])
        windows = tagging.sliding_windows(wave, sr, centres, 2.0)
        self.assertEqual(windows[0][:100].sum(), 0.0)  # left padding before t=0
        self.assertEqual(windows[1][0], 0.0)
        self.assertEqual(windows[1][-1], 199.0)

    def test_run_windowed_and_keep_labels(self):
        def model(windows, sr):
            return np.array([[w.mean() > 100, 0.01, 0.5] for w in windows], dtype=np.float32)

        times, out = tagging.run_windowed(np.arange(400, dtype=np.float32), 100, model, 2.0, 1.0, batch_size=2)
        self.assertEqual(out.shape, (5, 3))
        labels, kept = tagging.keep_labels(["a", "b", "c"], out, top=5, min_posterior=0.05)
        self.assertEqual(labels, ["a", "c"])
        with self.assertRaises(ValueError):
            tagging.run_windowed(np.zeros(200, dtype=np.float32), 100, lambda w, sr: np.zeros((1, 2)), 2.0, 1.0, 4)


class TimelineTests(unittest.TestCase):
    def test_format(self):
        text = timeline.format_timeline(
            [{"time": 42, "label": "Music", "superclass": "music", "strength": 0.9},
             {"time": 55, "label": "Applause", "superclass": "impulsive", "strength": 0.8}],
            [{"start": 63, "text": 'we "made" it to the top of the mountain finally today'}],
        )
        self.assertEqual(text, "[0:42 music starts] [0:55 applause] [1:03 speech: \"we 'made' it to the top of the...\"]")

    def test_window_and_budget(self):
        events = [{"time": float(t), "label": "Bang", "superclass": "impulsive", "strength": t / 100} for t in range(100)]
        items = timeline.timeline_items(events, [], 10, 50, {"max_items": 3})
        self.assertEqual([t for t, _, _ in items], [48.0, 49.0, 50.0])
        self.assertEqual(timeline.clock(3725), "1:02:05")
        self.assertEqual(timeline.format_timeline([], []), "")


class CueFileTests(unittest.TestCase):
    def setUp(self):
        import soundfile as sf

        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        sr = 16000
        t = np.arange(sr * 12) / sr
        wave = np.where((t > 6) & (t < 9), 0.3 * np.sin(2 * np.pi * 440 * t), 0.0).astype(np.float32)
        self.wav = self.dir / "audio_16k.wav"
        sf.write(self.wav, wave, sr)
        self.vad = {"config_hash": "vad-1", "speech": [[1.0, 3.0]],
                    "gaps": [{"start": 3.25, "end": 12.0, "rms": 0.0, "onset": 0.0}]}

    def tearDown(self):
        self.tmp.cleanup()

    def backends(self):
        def transcriber(chunk):
            return [{"start": 0.0, "end": 1.0, "text": "look at this",
                     "words": [{"start": 0.1, "end": 0.3, "word": "look"}, {"start": 0.35, "end": 0.5, "word": "at"},
                               {"start": 0.55, "end": 0.8, "word": "this"}]}]

        def tagger(windows, sr):
            return np.array([[1.0 if np.abs(w).max() > 0.1 else 0.0, 0.0] for w in windows], dtype=np.float32)

        def clap(windows, sr):
            return np.array([[1.0, 0.0] if np.abs(w).max() > 0.1 else [0.0, 1.0] for w in windows], dtype=np.float32)

        return cues_io.CueBackends(
            transcriber=transcriber, transcriber_name="fake-whisper", tagger=tagger, tagger_labels=["Music", "Speech"],
            tagger_name="fake-ast", clap=clap, clap_name="fake-clap",
            text_encoder=speech_cues.lexical_encoder(), text_encoder_name="lexical",
        )

    def test_compute_write_read_and_staleness(self):
        backends = self.backends()
        record = cues_io.compute_audio_cues(self.wav, self.vad, backends, {"tagger": {"window_seconds": 1.0}})
        self.assertIsNone(cues_io.malformed_cues(record))
        self.assertEqual(record["asr"]["source"], "whisper-vad-gated")
        self.assertEqual(record["asr"]["segments"][0]["start"], 0.8)  # region padded to 0.8
        self.assertEqual(record["asr"]["segments"][0]["words"][0]["start"], 0.9)  # 0.8 + 0.1
        music = [e for e in record["events"] if e["label"] == "Music"]
        self.assertTrue(music and 5.0 <= music[0]["time"] <= 7.0)
        self.assertTrue(any(c["kind"] == "deictic" for c in record["speech_cues"]))
        self.assertTrue(any(c["kind"] == "long_silence" for c in record["speech_cues"]))
        self.assertEqual(record["streams"]["diarization"].split(":")[0], "missing")
        cues_io.write_cues(self.dir, record)
        loaded, reason = cues_io.load_cues_for_selection(self.dir, self.wav, self.vad)
        self.assertIsNone(reason)
        current, reason = cues_io.load_current_cues(self.dir, self.wav, self.vad, backends, {"tagger": {"window_seconds": 1.0}})
        self.assertIsNone(reason)
        # A different config or VAD result makes the file stale.
        self.assertEqual(cues_io.load_current_cues(self.dir, self.wav, self.vad, backends)[1], cues_io.STALE_CUES)
        self.assertEqual(cues_io.load_cues_for_selection(self.dir, self.wav, {**self.vad, "config_hash": "other"})[1],
                         cues_io.STALE_CUES)

    def test_missing_backends_are_reported_not_silent(self):
        record = cues_io.compute_audio_cues(self.wav, self.vad, cues_io.CueBackends())
        self.assertTrue(record["streams"]["asr"].startswith("missing"))
        self.assertTrue(record["streams"]["tagger"].startswith("missing"))
        self.assertIsNone(record["tagger"])
        self.assertEqual(record["events"], [])

    def test_dataset_transcript_fallback(self):
        backends = cues_io.CueBackends(dataset_transcript=[{"start": 1.0, "end": 3.0, "text": "watch this"}])
        record = cues_io.compute_audio_cues(self.wav, self.vad, backends)
        self.assertEqual(record["asr"]["source"], "dataset")
        self.assertEqual(record["speech_cues"][0]["timing"], "interpolated")

    def test_malformed_and_unreadable(self):
        (self.dir / cues_io.CUES_NAME).write_text("{not json")
        self.assertEqual(cues_io.read_cues(self.dir)[1], cues_io.UNREADABLE_CUES)
        (self.dir / cues_io.CUES_NAME).write_text(json.dumps({"version": 1, "duration": 1.0, "streams": {},
                                                              "events": [{"time": "x", "strength": 1}], "speech_cues": []}))
        self.assertEqual(cues_io.read_cues(self.dir)[1], cues_io.MALFORMED_CUES)
        self.assertEqual(cues_io.read_cues(self.dir / "nope")[1], cues_io.NO_CUES)

    def test_malformed_tagger_labels_are_rejected_even_with_current_fingerprints(self):
        record = {
            "version": cues_io.CUES_VERSION, "duration": 12.0, "streams": {}, "events": [],
            "speech_cues": [], "audio_sha256": af.wav_digest(self.wav), "vad_hash": self.vad["config_hash"],
        }
        for labels in (7, True, 0, None, "X", {"Music": 0}, [7]):
            with self.subTest(labels=labels):
                record["tagger"] = {"times": [1.0], "labels": labels, "posteriors": [[0.2] if labels else []]}
                cues_io.write_cues(self.dir, record)
                self.assertEqual(cues_io.load_cues_for_selection(self.dir, self.wav, self.vad),
                                 (None, cues_io.MALFORMED_CUES))


class CueEvaluationTests(unittest.TestCase):
    def test_malformed_tagger_labels_do_not_abort_selector_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root, exp = root / "cache", root / "experiment"
            cfg = {"paths": {"cache_root": str(cache_root)}, "matrix": {"selectors": ["U"]},
                   "evaluation": {"resamples": 10}}
            run, = matrix.core_selections(cfg)
            records = []
            rows = []
            for video_id, labels in (("broken", 7), ("healthy", ["Music"])):
                media = make_media_cache(cache_root, video_id)
                cache, wav = media["cache"], media["wav"]
                vad = {"speech": [], "gaps": [], "config_hash": af.features_config_hash(wav)}
                af.write_features(cache, vad)
                cues_io.write_cues(cache, {
                    "version": cues_io.CUES_VERSION, "duration": 12.0, "streams": {}, "events": [],
                    "speech_cues": [], "audio_sha256": af.wav_digest(wav), "vad_hash": vad["config_hash"],
                    "tagger": {"times": [1.0], "labels": labels, "posteriors": [[0.2]]},
                })
                records.append(VideoRecord("mtg-ad", video_id, "validation", 12.0, media["source"],
                                           [ADEvent(6.0, None, "A person waves.")]))
                rows.append({"video_id": video_id, "duration": 12.0, "status": "ok", "k": 1, "times": [6.0]})
            io.append_jsonl(selection_stage.selections_path(exp, run), rows)

            result = selection_stage.evaluate_selections(cfg, [run], records, exp)

            broken, healthy = result["runs"]["U"]["videos"]
            self.assertEqual((broken["video_id"], healthy["video_id"]), ("broken", "healthy"))
            self.assertEqual((broken["f1"], healthy["f1"]), (1.0, 1.0))
            self.assertNotIn("n_silent_gt", broken)
            self.assertIn("n_silent_gt", healthy)


if __name__ == "__main__":
    unittest.main()
