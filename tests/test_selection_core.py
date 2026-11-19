"""Two-stage cue selector, hierarchical budgeting and the selector registry.

Synthetic cue tracks only; no models. Core selection logic is exercised for
real (never mocked): candidate generation, gap snapping, scoring, t_min, MMR,
running-memory novelty and budget allocation.
"""

import math
import unittest

import numpy as np

from src.generation.prompts import word_limit_for_gap
from src.selection import hierarchical as hi
from src.selection import two_stage as ts
from src.selection.cues import build_cue_bundle, make_grid
from src.selection.registry import SelectionContext, resolve_spec, run_selector
from src.utils.audio_features import available_gap
from src.utils.audio_selector import AudioSelector
from src.utils.selection import uniform_timestamps


def vad(speech, duration, margin=0.25, min_gap=1.0):
    """audio_features-like record with gaps inverted from speech (as the cache does)."""
    from src.utils.audio_features import invert_to_gaps

    gaps = invert_to_gaps(speech, duration, min_gap, margin)
    return {
        "config_hash": "vad1",
        "speech": [list(s) for s in speech],
        "gaps": [{"start": a, "end": b, "rms": 0.0, "onset": 0.0} for a, b in gaps],
    }


def visual(duration, peaks=(), shots=(), fps=2.0):
    times = [round(i / fps, 3) for i in range(int(duration * fps) + 1)]
    novelty = [0.01] * len(times)
    for p in peaks:
        novelty[min(len(times) - 1, int(round(p * fps)))] = 0.6
    return {"frame_times": times, "novelty": novelty, "shots": list(shots)}


def cues(events=(), speech_cues=(), streams=None):
    base = {"asr": "ok", "tagger": "ok", "clap": "missing", "acoustic": "missing",
            "semantic": "missing", "diarization": "missing"}
    return {
        "version": 1, "duration": 0.0, "streams": {**base, **(streams or {})},
        "events": [dict(e) for e in events], "speech_cues": [dict(c) for c in speech_cues],
        "asr": None, "tagger": None, "clap": None, "acoustic": None,
    }


TALKY_SPEECH = [(0.0, 20.0), (24.0, 50.0), (56.0, 90.0), (100.0, 118.0)]


class SnapTests(unittest.TestCase):
    def test_inside_gap_keeps_event_time(self):
        self.assertEqual(ts.snap(5.0, [(3.0, 8.0)], 3.0, 0.5), 5.0)

    def test_mid_speech_snaps_forward_within_lag(self):
        self.assertEqual(ts.snap(1.0, [(3.0, 8.0)], 3.0, 0.5), 3.0)

    def test_beyond_lag_is_infeasible(self):
        self.assertIsNone(ts.snap(-1.0 + 0.0, [(3.0, 8.0)], 3.0, 0.5))
        self.assertIsNone(ts.snap(10.0, [(3.0, 8.0)], 3.0, 0.5))

    def test_small_lead_allowed(self):
        self.assertEqual(ts.snap(8.3, [(3.0, 8.0)], 3.0, 0.5), 8.0)

    def test_feasible_intervals_leave_room_for_min_words(self):
        bundle = build_cue_bundle(60.0, vad([(0, 10), (12, 60)], 60.0))
        for s, e in ts.feasible_intervals(bundle, 3):
            self.assertGreaterEqual(word_limit_for_gap(available_gap(e, bundle.speech, 60.0)), 3)


class TwoStageTests(unittest.TestCase):
    def setUp(self):
        self.duration = 120.0
        self.vad = vad(TALKY_SPEECH, self.duration)

    def test_picks_are_feasible_separated_and_k_bounded(self):
        ctx = SelectionContext("v", self.duration, self.vad, cues(), visual(self.duration, peaks=(22, 52, 95)), None)
        result = run_selector("AV", ctx, 6)
        self.assertLessEqual(len(result.times), 6)
        self.assertEqual(result.times, sorted(result.times))
        for a, b in zip(result.times, result.times[1:]):
            self.assertGreaterEqual(b - a, 2.0 - 1e-9)
        speech = [tuple(s) for s in self.vad["speech"]]
        for t in result.times:
            self.assertGreaterEqual(word_limit_for_gap(available_gap(t, speech, self.duration)), 3)

    def test_visual_event_mid_gap_is_selected_at_event_time(self):
        ctx = SelectionContext("v", self.duration, self.vad, cues(), visual(self.duration, peaks=(95,)), None)
        result = run_selector("AV", ctx, 1)
        self.assertEqual(len(result.times), 1)
        self.assertAlmostEqual(result.times[0], 95.0, delta=0.5)
        self.assertEqual(result.details[0]["source"], "vn")

    def test_event_inside_speech_snaps_to_next_gap(self):
        # Peak at 48.5 s is in speech (24-50); the gap starts at 50.25 (margin).
        ctx = SelectionContext("v", self.duration, self.vad, cues(), visual(self.duration, peaks=(48.5,)), None)
        result = run_selector({"name": "AV1", "base": "AV", "options": {"fill": "none"}}, ctx, 1)
        self.assertEqual(len(result.times), 1)
        self.assertAlmostEqual(result.times[0], 50.25, places=2)
        self.assertGreater(result.details[0]["lag"], 0)

    def test_event_placement_keeps_time_and_penalizes_overlap(self):
        spec = {"name": "AVe", "base": "AV", "options": {"fill": "none", "placement": "event"}}
        ctx = SelectionContext("v", self.duration, self.vad, cues(), visual(self.duration, peaks=(48.5,)), None)
        result = run_selector(spec, ctx, 1)
        self.assertAlmostEqual(result.times[0], 48.5, places=2)
        self.assertGreater(result.details[0]["op"], 0.0)

    def test_speech_free_video_without_cues_spreads_picks(self):
        duration = 300.0
        ctx = SelectionContext("v", duration, vad([], duration), None, None, None)
        result = run_selector("A_cue", ctx, 10)
        self.assertEqual(len(result.times), 10)
        self.assertGreater(result.times[-1] - result.times[0], 0.6 * duration)
        self.assertNotEqual(result.times[:3], [0.0, 2.0, 4.0])
        self.assertEqual(result.status, "ok_degraded")

    def test_nse_sign_flips_preference(self):
        duration = 60.0
        events = [{"time": 20.0, "label": "Music", "superclass": "music", "strength": 0.9, "source": "tagger"}]
        ctx_args = ("v", duration, vad([], duration), cues(events), visual(duration, peaks=(40.0,)), None)
        trigger = run_selector({"name": "t", "base": "AV", "options": {"weights": {"nse": 2.0}, "fill": "none"}},
                               SelectionContext(*ctx_args), 1)
        suppress = run_selector({"name": "s", "base": "AV", "options": {"weights": {"nse": -2.0}, "fill": "none"}},
                                SelectionContext(*ctx_args), 1)
        self.assertAlmostEqual(trigger.times[0], 20.0, delta=1.0)
        self.assertAlmostEqual(suppress.times[0], 40.0, delta=1.0)

    def test_per_class_weights(self):
        duration = 60.0
        events = [
            {"time": 15.0, "label": "Music", "superclass": "music", "strength": 0.9, "source": "tagger"},
            {"time": 45.0, "label": "Applause", "superclass": "impulsive", "strength": 0.9, "source": "tagger"},
        ]
        opts = {"weights": {"vn": 0.0, "nse": 1.0}, "candidate_streams": ["nse"], "fill": "none",
                "nse_components": {"music": -1.0, "impulsive": 1.0}}
        ctx = SelectionContext("v", duration, vad([], duration), cues(events), None, None)
        result = run_selector({"name": "pc", "base": "A_cue", "options": opts}, ctx, 1)
        self.assertAlmostEqual(result.times[0], 45.0, delta=1.0)

    def test_av_disagreement_prefers_silent_visual_change(self):
        duration = 60.0
        events = [{"time": 20.0, "label": "Applause", "superclass": "impulsive", "strength": 1.0, "source": "tagger"}]
        opts = {"weights": {"vn": 0.0, "nse": 0.0, "sc": 0.0, "av_disagreement": 1.0}, "fill": "none",
                "candidate_streams": ["vn"]}
        # candidate_streams needs a nonzero weight to generate: use vn weight tiny.
        opts["weights"]["vn"] = 1e-6
        ctx = SelectionContext("v", duration, vad([], duration), cues(events), visual(duration, peaks=(20.0, 40.0)), None)
        result = run_selector({"name": "avd", "base": "AV", "options": {**opts, "novelty_mode": "local"}}, ctx, 1)
        self.assertAlmostEqual(result.times[0], 40.0, delta=0.5)

    def test_missing_vad_is_explicit(self):
        ctx = SelectionContext("v", 60.0, None, None, visual(60.0, peaks=(10,)), None)
        result = run_selector("AV", ctx, 3)
        self.assertEqual(result.status, "missing_vad")
        self.assertEqual(result.times, [])

    def test_missing_visual_is_explicit_for_av(self):
        ctx = SelectionContext("v", 60.0, vad([], 60.0), None, None, None, missing={"visual": "no_visual_features"})
        self.assertEqual(run_selector("AV", ctx, 3).status, "missing_visual_features")

    def test_no_valid_gap_gives_shortfall_not_crash(self):
        duration = 30.0
        ctx = SelectionContext("v", duration, vad([(0.0, 30.0)], duration), None, visual(duration, peaks=(10,)), None)
        result = run_selector("AV", ctx, 5)
        self.assertEqual(result.times, [])
        self.assertEqual(result.shortfall, 5)

    def test_memory_novelty_avoids_repeated_scene(self):
        duration = 60.0
        vis = visual(duration, peaks=(10.0, 30.0, 50.0))
        n = len(vis["frame_times"])
        emb = np.zeros((n, 3), dtype=np.float32)
        for i, t in enumerate(vis["frame_times"]):
            emb[i] = [1, 0, 0] if t < 40 else [0, 1, 0]   # scene A until 40 s, then scene B
        # Make 30 s slightly more novel locally than 50 s so local mode prefers it.
        vis["novelty"][int(30 * 2)] = 0.65
        base = {"fill": "none", "weights": {"nse": 0, "sc": 0, "gf": 0, "op": 0}, "candidate_streams": ["vn"]}
        local = run_selector({"name": "l", "base": "AV", "options": {**base, "novelty_mode": "local"}},
                             SelectionContext("v", duration, vad([], duration), None, vis, emb), 2)
        memory = run_selector({"name": "m", "base": "AV", "options": {**base, "novelty_mode": "memory"}},
                              SelectionContext("v", duration, vad([], duration), None, vis, emb), 2)
        self.assertIn(30.0, [round(t) for t in local.times])
        self.assertIn(50.0, [round(t) for t in memory.times])

    def test_mmr_requires_known_option_values(self):
        with self.assertRaises(ValueError):
            ts.merge_config({"diversity": "bogus"})
        with self.assertRaises(KeyError):
            ts.merge_config({"not_an_option": 1})


class HierarchyTests(unittest.TestCase):
    def test_allocate_sums_to_k_with_floor(self):
        alloc = hi.allocate(20, [100, 100, 100, 100], [0, 0, 0, 40], floor_fraction=0.5)
        self.assertEqual(sum(alloc), 20)
        self.assertTrue(all(a >= 2 for a in alloc))  # floor = 0.5 * 5
        self.assertEqual(max(alloc), alloc[3])

    def test_allocate_respects_capacity(self):
        alloc = hi.allocate(10, [50, 50], [100, 0], 0.0, capacities=[3, 20])
        self.assertEqual(alloc, [3, 7])
        self.assertEqual(sum(hi.allocate(10, [50, 50], [1, 1], 0.5, capacities=[2, 3])), 5)

    def test_allocate_zero_mass_falls_back_to_length(self):
        self.assertEqual(hi.allocate(9, [100, 200], [0, 0], 0.0), [3, 6])

    def test_segments_respect_bounds_and_prefer_long_silence(self):
        cuts = [(150.0, 1, "shot"), (170.0, 3, "long_silence"), (400.0, 1, "shot")]
        segments = hi.segment(600.0, cuts, {**hi.DEFAULT_HIERARCHY})
        self.assertEqual(segments[0], (0.0, 170.0))
        for s, e in segments[:-1]:
            self.assertLessEqual(e - s, 300.0)
            self.assertGreaterEqual(e - s, 60.0)
        self.assertEqual(segments[-1][1], 600.0)

    def test_hierarchical_selection_long_video(self):
        duration = 1800.0
        speech = [(t, t + 20.0) for t in np.arange(0, duration - 30, 35.0)]
        vis = visual(duration, peaks=tuple(range(25, 1800, 70)), shots=(300.0, 900.0))
        ctx = SelectionContext("long", duration, vad(speech, duration), None, vis, None)
        spec = {"name": "AVh", "base": "AV", "budget": "hierarchical"}
        result = run_selector(spec, ctx, 60)
        self.assertLessEqual(len(result.times), 60)
        self.assertGreater(len(result.times), 40)
        for a, b in zip(result.times, result.times[1:]):
            self.assertGreaterEqual(b - a, 2.0 - 1e-9)
        segs = result.diagnostics["segments"]
        self.assertEqual(sum(s["k"] for s in segs), 60)
        self.assertTrue(all(s["k"] >= 1 for s in segs))

    def test_capacity(self):
        self.assertEqual(hi.capacity([0, 1, 2, 3, 4], 2.0), 3)


class RegistryTests(unittest.TestCase):
    def test_u_and_o(self):
        ctx = SelectionContext("v", 100.0, ground_truth=[5, 50, 95])
        self.assertEqual(run_selector("U", ctx, 4).times, uniform_timestamps(100.0, 4))
        self.assertEqual(run_selector("O", ctx, 2).times, [5.0, 95.0])
        self.assertEqual(run_selector("O", SelectionContext("v", 100.0), 2).status, "missing_ground_truth")

    def test_team_a_is_unchanged(self):
        features = vad(TALKY_SPEECH, 120.0)
        ctx = SelectionContext("v", 120.0, features)
        expected = [c.time for c in AudioSelector().select(features, 5)]
        self.assertEqual(run_selector("A", ctx, 5).times, expected)

    def test_v_needs_visual(self):
        self.assertEqual(run_selector("V", SelectionContext("v", 60.0), 3).status, "missing_visual_features")
        ctx = SelectionContext("v", 60.0, visual_features=visual(60.0, peaks=(10, 30)))
        self.assertEqual(len(run_selector("V", ctx, 2).times), 2)

    def test_resolve_spec_merges_nested_options(self):
        spec = resolve_spec({"name": "x", "base": "A_cue", "options": {"weights": {"nse": -1.0}}})
        self.assertEqual(spec["kind"], "two_stage")
        self.assertEqual(spec["options"]["weights"]["vn"], 0.0)
        self.assertEqual(spec["options"]["weights"]["nse"], -1.0)
        with self.assertRaises(KeyError):
            resolve_spec("nope")

    def test_grid(self):
        self.assertEqual(make_grid(1.0, 0.5), [0.0, 0.5, 1.0])
        with self.assertRaises(ValueError):
            make_grid(0.0, 0.5)


if __name__ == "__main__":
    unittest.main()
