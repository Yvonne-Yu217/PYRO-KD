from contextlib import redirect_stdout
import io
from itertools import combinations
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.generation.prompts import MAX_WORDS, word_limit_for_gap
from src.utils.audio_features import (
    Gap, available_gap, gaps_of, invert_to_gaps, load_features, mean_in_window,
    merge_segments, speech_of, write_features,
)
from src.utils.audio_selector import AudioSelector
from src.utils.selection import select_top_k

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_moments  # noqa: E402

try:
    import librosa  # noqa: F401
    import numpy as np
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


def features_for(gaps):
    return {"speech": [], "gaps": [g.__dict__ for g in gaps]}


class TopKTests(unittest.TestCase):
    def test_min_separation_and_time_order(self):
        candidates = [(10.0, 0.9), (11.0, 0.95), (20.0, 0.5), (30.0, 0.7)]
        chosen = select_top_k(candidates, 3, min_separation=2.0)
        self.assertEqual([c[0] for c in chosen], [11.0, 20.0, 30.0])

    def test_shortfall_is_not_padded_and_ties_prefer_earlier(self):
        chosen = select_top_k([(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)], 3, min_separation=5.0)
        self.assertEqual(chosen, [(1.0, 1.0)])
        with self.assertRaises(ValueError):
            select_top_k([], 0)


class GapTests(unittest.TestCase):
    def test_invert_trims_speech_edges_only(self):
        gaps = invert_to_gaps([(3.0, 7.4), (13.5, 16.1)], 30.0, min_gap=1.0, margin=0.25)
        self.assertEqual(gaps, [(0.0, 2.75), (7.65, 13.25), (16.35, 30.0)])

    def test_drops_short_gaps_and_clips_speech(self):
        self.assertEqual(invert_to_gaps([(0.0, 5.0), (5.5, 20.0)], 20.0, margin=0.25), [])
        self.assertEqual(invert_to_gaps([], 12.0), [(0.0, 12.0)])
        self.assertEqual(invert_to_gaps([(-1.0, 2.0), (9.0, 40.0)], 10.0, margin=0.0, min_gap=1.0), [(2.0, 9.0)])

    def test_merge_segments(self):
        self.assertEqual(merge_segments([(5, 6), (1, 2), (2.05, 3)], max_gap=0.1), [(1, 3), (5, 6)])

    def test_available_gap(self):
        speech = [(3.0, 7.0), (13.0, 16.0)]
        self.assertEqual(available_gap(1.0, speech, 30.0), 2.0)
        self.assertEqual(available_gap(5.0, speech, 30.0), 0.0)
        self.assertEqual(available_gap(8.0, speech, 30.0), 5.0)
        self.assertEqual(available_gap(20.0, speech, 30.0), 10.0)
        self.assertEqual(available_gap(2.0, [], 30.0), 28.0)

    def test_mean_in_window(self):
        self.assertEqual(mean_in_window([0, 1, 2, 3], [1, 2, 3, 4], 1, 2), 2.5)
        self.assertEqual(mean_in_window([0, 1], [1, 2], 5, 6), 0.0)

    def test_word_limit_cap(self):
        self.assertEqual(word_limit_for_gap(0.0), 0)
        self.assertEqual(word_limit_for_gap(-1.0), 0)
        self.assertEqual(word_limit_for_gap(0.2), 0)
        self.assertEqual(word_limit_for_gap(0.4), 1)
        self.assertEqual(word_limit_for_gap(4.0), 12)
        self.assertEqual(word_limit_for_gap(60.0), MAX_WORDS)

    def test_features_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_features(tmp))
            write_features(tmp, {"speech": [[1.0, 2.0]], "gaps": [{"start": 2.0, "end": 5.0, "rms": 0.1, "onset": 0.2}]})
            features = load_features(tmp)
        self.assertEqual(speech_of(features), [(1.0, 2.0)])
        self.assertEqual(gaps_of(features), [Gap(2.0, 5.0, 0.1, 0.2)])


class AudioSelectorTests(unittest.TestCase):
    GAPS = [Gap(0.0, 3.0, 0.0, 0.0), Gap(10.0, 18.0, 0.0, 0.0), Gap(25.0, 28.0, 0.5, 0.0)]

    def test_prefers_long_quiet_gaps(self):
        selected = AudioSelector().select(features_for(self.GAPS), 2)
        self.assertEqual([c.time for c in selected], [10.0, 12.0])
        self.assertEqual(selected[1].gap_end, 18.0)

    def test_energy_penalty_demotes_loud_gap(self):
        scores = {c.time: c.score for c in AudioSelector(energy_weight=1.0).candidates(self.GAPS)}
        self.assertLess(scores[25.0], scores[0.0])

    def test_onset_sign_flips_ranking(self):
        gaps = [Gap(0.0, 5.0, 0.0, 1.0), Gap(10.0, 15.0, 0.0, 0.0)]
        penalty = AudioSelector(onset_weight=1.0).select(features_for(gaps), 1)
        bonus = AudioSelector(onset_weight=-1.0).select(features_for(gaps), 1)
        self.assertEqual((penalty[0].time, bonus[0].time), (10.0, 0.0))

    def test_shortfall_and_empty(self):
        self.assertEqual(len(AudioSelector().select(features_for(self.GAPS), 10)), 8)
        self.assertEqual(AudioSelector().select(features_for([]), 3), [])

    def test_long_silent_video_returns_k_greater_than_one(self):
        selected = AudioSelector().select(features_for([Gap(0, 30, 0, 0)]), 8)
        self.assertEqual([c.time for c in selected], list(range(0, 16, 2)))

    def test_short_gaps_carry_separation_and_do_not_create_false_shortfall(self):
        gaps = [Gap(0, 0.5, 1, 0), Gap(1, 2.4, 0, 0), Gap(3, 4.4, 0, 0)]
        selected = AudioSelector().select(features_for(gaps), 3)
        self.assertEqual([c.time for c in selected], [0, 2, 4])

    def test_subword_lattice_tail_is_not_feasible_capacity(self):
        gap = Gap(0, 4.2, 0, 0)
        selector = AudioSelector()
        candidates = selector.candidates([gap])
        selected = selector.select(features_for([gap]), 3)
        self.assertEqual([c.time for c in candidates], [0, 2])
        self.assertEqual(selected, candidates)
        self.assertTrue(all(word_limit_for_gap(c.gap_end - c.time) >= 1 for c in selected))
        self.assertEqual(word_limit_for_gap(gap.end - 4), 0)

    def test_k_is_reached_with_genuinely_speakable_timestamps(self):
        gaps = [Gap(0, 4.2, 0, 0), Gap(8, 10.4, 0, 0)]
        selector = AudioSelector()
        selected = selector.select(features_for(gaps), 4)
        self.assertEqual([c.time for c in selected], [0, 2, 8, 10])
        self.assertEqual(selected, selector.select(features_for(list(reversed(gaps))), 4))
        self.assertTrue(all(b.time - a.time >= 2 for a, b in zip(selected, selected[1:])))
        self.assertTrue(all(word_limit_for_gap(c.gap_end - c.time) > 0 for c in selected))

    def test_insufficient_non_speech_space_never_pads(self):
        gaps = [Gap(0, 0.5, 0, 0), Gap(2.5, 3, 0, 0)]
        self.assertEqual([c.time for c in AudioSelector().select(features_for(gaps), 5)], [0, 2.5])

    def test_equal_scores_use_earlier_times_and_ignore_input_order(self):
        gaps = [Gap(0.3, 5, 0, 0), Gap(8, 12, 0, 0)]
        selector = AudioSelector(duration_weight=0, energy_weight=0, onset_weight=0)
        selected = selector.select(features_for(gaps), 4)
        self.assertEqual(selected, selector.select(features_for(list(reversed(gaps))), 4))
        self.assertEqual(selected[0].time, 0.3)
        self.assertTrue(all(b.time - a.time >= 2 for a, b in zip(selected, selected[1:])))

    def test_speech_is_removed_even_from_crossing_cached_gaps(self):
        features = features_for([Gap(0, 10, 0, 0)])
        features["speech"] = [[1, 5], [6, 8]]
        selected = AudioSelector().select(features, 10)
        self.assertEqual([c.time for c in selected], [0, 5, 8])
        self.assertTrue(all(not any(s <= c.time < e for s, e in features["speech"]) for c in selected))

    def test_custom_separation_and_half_open_endpoints(self):
        selected = AudioSelector().select(features_for([Gap(0, 9, 0, 0)]), 10, min_separation=3)
        self.assertEqual([c.time for c in selected], [0, 3, 6])
        for separation in (0, -1, float("inf")):
            with self.subTest(separation=separation), self.assertRaises(ValueError):
                AudioSelector().select(features_for(self.GAPS), 2, min_separation=separation)

    def test_capacity_matches_exhaustive_small_interval_oracle(self):
        # Half-second endpoints ensure every earliest feasible schedule lies on
        # this grid. Enumerate all feasible subsets independently of the selector.
        grid = [i / 2 for i in range(8)]
        for mask in range(1 << len(grid)):
            available = [t for i, t in enumerate(grid) if mask & (1 << i)]
            gaps = [Gap(t, t + 0.5, (i % 3) / 2, 0) for i, t in enumerate(available)]
            capacity = max(
                (len(subset) for size in range(len(available) + 1)
                 for subset in combinations(available, size)
                 if all(b - a >= 2 for a, b in zip(subset, subset[1:]))), default=0,
            )
            for k in (1, 2, 4):
                with self.subTest(mask=mask, k=k):
                    selected = AudioSelector().select(features_for(gaps), k)
                    self.assertEqual(len(selected), min(k, capacity))
                    self.assertTrue(all(any(g.start <= c.time < g.end for g in gaps) for c in selected))
                    self.assertTrue(all(b.time - a.time >= 2 for a, b in zip(selected, selected[1:])))


class BuildMomentsTests(unittest.TestCase):
    def test_missing_audio_features_return_no_moments(self):
        self.assertEqual(build_moments.SELECTORS["A"](100.0, 5, None), [])

    def test_registered_selectors(self):
        self.assertEqual(sorted(build_moments.SELECTORS), ["A", "U"])
        self.assertEqual(build_moments.SELECTORS["U"](10.0, 4, None), [2.0, 4.0, 6.0, 8.0])

    def run_batch(self, selector, videos, features, no_audio=()):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            splits = root / "splits.json"
            splits.write_text(json.dumps({"k_rule": {"rate_per_minute": 3}}))
            output = root / "moments.jsonl"
            for video_id, _, _ in videos:
                cache = root / video_id
                cache.mkdir()
                (cache / "manifest.json").write_text("{}")
            manifests = {v: {"audio_path": None if v in no_audio else "audio.wav"} for v, _, _ in videos}
            stdout = io.StringIO()
            with (
                patch.object(sys, "argv", ["build_moments.py", "--selector", selector,
                                          "--splits", str(splits), "--cache-dir", str(root),
                                          "--output", str(output)]),
                patch.object(build_moments, "load_targets", return_value=videos),
                patch.object(build_moments, "load_manifest", side_effect=lambda p: manifests[p.name]),
                patch.object(build_moments, "load_features", side_effect=lambda p: features.get(p.name)),
                patch.object(build_moments, "frame_window", return_value=[]),
                redirect_stdout(stdout),
            ):
                build_moments.main()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            report = json.loads(output.with_suffix(".report.json").read_text())
            return rows, report, stdout.getvalue()

    def test_mixed_batch_continues_past_missing_audio_and_records_k(self):
        rows, report, stdout = self.run_batch(
            "A", [("missing", 60, []), ("good", 60, [])],
            {"good": features_for([Gap(0, 60, 0, 0)])},
        )
        self.assertEqual([r["video_id"] for r in rows], ["good"] * 3)
        self.assertEqual(report["videos"][0]["status"], "missing_audio_features")
        self.assertEqual(report["videos"][0]["k_shortfall"], 3)
        self.assertEqual(report["videos"][0]["num_selected"], 0)
        self.assertEqual(report["videos"][0]["num_written"], 0)
        self.assertEqual(report["videos"][0]["selection_shortfall"], 0)
        self.assertEqual(report["videos"][1]["k_shortfall"], 0)
        self.assertIn("zero A moments: missing", stdout)
        self.assertNotIn("fewer than K feasible moments", stdout)

    def test_missing_audio_and_genuine_capacity_shortfall_are_distinct(self):
        features = {"limited": features_for([Gap(0, 2.1, 0, 0)]),
                    "good": features_for([Gap(0, 60, 0, 0)])}
        rows, report, stdout = self.run_batch(
            "A", [("missing", 60, []), ("limited", 60, []), ("good", 60, [])], features,
        )
        self.assertEqual(report["videos"][0]["selection_shortfall"], 0)
        self.assertEqual(report["videos"][1]["selection_shortfall"], 2)
        self.assertEqual(report["videos"][1]["status"], "shortfall")
        summary = next(line for line in stdout.splitlines() if "fewer than K feasible moments" in line)
        self.assertIn("limited -2", summary)
        self.assertNotIn("missing", summary)
        self.assertEqual([r["video_id"] for r in rows], ["limited", "good", "good", "good"])

    def test_audio_selection_and_written_counts_agree_after_tail_filtering(self):
        for gap, expected in ((Gap(0, 4.2, 0, 0), 2), (Gap(0, 4.4, 0, 0), 3)):
            with self.subTest(gap=gap):
                features = features_for([gap])
                features["speech"] = [[gap.end, 60]]
                rows, report, _ = self.run_batch("A", [("v", 60, [])], {"v": features})
                entry = report["videos"][0]
                self.assertEqual(entry["num_selected"], expected)
                self.assertEqual(entry["num_written"], expected)
                self.assertEqual(entry["selection_shortfall"], 3 - expected)
                self.assertEqual(entry["skipped_moments"], [])
                self.assertTrue(all(r["word_limit"] > 0 for r in rows))

    def test_no_audio_stream_does_not_use_stale_features_or_stop_batch(self):
        features = {name: features_for([Gap(0, 60, 0, 0)]) for name in ("no_audio", "good")}
        rows, report, _ = self.run_batch(
            "A", [("no_audio", 60, []), ("good", 60, [])], features, no_audio=("no_audio",),
        )
        self.assertEqual([r["video_id"] for r in rows], ["good"] * 3)
        self.assertEqual(report["videos"][0]["k_shortfall"], 3)
        self.assertEqual(report["videos"][1]["k_shortfall"], 0)

    def test_inside_speech_and_subword_gaps_are_reported_not_generated(self):
        features = features_for([Gap(30, 60, 0, 0)])
        features["speech"] = [[14, 29.8], [30.2, 40]]
        rows, report, stdout = self.run_batch("U", [("v", 60, [])], {"v": features})
        # Uniform timestamps are 15, 30, 45. First two have zero-word budgets.
        self.assertEqual([r["time"] for r in rows], [45])
        self.assertEqual(rows[0]["word_limit"], 30)
        entry = report["videos"][0]
        self.assertEqual(entry["num_selected"], 3)
        self.assertEqual(entry["selection_shortfall"], 0)
        self.assertEqual(entry["k_shortfall"], 2)
        self.assertEqual([r["time"] for r in entry["skipped_moments"]], [15, 30])
        self.assertTrue(all(r["word_limit"] == 0 for r in entry["skipped_moments"]))
        self.assertIn("not safely speakable", stdout)

    def test_unknown_gap_keeps_explicit_fallback_interface(self):
        rows, report, _ = self.run_batch("U", [("v", 60, [])], {})
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["word_limit"] is None for r in rows))
        self.assertEqual(report["videos"][0]["k_shortfall"], 0)


@unittest.skipUnless(HAS_LIBROSA, "librosa not installed")
class LibrosaFeatureTests(unittest.TestCase):
    def test_gap_features_see_noise_burst(self):
        from src.utils.audio_features import gap_features

        sr = 16000
        waveform = np.zeros(10 * sr, dtype=np.float32)
        rng = np.random.default_rng(0)
        waveform[4 * sr: 5 * sr] = (0.5 * rng.standard_normal(sr)).astype(np.float32)
        quiet, loud = gap_features(waveform, sr, [(7.0, 9.0), (3.0, 6.0)])
        self.assertEqual(quiet.rms, 0.0)
        self.assertEqual(quiet.onset, 0.0)
        self.assertGreater(loud.rms, 0.05)
        self.assertGreater(loud.onset, 0.0)
        # The burst's energy sits at 4-5 s, so a window around it beats one next to it.
        near, away = gap_features(waveform, sr, [(4.0, 5.0), (5.5, 6.5)])
        self.assertGreater(near.rms, away.rms)


if __name__ == "__main__":
    unittest.main()
