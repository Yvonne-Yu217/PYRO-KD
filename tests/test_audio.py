import json
from pathlib import Path
import sys
import tempfile
import unittest

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
        self.assertEqual(word_limit_for_gap(0.0), 3)
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
        self.assertEqual([c.time for c in selected], [0.0, 10.0])
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
        self.assertEqual(len(AudioSelector().select(features_for(self.GAPS), 10)), 3)
        self.assertEqual(AudioSelector().select(features_for([]), 3), [])


class BuildMomentsTests(unittest.TestCase):
    def test_audio_selector_requires_features(self):
        with self.assertRaises(SystemExit):
            build_moments.SELECTORS["A"](100.0, 5, None)

    def test_registered_selectors(self):
        self.assertEqual(sorted(build_moments.SELECTORS), ["A", "U"])
        self.assertEqual(build_moments.SELECTORS["U"](10.0, 4, None), [2.0, 4.0, 6.0, 8.0])


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
