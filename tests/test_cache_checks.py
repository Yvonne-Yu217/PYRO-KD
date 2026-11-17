import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.utils import cache_checks as cc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_fixtures as fx  # noqa: E402


class LoadAudioFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def good_cache(self, video_id="v", **kwargs):
        paths = fx.make_media_cache(self.root, video_id, **kwargs)
        if paths["manifest"]["audio_path"]:
            fx.write_audio_features(paths["cache"], paths["manifest"], speech=[(1.0, 2.0)], gaps=[(2.25, 9.0)])
        return paths

    def load(self, paths):
        with fx.patched_audio_helpers():
            return cc.load_audio_features(paths["cache"])

    def test_valid_cache_loads(self):
        features, reason = self.load(self.good_cache())
        self.assertIsNone(reason)
        self.assertEqual(features["speech"], [[1.0, 2.0]])

    def test_missing_and_unreadable_manifest(self):
        self.assertEqual(cc.load_audio_features(self.root / "nowhere"), (None, cc.NO_MANIFEST))
        paths = self.good_cache()
        (paths["cache"] / "manifest.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(self.load(paths), (None, cc.UNREADABLE_MANIFEST))
        (paths["cache"] / "manifest.json").write_text("[1, 2]", encoding="utf-8")
        self.assertEqual(self.load(paths), (None, cc.UNREADABLE_MANIFEST))

    def test_incomplete_media_cache(self):
        paths = self.good_cache()
        paths["frame"].unlink()
        self.assertEqual(self.load(paths), (None, cc.MEDIA_INCOMPLETE))

    def test_complete_but_stale_media_cache(self):
        # A rebuild for a replaced source failed and left the old cache complete.
        paths = self.good_cache()
        paths["source"].write_bytes(b"a different, longer replacement video")
        self.assertEqual(self.load(paths), (None, cc.MEDIA_STALE))

    def test_no_audio_stream_and_no_features(self):
        self.assertEqual(self.load(self.good_cache("silent", audio=False)), (None, cc.NO_AUDIO_STREAM))
        paths = fx.make_media_cache(self.root, "nofeatures")
        self.assertEqual(self.load(paths), (None, cc.NO_AUDIO_FEATURES))

    def test_corrupt_or_malformed_features_never_raise(self):
        paths = self.good_cache()
        features_path = paths["cache"] / "audio_features.json"
        for text, expected in (
            ("{broken", cc.UNREADABLE_AUDIO_FEATURES),
            ("\\xff", cc.UNREADABLE_AUDIO_FEATURES),
            ("[]", cc.MALFORMED_AUDIO_FEATURES),
            (json.dumps({"speech": [], "config_hash": "x"}), cc.MALFORMED_AUDIO_FEATURES),
        ):
            with self.subTest(text=text):
                features_path.write_text(text, encoding="utf-8")
                self.assertEqual(self.load(paths), (None, expected))

    def test_replaced_wav_makes_features_stale(self):
        paths = self.good_cache()
        paths["wav"].write_bytes(b"RIFF....a completely different, longer recording")
        self.assertEqual(self.load(paths), (None, cc.STALE_AUDIO_FEATURES))

    def test_changed_manifest_hash_makes_features_stale(self):
        paths = self.good_cache()
        manifest = {**paths["manifest"], "config_hash": "m2"}
        (paths["cache"] / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(self.load(paths), (None, cc.STALE_AUDIO_FEATURES))

    def test_missing_wav_is_reported(self):
        paths = self.good_cache()
        # cache_is_complete already requires the WAV, so this checks the order of the reasons.
        paths["wav"].unlink()
        self.assertEqual(self.load(paths), (None, cc.MEDIA_INCOMPLETE))

    def test_features_cannot_be_verified_without_the_fingerprint_helpers(self):
        paths = self.good_cache()
        with (
            patch.object(cc, "features_config_hash", None),
            patch.object(cc, "features_are_current", None),
        ):
            self.assertEqual(cc.load_audio_features(paths["cache"]), (None, cc.UNVERIFIABLE_AUDIO_FEATURES))

    def test_features_without_a_stored_hash_are_stale(self):
        paths = self.good_cache()
        features_path = paths["cache"] / "audio_features.json"
        features_path.write_text(json.dumps({"speech": [], "gaps": []}), encoding="utf-8")
        self.assertEqual(self.load(paths), (None, cc.STALE_AUDIO_FEATURES))


GOOD_GAP = {"start": 2.0, "end": 9.0, "rms": 0.1, "onset": 0.2}
MALFORMED_PAYLOADS = {
    "speech pair with one value": {"raw_speech": [[1.0]]},
    "speech pair with three values": {"raw_speech": [[1.0, 2.0, 3.0]]},
    "speech entry not a list": {"raw_speech": [5]},
    "speech with a string": {"raw_speech": [["a", 2.0]]},
    "speech with a boolean": {"raw_speech": [[True, 2.0]]},
    "speech reversed": {"raw_speech": [[3.0, 1.0]]},
    "speech negative": {"raw_speech": [[-1.0, 2.0]]},
    "speech not finite": {"raw_speech": [[1.0, float("inf")]]},
    "speech nan": {"raw_speech": [[float("nan"), 2.0]]},
    "empty gap record": {"raw_gaps": [{}]},
    "gap missing fields": {"raw_gaps": [{"start": 1.0, "end": 2.0}]},
    "gap with an extra field": {"raw_gaps": [{**GOOD_GAP, "extra": 1}]},
    "gap not a record": {"raw_gaps": [[1.0, 2.0]]},
    "gap null": {"raw_gaps": [None]},
    "gap reversed": {"raw_gaps": [{**GOOD_GAP, "start": 9.0, "end": 2.0}]},
    "gap with a string value": {"raw_gaps": [{**GOOD_GAP, "rms": "loud"}]},
    "gap not finite": {"raw_gaps": [{**GOOD_GAP, "onset": float("nan")}]},
    "gap negative start": {"raw_gaps": [{**GOOD_GAP, "start": -1.0}]},
}


class MalformedAudioPayloadTests(unittest.TestCase):
    """A correctly stamped cache can still carry a payload that would crash a consumer."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def cache_with(self, **payload):
        paths = fx.make_media_cache(self.root, "v")
        fx.write_audio_features(paths["cache"], paths["manifest"], **payload)
        return paths

    def test_the_helper_describes_each_defect(self):
        self.assertIsNone(cc.malformed_audio_features(
            {"speech": [[1.0, 2.0]], "gaps": [GOOD_GAP]}))
        self.assertIn("pair", cc.malformed_audio_features({"speech": [[1.0]], "gaps": []}))
        self.assertIn("exactly", cc.malformed_audio_features({"speech": [], "gaps": [{}]}))
        self.assertIsNotNone(cc.malformed_audio_features([]))
        self.assertIsNotNone(cc.malformed_audio_features({"speech": {}, "gaps": []}))

    def test_every_malformed_payload_is_unavailable_not_an_exception(self):
        for name, payload in MALFORMED_PAYLOADS.items():
            with self.subTest(payload=name):
                paths = self.cache_with(**payload)
                with fx.patched_audio_helpers():
                    self.assertEqual(cc.load_audio_features(paths["cache"]), (None, cc.MALFORMED_AUDIO_FEATURES))

    def test_structure_is_checked_before_the_fingerprint_so_it_is_never_masked(self):
        # Without the fingerprint helpers a malformed payload must still be reported as malformed,
        # not as merely unverifiable.
        paths = self.cache_with(raw_speech=[[1.0]])
        with (
            patch.object(cc, "features_config_hash", None),
            patch.object(cc, "features_are_current", None),
        ):
            self.assertEqual(cc.load_audio_features(paths["cache"]), (None, cc.MALFORMED_AUDIO_FEATURES))

    def test_well_formed_edge_values_are_accepted(self):
        paths = self.cache_with(
            raw_speech=[[0.0, 0.0], [1, 2]], raw_gaps=[{"start": 0, "end": 0, "rms": 0, "onset": 0}])
        with fx.patched_audio_helpers():
            features, reason = cc.load_audio_features(paths["cache"])
        self.assertIsNone(reason)
        self.assertEqual(features["speech"], [[0.0, 0.0], [1, 2]])

    @unittest.skipIf(cc.features_config_hash is None, "needs #16's fingerprint helpers (not merged yet)")
    def test_malformed_payloads_are_rejected_with_the_real_fingerprint_helpers(self):
        # Stamp with #16's actual helpers, so the cache is current and only its payload is bad.
        for name, payload in MALFORMED_PAYLOADS.items():
            with self.subTest(payload=name):
                paths = fx.make_media_cache(self.root, f"real-{abs(hash(name))}")
                real_hash = cc.features_config_hash(
                    paths["manifest"]["audio_path"], None, paths["manifest"].get("config_hash"))
                fx.write_audio_features(paths["cache"], paths["manifest"], hash_override=real_hash, **payload)
                self.assertEqual(cc.load_audio_features(paths["cache"]), (None, cc.MALFORMED_AUDIO_FEATURES))


if __name__ == "__main__":
    unittest.main()
