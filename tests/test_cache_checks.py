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


if __name__ == "__main__":
    unittest.main()
