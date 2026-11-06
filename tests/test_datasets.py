from collections import Counter
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import build_moments, prepare_reframed

from src.datasets.egoschema import load_items, parse_items, split_pilot
from src.datasets.reframed import (
    TEST_SELECTION_VERSION, decrypt_row, load_split, merge_cues, parse_row, select_pilot, select_test,
)
from src.datasets.srt import SubtitleItem, parse_srt, srt_time_to_seconds

SRT = (
    "﻿1\r\n00:02:22,350 --> 00:02:23,060\r\nShe grabs him.\r\n\r\n"
    "2\n00:02:42,700 --> 00:02:44,710\nShe leaves Mark\nat the table.\n\n"
)


def make_row(video_id="vid1", imdb_id="tt1", ad_us=SRT, screenplay="null"):
    return {
        "rottentomatoes_id": "movie", "imdb_id": imdb_id, "video_id": video_id,
        "audio_description_us": ad_us, "audio_description_uk": "", "subtitles": SRT,
        "sdh": SRT, "screenplay": screenplay, "movie_duration": "7241.38",
        "video_start": "154.40", "video_duration": "167.58",
    }


class SrtTests(unittest.TestCase):
    def test_parses_bom_crlf_and_multiline(self):
        items = parse_srt(SRT)
        self.assertEqual(items[0], SubtitleItem(142.35, 143.06, "She grabs him."))
        self.assertEqual(items[1].text, "She leaves Mark\nat the table.")
        self.assertAlmostEqual(items[1].end, 164.71)

    def test_empty_and_missing_index(self):
        self.assertEqual(parse_srt(""), [])
        self.assertEqual(parse_srt(None), [])
        self.assertEqual(parse_srt("00:00:01.5 --> 00:00:02,000\nx")[0].start, 1.5)

    def test_rejects_malformed(self):
        with self.assertRaises(ValueError):
            parse_srt("1\nnot a timing line\ntext")
        with self.assertRaises(ValueError):
            parse_srt("1\n00:00:05,000 --> 00:00:04,000\ntext")
        with self.assertRaises(ValueError):
            srt_time_to_seconds("00:61:00,000")


class ReframedTests(unittest.TestCase):
    def test_parse_row(self):
        clip = parse_row(make_row(screenplay='{"scene_1": "<scene/>"}'), "validation")
        self.assertEqual(len(clip.ad_us), 2)
        self.assertEqual(clip.ad_uk, [])
        self.assertEqual(clip.screenplay, {"scene_1": "<scene/>"})
        self.assertAlmostEqual(clip.video_duration, 167.58)
        self.assertEqual(clip.movie_id, "tt1")
        self.assertIn("vid1", clip.video_url)
        self.assertIsNone(parse_row(make_row(), "validation").screenplay)

    def test_merge_cues(self):
        items = [SubtitleItem(0, 1, "a,"), SubtitleItem(1.2, 2, "b."), SubtitleItem(5, 6, "c")]
        merged = merge_cues(items, max_gap=0.5)
        self.assertEqual(merged, [SubtitleItem(0, 2, "a, b."), SubtitleItem(5, 6, "c")])

    def test_select_pilot_one_per_movie_and_deterministic(self):
        clips = [
            parse_row(make_row(video_id=f"v{m}{i}", imdb_id=f"tt{m}", ad_us=SRT * 2), "validation")
            for m in range(4) for i in range(3)
        ]
        pilot = select_pilot(clips, num_clips=6, min_ad_items=3, seed=1)
        self.assertEqual(len(pilot), 6)
        self.assertEqual(len({c.movie_id for c in pilot[:4]}), 4)
        self.assertEqual(pilot, select_pilot(list(reversed(clips)), 6, 3, seed=1))
        self.assertEqual(select_pilot(clips, 6, min_ad_items=5), [])

    def test_load_validation_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [make_row(video_id="a"), make_row(video_id="b")]
            Path(tmp, "validation.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            clips = load_split(tmp, "validation")
        self.assertEqual([c.video_id for c in clips], ["a", "b"])
        with self.assertRaises(ValueError):
            load_split("unused", "dev")

    def test_test_selection_ignores_text_counts_and_density_when_available(self):
        clips = [
            parse_row(make_row(video_id=f"v{m}{i}", imdb_id=f"tt{m}"), "test")
            for m in range(4) for i in range(3)
        ]
        original = [c.video_id for c in select_test(clips, 6, seed=1)]
        sparse = [replace(c, ad_us=[SubtitleItem(100, 200, "One changed cue")],
                          ad_uk=[], subtitles=[SubtitleItem(20, 21, "Changed subtitle")],
                          sdh=[], screenplay=None) for c in clips]
        changed = [replace(c, ad_us=c.ad_us * 20, ad_uk=c.ad_us * 10,
                           subtitles=[SubtitleItem(0, 1, "Different text")],
                           screenplay={"scene": "Different annotation"}) for c in clips]
        self.assertEqual(original, [c.video_id for c in select_test(sparse, 6, seed=1)])
        self.assertEqual(original, [c.video_id for c in select_test(changed, 6, seed=1)])
        # These availability objects forbid inspecting cue counts or content.
        class Available:
            def __bool__(self):
                return True

            def __len__(self):
                raise AssertionError("test selection must not inspect cue counts")

            def __iter__(self):
                raise AssertionError("test selection must not inspect cue contents")

        metadata = [SimpleNamespace(video_id=c.video_id, movie_id=c.movie_id,
                                    ad_us=Available(), ad_uk=[], subtitles=Available()) for c in clips]
        self.assertEqual(original, [c.video_id for c in select_test(metadata, 6, seed=1)])

    def test_test_selection_requires_us_or_uk_reference_availability(self):
        base = parse_row(make_row(), "test")
        clips = [replace(base, video_id="us"),
                 replace(base, video_id="uk", ad_us=[], ad_uk=base.ad_us),
                 replace(base, video_id="neither", ad_us=[], ad_uk=[])]
        self.assertEqual({c.video_id for c in select_test(clips, 10)}, {"us", "uk"})

    def test_test_selection_requires_subtitle_availability(self):
        base = parse_row(make_row(), "test")
        clips = [replace(base, video_id="present"), replace(base, video_id="absent", subtitles=[])]
        self.assertEqual([c.video_id for c in select_test(clips, 10)], ["present"])

    def test_committed_test_config_contains_per_movie_counts(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/reframed_splits.json").read_text())
        test = config["test"]
        self.assertEqual(test["selection_version"], TEST_SELECTION_VERSION)
        self.assertEqual(sum(test["per_movie_counts"].values()), len(test["video_ids"]))
        self.assertEqual(len(test["per_movie_counts"]), test["num_movies"])
        self.assertTrue(all(count > 0 for count in test["per_movie_counts"].values()))

    def test_test_selection_is_movie_balanced_seeded_and_order_independent(self):
        clips = [
            parse_row(make_row(video_id=f"v{m}{i}", imdb_id=f"tt{m}"), "test")
            for m in range(4) for i in range(3)
        ]
        selected = select_test(clips, 6, seed=1)
        self.assertEqual(len({c.movie_id for c in selected[:4]}), 4)
        self.assertEqual(sorted(Counter(c.movie_id for c in selected).values()), [1, 1, 2, 2])
        self.assertEqual(selected, select_test(list(reversed(clips)), 6, seed=1))
        self.assertNotEqual(selected, select_test(clips, 6, seed=2))
        self.assertEqual(len(select_test(clips, 100)), len(clips))
        self.assertEqual(select_test(clips, 0), [])

    def test_prepare_uses_pilot_criteria_only_for_validation(self):
        validation = [parse_row(make_row(ad_us=SRT * 2), "validation")]
        test = [
            parse_row(make_row(video_id=f"test{i}", imdb_id=f"tt{i % 2}"), "test")
            for i in range(6)
        ]
        test[0] = replace(test[0], ad_us=[], ad_uk=[])
        test[1] = replace(test[1], subtitles=[])
        test[2] = replace(test[2], ad_us=[], ad_uk=test[2].ad_us)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp, "splits.json")
            with (
                patch("sys.argv", ["prepare_reframed.py", "--num-pilot", "1", "--num-test", "6",
                                   "--splits-out", str(output)]),
                patch.object(prepare_reframed, "download_metadata"),
                patch.object(prepare_reframed, "load_split", side_effect=[validation, test]),
                redirect_stdout(io.StringIO()),
            ):
                prepare_reframed.main()
            record = json.loads(output.read_text())
        self.assertEqual(len(record["clips"]), 1)
        self.assertEqual(set(record["test"]["video_ids"]), {"test2", "test3", "test4", "test5"})
        self.assertEqual(record["test"]["per_movie_counts"], {"tt0": 2, "tt1": 2})
        self.assertEqual(record["test"]["selection_version"], TEST_SELECTION_VERSION)

    def test_legacy_test_selection_is_rejected_until_regenerated(self):
        splits = {"test": {"video_ids": ["vid1"]}}
        for version in (None, "movie-balanced-metadata-v1", "annotation-filtered"):
            splits["test"]["selection_version"] = version
            with self.subTest(version=version), patch.object(build_moments.reframed, "load_split") as loader:
                with self.assertRaisesRegex(ValueError, "legacy REFRAMED test selection"):
                    build_moments.load_targets("reframed", "test", splits, Path("unused"))
                loader.assert_not_called()
        splits["test"]["selection_version"] = TEST_SELECTION_VERSION
        clip = parse_row(make_row(), "test")
        with patch.object(build_moments.reframed, "load_split", return_value=[clip]):
            targets = build_moments.load_targets("reframed", "test", splits, Path("unused"))
        self.assertEqual(targets[0][0], "vid1")

    def test_decrypt_row_json_decodes_each_field(self):
        encrypted = {k: f"enc:{json.dumps(v)}" for k, v in make_row().items()}
        row = decrypt_row(encrypted, lambda value: value.removeprefix("enc:"))
        self.assertEqual(row, make_row())
        clip = load_split_with_decryptor(encrypted)
        self.assertEqual(clip.video_id, "vid1")


def load_split_with_decryptor(encrypted_row):
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "test.jsonl").write_text(json.dumps(encrypted_row) + "\n")
        return load_split(tmp, "test", decrypt=lambda value: value.removeprefix("enc:"))[0]


class EgoSchemaTests(unittest.TestCase):
    QUESTIONS = [
        {"q_uid": f"q{i:02d}", "google_drive_id": f"g{i}", "question": f"Q{i}?",
         **{f"option {j}": f"opt{j}" for j in range(5)}}
        for i in range(20)
    ]

    def test_parse_and_labeled_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "questions.json").write_text(json.dumps(self.QUESTIONS))
            Path(tmp, "subset_answers.json").write_text(json.dumps({"q03": 2, "q07": 0}))
            labeled = load_items(tmp)
            everything = load_items(tmp, labeled_only=False)
        self.assertEqual([(i.q_uid, i.answer) for i in labeled], [("q03", 2), ("q07", 0)])
        self.assertEqual(len(everything), 20)
        self.assertEqual(labeled[0].options, tuple(f"opt{j}" for j in range(5)))
        self.assertEqual(labeled[0].video_filename, "q03.mp4")

    def test_rejects_out_of_range_answer(self):
        with self.assertRaises(ValueError):
            parse_items(self.QUESTIONS[:1], {"q00": 5})

    def test_pilot_split_is_disjoint_and_deterministic(self):
        items = parse_items(self.QUESTIONS, {q["q_uid"]: 1 for q in self.QUESTIONS})
        pilot, rest = split_pilot(items, num_pilot=5, seed=3)
        self.assertEqual(len(pilot), 5)
        self.assertEqual(len(rest), 15)
        self.assertFalse({i.q_uid for i in pilot} & {i.q_uid for i in rest})
        self.assertEqual(pilot, split_pilot(list(reversed(items)), 5, seed=3)[0])


if __name__ == "__main__":
    unittest.main()
