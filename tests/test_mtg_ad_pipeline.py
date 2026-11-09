from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
import unittest

from src.datasets.mtg_ad import load_split, parse_row, select_dev, select_stratified
from src.datasets.srt import SubtitleItem
from src.evaluation.parsing import ADItem
from src.generation.hf_vlm import build_conversation, clean, resolve_model
from src.generation.prompts import build_prompt, transcript_context, word_limit_for_gap
from src.utils.selection import k_for_duration, uniform_timestamps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_moments import load_targets  # noqa: E402
from run_ad_vlm import done_ids, read_jsonl  # noqa: E402


def make_row(video_id="vid", category="Vlog", num_ad=4, duration=300, split="validation"):
    return {
        "video_id": video_id, "split": split, "category": category,
        "duration_seconds": duration,
        "gt_ad_segments": [
            {"start_seconds": float(10 * (num_ad - i)), "text": f" ad {i} "} for i in range(num_ad)
        ],
        "transcript_segments": [
            {"start_seconds": 20.0, "end_seconds": 25.0, "text": "second"},
            {"start_seconds": 1.0, "end_seconds": 4.0, "text": "first"},
        ],
        "visual_info": None,
    }


class MTGADTests(unittest.TestCase):
    def test_parse_row_sorts_and_strips(self):
        video = parse_row(make_row(num_ad=3))
        self.assertEqual(video.ad, [ADItem(10.0, "ad 2"), ADItem(20.0, "ad 1"), ADItem(30.0, "ad 0")])
        self.assertEqual(video.transcript[0], SubtitleItem(1.0, 4.0, "first"))
        self.assertEqual(video.youtube_url, "https://www.youtube.com/watch?v=vid")
        self.assertEqual(video.video_filename, "vid.mp4")

    def test_parse_row_without_speech(self):
        row = make_row()
        row["transcript_segments"] = []
        self.assertEqual(parse_row(row).transcript, [])

    def test_load_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [make_row("a"), make_row("b")]
            Path(tmp, "validation.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            self.assertEqual([v.video_id for v in load_split(tmp, "validation")], ["a", "b"])
            Path(tmp, "test.jsonl").write_text('{"video_id": "x"}\n')
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_split(tmp, "test")
        with self.assertRaises(ValueError):
            load_split("unused", "train")

    def test_stratified_sample_matches_category_mix(self):
        videos = [parse_row(make_row(f"v{i:03d}", "Vlog")) for i in range(60)]
        videos += [parse_row(make_row(f"e{i:03d}", "Education")) for i in range(30)]
        videos += [parse_row(make_row(f"o{i:03d}", "Other")) for i in range(10)]
        sample = select_stratified(videos, 10, seed=2)
        self.assertEqual(Counter(v.category for v in sample), {"Vlog": 6, "Education": 3, "Other": 1})
        self.assertEqual(sample, select_stratified(list(reversed(videos)), 10, seed=2))
        self.assertNotEqual(sample, select_stratified(videos, 10, seed=3))
        self.assertEqual(len(select_stratified(videos[:5], 10)), 5)

    def test_dev_requires_enough_ad(self):
        videos = [parse_row(make_row(f"v{i}", num_ad=2 + i % 2)) for i in range(20)]
        dev = select_dev(videos, 5, min_ad_items=3)
        self.assertEqual(len(dev), 5)
        self.assertTrue(all(len(v.ad) >= 3 for v in dev))


class LoadTargetsTests(unittest.TestCase):
    def test_mtg_ad_dev_and_test_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            for split in ("validation", "test"):
                rows = [make_row("a", split=split, duration=120), make_row("b", split=split, duration=200)]
                Path(tmp, f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            splits = {"dev": {"videos": [{"video_id": "b"}]}, "test": {"video_ids": ["a"]}}
            dev = load_targets("mtg-ad", "dev", splits, Path(tmp))
            test = load_targets("mtg-ad", "test", splits, Path(tmp))
        self.assertEqual([(i, d) for i, d, _ in dev], [("b", 200.0)])
        self.assertEqual([(i, d) for i, d, _ in test], [("a", 120.0)])
        self.assertEqual(dev[0][2][0].text, "first")


class SelectionTests(unittest.TestCase):
    def test_k_rule(self):
        self.assertEqual(k_for_duration(300, 3.5), 18)
        self.assertEqual(k_for_duration(300, 5.0), 25)
        self.assertEqual(k_for_duration(5, 3.5), 1)
        with self.assertRaises(ValueError):
            k_for_duration(0)

    def test_uniform_excludes_endpoints(self):
        self.assertEqual(uniform_timestamps(10, 4), [2.0, 4.0, 6.0, 8.0])
        self.assertEqual(uniform_timestamps(9, 1), [4.5])
        with self.assertRaises(ValueError):
            uniform_timestamps(10, 0)


class PromptTests(unittest.TestCase):
    TRANSCRIPT = [SubtitleItem(0, 4, "hello"), SubtitleItem(10, 12, "there"), SubtitleItem(50, 55, "far")]

    def test_transcript_context_window(self):
        self.assertEqual(transcript_context(self.TRANSCRIPT, 15.0, window=15.0), "hello there")
        self.assertEqual(transcript_context(self.TRANSCRIPT, 100.0), "")

    def test_word_limit(self):
        self.assertEqual(word_limit_for_gap(None), 15)
        self.assertEqual(word_limit_for_gap(2.9), 8)
        self.assertEqual(word_limit_for_gap(0.2), 3)

    def test_build_prompt(self):
        with_speech = build_prompt("P0", 75.5, 4, 2.0, 12, "nice dog")
        self.assertIn("4 frames", with_speech)
        self.assertIn("1:15.5", with_speech)
        self.assertIn("at most 12 words", with_speech)
        self.assertIn('"nice dog"', with_speech)
        self.assertNotIn("speech", build_prompt("P0", 1.0, 4, 2.0, 12))
        with self.assertRaises(ValueError):
            build_prompt("P9", 1.0, 4, 2.0, 12)


class GenerationHelperTests(unittest.TestCase):
    def test_clean_and_models(self):
        self.assertEqual(clean("<think>hmm</think> A dog runs.<|im_end|>"), "A dog runs.")
        self.assertEqual(resolve_model("qwen3-vl-8b"), "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(resolve_model("org/custom"), "org/custom")

    def test_clean_extracts_muse_final_answer(self):
        scratchpad = (
            "to=selfThese 4 frames are from a video, in order, spanning about 2 seconds around 0:02.4.\n"
            "Write one audio description ... Output only the description.\n\n"
            "Maybe: Color bars with moving rainbow line.\n\n"
            'Count: Rainbow(1) line2 moves3 up4. Good.\n\nOutput only description.'
        )
        raw = scratchpad + "assistant to=userRainbow line moves across color bars while gray shapes change."
        self.assertEqual(clean(raw), "Rainbow line moves across color bars while gray shapes change.")
        self.assertEqual(clean(raw + "<|im_end|>"), "Rainbow line moves across color bars while gray shapes change.")
        # Cut off before the answer: the marker is missing, so the scratchpad is returned unchanged.
        self.assertTrue(clean(scratchpad).startswith("to=selfThese 4 frames"))
        # Other models have no marker and pass through untouched.
        self.assertEqual(clean("A dog runs across the yard."), "A dog runs across the yard.")

    def test_conversation_orders_frames_before_text(self):
        conversation = build_conversation(["f1", "f2"], "Describe.", system="Be brief.")
        self.assertEqual(conversation[0]["role"], "system")
        content = conversation[1]["content"]
        self.assertEqual([c["type"] for c in content], ["image", "image", "text"])
        self.assertEqual(content[1]["image"], "f2")

    def test_resume_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp, "out.jsonl")
            out.write_text('{"id": "a", "output": "x"}\n{"id": "b", "error": "oom"}\n{"id": "c", "ou')
            self.assertEqual(done_ids(out), {"a"})
            self.assertEqual(done_ids(Path(tmp, "missing.jsonl")), set())
            moments = Path(tmp, "m.jsonl")
            moments.write_text('{"id": "m", "frames": [], "time": 1.0}\n\n')
            self.assertEqual(read_jsonl(moments)[0]["id"], "m")
            moments.write_text('{"id": "m"}\n')
            with self.assertRaises(ValueError):
                read_jsonl(moments)


if __name__ == "__main__":
    unittest.main()
