from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.datasets.mtg_ad import (
    EXPECTED_SPLIT_COUNTS, YTDLP_DOWNLOAD_TIMEOUT_SECONDS, YTDLP_FORMAT,
    YTDLP_SOCKET_TIMEOUT_SECONDS, annotation_file_is_valid, download_annotations,
    download_video, load_split, parse_row, select_dev, select_stratified,
)
from src.datasets.srt import SubtitleItem
from src.evaluation.parsing import ADItem
from src.generation.hf_vlm import build_conversation, clean, resolve_model
from src.generation.prompts import build_prompt, transcript_context, word_limit_for_gap
from src.utils.selection import k_for_duration, uniform_timestamps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_mtg_ad  # noqa: E402
from build_moments import load_targets  # noqa: E402
from run_ad_vlm import done_ids, prepare, read_jsonl  # noqa: E402


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

    def test_load_split_decodes_utf8(self):
        row = make_row("a", num_ad=1)
        row["gt_ad_segments"][0]["text"] = "č"
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "validation.jsonl").write_bytes((json.dumps(row) + "\n").encode("utf-8"))
            self.assertEqual(load_split(tmp, "validation")[0].ad[0].text, "č")

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

    def test_download_reuses_decodable_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp, "vid.mp4")
            target.write_bytes(b"video")
            info = SimpleNamespace(has_video=True)
            with (
                patch("src.datasets.mtg_ad.shutil.which", return_value="/bin/ffprobe"),
                patch("src.datasets.mtg_ad.probe", return_value=info),
                patch("src.datasets.mtg_ad.subprocess.run") as run,
            ):
                self.assertEqual(download_video(parse_row(make_row()), tmp), target)
            run.assert_not_called()

    def test_download_replaces_undecodable_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp, "vid.mp4")
            target.write_bytes(b"not a video")
            with (
                patch("src.datasets.mtg_ad.shutil.which", return_value="/bin/yt-dlp"),
                patch("src.datasets.mtg_ad.probe", side_effect=RuntimeError("invalid")),
                patch("src.datasets.mtg_ad.subprocess.run") as run,
            ):
                run.return_value.returncode = 0
                run.return_value.stderr = ""
                with self.assertRaisesRegex(RuntimeError, "yt-dlp failed"):
                    download_video(parse_row(make_row()), tmp)
            run.assert_called_once()
            self.assertFalse(target.exists())

    def test_download_requires_ffprobe_before_reusing_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp, "vid.mp4")
            target.write_bytes(b"video")
            with (
                patch("src.datasets.mtg_ad.shutil.which", return_value=None),
                patch("src.datasets.mtg_ad.subprocess.run") as run,
            ):
                with self.assertRaisesRegex(RuntimeError, "ffprobe"):
                    download_video(parse_row(make_row()), tmp)
            run.assert_not_called()
            self.assertTrue(target.is_file())

    def test_download_timeout_removes_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("src.datasets.mtg_ad.shutil.which", return_value="/bin/yt-dlp"),
                patch(
                    "src.datasets.mtg_ad.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(cmd="yt-dlp", timeout=1),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    download_video(parse_row(make_row()), tmp)
            self.assertFalse(Path(tmp, "vid.mp4").exists())

    def test_download_rejects_zero_byte_output_and_uses_progressive_https_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp, "vid.mp4")
            target.touch()
            with (
                patch("src.datasets.mtg_ad.shutil.which", return_value="/bin/yt-dlp"),
                patch("src.datasets.mtg_ad.subprocess.run") as run,
            ):
                run.return_value.returncode = 0
                run.return_value.stderr = ""
                with self.assertRaisesRegex(RuntimeError, "yt-dlp failed"):
                    download_video(parse_row(make_row()), tmp)
            command = run.call_args.args[0]
            self.assertIn("--retries", command)
            self.assertIn("--socket-timeout", command)
            self.assertEqual(
                command[command.index("--socket-timeout") + 1],
                str(YTDLP_SOCKET_TIMEOUT_SECONDS),
            )
            self.assertEqual(run.call_args.kwargs["timeout"], YTDLP_DOWNLOAD_TIMEOUT_SECONDS)
            self.assertEqual(command[command.index("-f") + 1], YTDLP_FORMAT)
            self.assertIn("protocol=https", YTDLP_FORMAT)
            self.assertNotIn("protocol^=http", YTDLP_FORMAT)
            self.assertFalse(target.exists())

    def test_download_passes_local_browser_auth_and_extractor_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp, "vid.mp4")

            def successful_download(command, **_kwargs):
                target.write_bytes(b"video")
                return type("Result", (), {"returncode": 0, "stderr": ""})()

            with (
                patch("src.datasets.mtg_ad.shutil.which", return_value="/bin/yt-dlp"),
                patch("src.datasets.mtg_ad.subprocess.run", side_effect=successful_download) as run,
            ):
                result = download_video(
                    parse_row(make_row()), tmp,
                    cookies_from_browser="chrome:Default",
                    extractor_args="youtube:player_client=mweb,tv",
                )
            self.assertEqual(result, target)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--cookies-from-browser") + 1], "chrome:Default")
            self.assertEqual(
                command[command.index("--extractor-args") + 1],
                "youtube:player_client=mweb,tv",
            )

    def test_download_rejects_two_cookie_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "either"):
                download_video(
                    parse_row(make_row()), tmp,
                    cookies="cookies.txt", cookies_from_browser="chrome",
                )


def _jsonl_bytes(count: int) -> bytes:
    return "".join(json.dumps({"i": i}) + "\n" for i in range(count)).encode("utf-8")


class AnnotationDownloadTests(unittest.TestCase):
    def _urlopen(self, bodies):
        class Response:
            def __init__(self, payload):
                self.payload = payload

            def read(self):
                return self.payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def urlopen(url, timeout=None):
            self.assertEqual(timeout, 60)
            for suffix, payload in bodies.items():
                if str(url).endswith(suffix):
                    return Response(payload)
            raise AssertionError(url)

        return urlopen

    def test_invalid_download_keeps_existing_file_and_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp, "validation.jsonl")
            existing.write_bytes("keep\n".encode("utf-8"))
            bodies = {"data/annotations/validation.jsonl": b"\xff"}
            with patch("src.datasets.mtg_ad.urllib.request.urlopen", side_effect=self._urlopen(bodies)):
                with self.assertRaisesRegex(RuntimeError, "validation.jsonl"):
                    download_annotations(tmp)
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep\n")
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_complete_download_replaces_partial_jsonl(self):
        bodies = {
            "data/annotations/validation.jsonl": _jsonl_bytes(EXPECTED_SPLIT_COUNTS["validation"]),
            "data/annotations/test.jsonl": _jsonl_bytes(EXPECTED_SPLIT_COUNTS["test"]),
            "validation_video_ids.txt": "a\n".encode("utf-8"),
            "test_video_ids.txt": "b\n".encode("utf-8"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "validation.jsonl").write_bytes(b"partial\n")
            with patch("src.datasets.mtg_ad.urllib.request.urlopen", side_effect=self._urlopen(bodies)):
                download_annotations(tmp)
            self.assertTrue(annotation_file_is_valid(Path(tmp, "validation.jsonl")))
            self.assertTrue(annotation_file_is_valid(Path(tmp, "test.jsonl")))
            self.assertEqual(Path(tmp, "validation_video_ids.txt").read_text(encoding="utf-8"), "a\n")

    def test_main_refuses_to_rewrite_splits_for_short_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            splits = Path(tmp, "splits.json")
            splits.write_text("original\n", encoding="utf-8")
            with (
                patch.object(prepare_mtg_ad, "download_annotations"),
                patch.object(
                    prepare_mtg_ad, "load_split",
                    side_effect=[[parse_row(make_row())], []],
                ),
                patch.object(sys, "argv", [
                    "prepare_mtg_ad.py", "--data-dir", tmp, "--splits-out", str(splits),
                ]),
            ):
                with self.assertRaises(SystemExit) as raised:
                    prepare_mtg_ad.main()
            self.assertEqual(raised.exception.code, 1)
            self.assertEqual(splits.read_text(encoding="utf-8"), "original\n")


class PrepareTargetsTests(unittest.TestCase):
    def test_batch_reports_missing_source_and_continues_after_preprocess_error(self):
        videos = [parse_row(make_row(video_id)) for video_id in ("good", "missing", "bad")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            cache = root / "cache"
            (data / "videos").mkdir(parents=True)
            for video_id in ("good", "bad"):
                (data / "videos" / f"{video_id}.mp4").write_bytes(b"video")

            def fake_preprocess(path, cache_dir, fps):
                if path.stem == "bad":
                    raise RuntimeError("broken media")
                cache_dir.mkdir(parents=True)
                (cache_dir / "manifest.json").write_text("{}")
                return {
                    "media_info": {"duration": 300.0},
                    "frames": [{"path": "frame.jpg"}],
                    "audio_path": "audio.wav",
                }

            with patch.object(prepare_mtg_ad, "preprocess_video", side_effect=fake_preprocess):
                results = prepare_mtg_ad.prepare_targets(
                    videos, data, cache, download_videos=False, preprocess=True, fps=2.0,
                )
        self.assertEqual([item["status"] for item in results], ["ready", "unavailable", "failed"])
        self.assertEqual(results[0]["preprocess"], "written")
        self.assertEqual(results[1]["stage"], "source")
        self.assertEqual(results[2]["stage"], "preprocess")

    def test_batch_marks_unchanged_manifest_as_reused(self):
        video = parse_row(make_row("cached"))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            cache_dir = root / "cache" / "cached"
            (data / "videos").mkdir(parents=True)
            (data / "videos" / "cached.mp4").write_bytes(b"video")
            cache_dir.mkdir(parents=True)
            (cache_dir / "manifest.json").write_text("{}")
            manifest = {
                "media_info": {"duration": 300.0},
                "frames": [{"path": "frame.jpg"}],
                "audio_path": None,
            }
            with patch.object(prepare_mtg_ad, "preprocess_video", return_value=manifest):
                [result] = prepare_mtg_ad.prepare_targets(
                    [video], data, root / "cache", False, True, 2.0,
                )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["preprocess"], "reused")


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
        self.assertEqual(word_limit_for_gap(0.2), 0)
        self.assertEqual(word_limit_for_gap(0.4), 1)
        self.assertEqual(word_limit_for_gap(0), 0)

    def test_build_prompt(self):
        with_speech = build_prompt("P0", 75.5, 4, 2.0, 12, "nice dog")
        self.assertIn("4 frames", with_speech)
        self.assertIn("1:15.5", with_speech)
        self.assertIn("at most 12 words", with_speech)
        self.assertIn('"nice dog"', with_speech)
        self.assertNotIn("speech", build_prompt("P0", 1.0, 4, 2.0, 12))
        with self.assertRaises(ValueError):
            build_prompt("P9", 1.0, 4, 2.0, 12)
        with self.assertRaisesRegex(ValueError, "not safely speakable"):
            build_prompt("P0", 1.0, 4, 2.0, 0)

    def test_word_cap_has_no_minimum_and_rejects_unknown_nonfinite_capacity(self):
        for gap, expected in ((0.0, 0), (-2.0, 0), (0.2, 0), (0.4, 1), (1.0, 3), (20.0, 30)):
            with self.subTest(gap=gap):
                self.assertEqual(word_limit_for_gap(gap), expected)
        for gap in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                word_limit_for_gap(gap)
        self.assertEqual(word_limit_for_gap(None, max_words=5), 5)

    def test_generator_rejects_zero_budget_before_opening_frames(self):
        with self.assertRaisesRegex(ValueError, "zero-word budget"):
            prepare({"word_limit": 0}, SimpleNamespace())


class GenerationHelperTests(unittest.TestCase):
    def test_clean_and_models(self):
        self.assertEqual(clean("<think>hmm</think> A dog runs.<|im_end|>"), "A dog runs.")
        self.assertEqual(resolve_model("qwen3-vl-8b"), "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(resolve_model("org/custom"), "org/custom")

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
