from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest

from src.generation.vlm_client import VLMClient
from src.utils.media import frame_window, preprocess_video, probe

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_test_video(path: Path, seconds: float = 3.0, audio: bool = True) -> None:
    command = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=25:duration={seconds}",
    ]
    if audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    command += ["-c:a", "aac", "-ac", "2"] if audio else []
    command += ["-shortest", str(path)]
    subprocess.run(command, check=True)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not installed")
class MediaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.video = self.dir / "clip.mp4"
        make_test_video(self.video)

    def tearDown(self):
        self.tmp.cleanup()

    def test_probe(self):
        info = probe(self.video)
        self.assertTrue(info.has_video and info.has_audio)
        self.assertAlmostEqual(info.duration, 3.0, delta=0.15)
        self.assertEqual((info.width, info.height), (320, 240))
        self.assertAlmostEqual(info.video_fps, 25.0)

    def test_preprocess_writes_audio_frames_manifest_and_caches(self):
        cache = self.dir / "cache"
        manifest = preprocess_video(self.video, cache, fps=2.0, max_side=160)
        self.assertEqual(manifest["video_id"], "clip")
        self.assertEqual([f["timestamp"] for f in manifest["frames"]], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        audio = probe(manifest["audio_path"])
        self.assertEqual((audio.audio_sample_rate, audio.audio_channels), (16000, 1))
        self.assertFalse(audio.has_video)
        first_frame = probe(manifest["frames"][0]["path"])
        self.assertEqual((first_frame.width, first_frame.height), (160, 120))

        mtime = (cache / "manifest.json").stat().st_mtime_ns
        self.assertEqual(preprocess_video(self.video, cache, fps=2.0, max_side=160), manifest)
        self.assertEqual((cache / "manifest.json").stat().st_mtime_ns, mtime)
        changed = preprocess_video(self.video, cache, fps=1.0, max_side=160)
        self.assertEqual(len(changed["frames"]), 3)
        self.assertEqual(len(list((cache / "frames").glob("*.jpg"))), 3)

    def test_video_without_audio(self):
        silent = self.dir / "silent.mp4"
        make_test_video(silent, audio=False)
        manifest = preprocess_video(silent, self.dir / "silent_cache", fps=1.0)
        self.assertIsNone(manifest["audio_path"])
        self.assertEqual(len(manifest["frames"]), 3)


class FrameWindowTests(unittest.TestCase):
    MANIFEST = {"frames": [
        {"index": i, "timestamp": i * 0.5, "path": f"{i:06d}.jpg"} for i in range(11)
    ]}

    def test_window_spans_center(self):
        window = frame_window(self.MANIFEST, 2.0, num_frames=4, span_seconds=2.0)
        self.assertEqual([f.timestamp for f in window], [1.0, 1.5, 2.5, 3.0])

    def test_window_clamps_and_keeps_count(self):
        window = frame_window(self.MANIFEST, 0.0, num_frames=4, span_seconds=2.0)
        self.assertEqual(len(window), 4)
        self.assertEqual(window[0].timestamp, 0.0)
        self.assertEqual(frame_window(self.MANIFEST, 9.0, 1)[0].timestamp, 5.0)
        self.assertEqual(frame_window({"frames": []}, 1.0), [])


class _FakeChatHandler(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).received.append((self.path, self.headers.get("Authorization"), body))
        payload = json.dumps({
            "model": body["model"],
            "choices": [{"message": {"content": "  A man opens the door. "}}],
            "usage": {"prompt_tokens": 321, "completion_tokens": 7},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class VLMClientTests(unittest.TestCase):
    def test_request_shape_and_result(self):
        _FakeChatHandler.received = []
        server = HTTPServer(("127.0.0.1", 0), _FakeChatHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                frames = []
                for name, data in (("a.jpg", b"\xff\xd8one"), ("b.png", b"\x89PNGtwo")):
                    Path(tmp, name).write_bytes(data)
                    frames.append(Path(tmp, name))
                client = VLMClient("test/model", f"http://127.0.0.1:{server.server_port}/v1/", "k")
                result = client.generate(frames, "Describe.", system="Be brief.", max_tokens=20)
        finally:
            server.shutdown()
            server.server_close()

        path, auth, body = _FakeChatHandler.received[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(auth, "Bearer k")
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["messages"][0], {"role": "system", "content": "Be brief."})
        content = body["messages"][1]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(content[2], {"type": "text", "text": "Describe."})
        self.assertEqual(result.text, "A man opens the door.")
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (321, 7))
        self.assertEqual(result.request["num_images"], 2)
        self.assertNotIn("messages", result.request)


if __name__ == "__main__":
    unittest.main()
