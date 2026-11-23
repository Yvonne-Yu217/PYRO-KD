"""Ablation 5 must not receive dataset subtitles through the legacy path."""
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from src.experiments.generation_stage import moments_path, run_generation
from src.experiments.matrix import ablation5_runs
from src.generation.backends import GenResult


class CapturingBackend:
    name = 'capture'
    model_id = 'synthetic-test'

    def __init__(self, supports_audio):
        self.supports_audio = supports_audio
        self.requests = []

    def generate(self, requests):
        self.requests.extend(requests)
        return [GenResult(r.id, text='A tree.', new_tokens=3) for r in requests]


class AblationContextTests(TestCase):
    def test_three_conditions_exclude_subtitles_and_isolate_audio_inputs(self):
        cfg = {
            'models': {'qwen': {'backend': 'hf_vlm'}, 'omni': {'backend': 'qwen_omni'}},
            'ablations': {'timeline': {'selector': 'A_cue', 'model': 'qwen',
                                       'audio_native_model': 'omni', 'prior': 'P0'}},
            'generation': {'word_limit_policy': 'flag'},
        }
        runs = ablation5_runs(cfg)
        moment = {'id': 'clip_A_0', 'video_id': 'clip', 'selector': 'A_cue',
                  'time': 2., 'frames': ['frame.jpg'], 'frame_times': [2.],
                  'word_limit': 10, 'transcript': 'SUBTITLE_SENTINEL',
                  'audio_timeline': 'TIMELINE_SENTINEL',
                  'audio_clip': {'path': 'audio.wav', 'start': 0., 'end': 4.}}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = moments_path(root, runs[0].selection)
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(moment) + '\n')
            for run in runs:
                with self.subTest(model=run.model_label, timeline=run.timeline):
                    native = run.model_label == 'omni'
                    backend = CapturingBackend(native)
                    run_generation(cfg, run, root, lambda _: backend)
                    request, = backend.requests
                    self.assertNotIn('SUBTITLE_SENTINEL', request.prompt)
                    self.assertEqual('TIMELINE_SENTINEL' in request.prompt, run.timeline)
                    self.assertEqual(request.audio, moment['audio_clip'] if native else None)
                    self.assertEqual(request.images, moment['frames'])
            # Legacy non-ablation calls intentionally retain their old context.
            legacy = replace(runs[0], model_label='legacy', ablation=None)
            backend = CapturingBackend(False)
            run_generation(cfg, legacy, root, lambda _: backend)
            self.assertIn('SUBTITLE_SENTINEL', backend.requests[0].prompt)

    def test_explicit_native_window_matches_timeline_and_clips_to_media(self):
        from src.experiments.generation_stage import native_audio_clip
        from src.experiments.config import ConfigError
        cfg = {'generation': {'native_audio_window_seconds': 30.}}
        for center, expected in [(40., (25., 55.)), (2., (0., 17.)), (99., (84., 100.))]:
            with self.subTest(center=center):
                clip = native_audio_clip(cfg, 'audio.wav', center, 100., 2.)
                self.assertEqual((clip['start'], clip['end']), expected)
        self.assertIsNone(native_audio_clip(cfg, None, 40., 100., 2.))
        legacy = native_audio_clip({}, 'audio.wav', 40., 100., 2.)
        self.assertEqual((legacy['start'], legacy['end']), (38., 42.))
        for invalid in [0., -1., float('nan'), float('inf')]:
            with self.subTest(invalid=invalid), self.assertRaises(ConfigError):
                native_audio_clip({'generation': {'native_audio_window_seconds': invalid}},
                                  'audio.wav', 40., 100., 2.)

    def test_native_window_changes_run_identity(self):
        from src.experiments.matrix import _gen_identity
        self.assertNotIn('native_audio_window_seconds', _gen_identity({}))
        self.assertNotEqual(_gen_identity({'native_audio_window_seconds': 4.}),
                            _gen_identity({'native_audio_window_seconds': 30.}))
