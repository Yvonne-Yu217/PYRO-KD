import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from src.experiments.io import experiment_dir, append_jsonl, read_jsonl
from src.experiments.matrix import core_selections
from src.experiments.selection_stage import evaluate_selections
from scripts.run_experiment import _backend_factory

class ResumeReadinessTests(TestCase):
    def test_changed_configuration_refuses_stale_directory(self):
        with TemporaryDirectory() as d:
            cfg = {'paths': {'output_root': d}, 'experiment': {'name': 'same'},
                   'selection': {'num_frames': 4}}
            experiment_dir(cfg)
            experiment_dir(cfg)
            changed = copy.deepcopy(cfg)
            changed['selection']['num_frames'] = 8
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                experiment_dir(changed)

    def test_tail_recovery_before_append_preserves_records_and_backup(self):
        with TemporaryDirectory() as d:
            p = Path(d)/'out.jsonl'
            original = b'{"id": "first"}\n{"id": "broken'
            p.write_bytes(original)
            append_jsonl(p, [{'id': 'second'}])
            self.assertEqual(read_jsonl(p), [{'id': 'first'}, {'id': 'second'}])
            self.assertEqual(next(Path(d).glob('*.bin')).read_bytes(), original)
            p.write_text('{"id": "first"}\nbroken\n')
            with self.assertRaises(ValueError): read_jsonl(p)

    def test_explicit_api_ledger_is_absolute(self):
        with patch('src.generation.backends.build_backend') as build:
            _backend_factory(False)({'backend': 'openai', 'ledger_path': 'ledger.json'})
            self.assertTrue(Path(build.call_args.args[0]['ledger_path']).is_absolute())

    def test_missing_video_is_in_evaluation_coverage(self):
        from types import SimpleNamespace
        cfg = {'matrix': {'selectors': ['U']}, 'evaluation': {'resamples': 10}}
        with TemporaryDirectory() as d:
            runs = core_selections(cfg)
            report = evaluate_selections(cfg, runs, [SimpleNamespace(video_id='absent', ad=[])], Path(d))
            row = report['runs']['U']
            self.assertEqual(row['missing_video_ids'], ['absent'])
            self.assertEqual(row['not_run']['missing'], 1)
            self.assertFalse(row['complete'])
