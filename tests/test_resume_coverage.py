"""Regression tests for selection input identity and fixed evaluation coverage."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_experiment_generation_inputs as fixtures
from src.experiments import generation_stage as stage, io, matrix, report
from src.experiments.selection_stage import run_selection
from src.selection.registry import SelectionContext


class SelectionResumeTests(unittest.TestCase):
    def test_changed_budget_recomputes_and_identical_inputs_resume(self):
        cfg = {'matrix': {'selectors': ['U']}, 'k_rule': {'rate_per_minute': 1}}
        run = matrix.core_selections(cfg)[0]
        record = SimpleNamespace(video_id='v')
        context = SelectionContext('v', 60)
        with tempfile.TemporaryDirectory() as d, patch(
                'src.experiments.selection_stage.load_context', return_value=(context, {})):
            path = run_selection(cfg, run, [record], Path(d))
            cfg['k_rule']['rate_per_minute'] = 12
            run_selection(cfg, run, [record], Path(d))
            rows = io.read_jsonl(path)
            self.assertEqual([r['k'] for r in rows], [1, 12])
            self.assertEqual(len(rows[-1]['times']), 12)
            run_selection(cfg, run, [record], Path(d))
            self.assertEqual(io.read_jsonl(path), rows)

    def test_legacy_rows_and_same_times_changed_budget_are_not_settled(self):
        cfg = {'matrix': {'selectors': ['O']}, 'k_rule': {'rate_per_minute': 1}}
        run = matrix.core_selections(cfg)[0]
        context = SelectionContext('v', 60, ground_truth=[10])
        with tempfile.TemporaryDirectory() as d, patch(
                'src.experiments.selection_stage.load_context', return_value=(context, {})):
            path = run_selection(cfg, run, [SimpleNamespace(video_id='v')], Path(d))
            old = io.read_jsonl(path)[0]; old.pop('input_fingerprint')
            path.write_text(__import__('json').dumps(old) + '\n')
            run_selection(cfg, run, [SimpleNamespace(video_id='v')], Path(d))
            self.assertEqual(len(io.read_jsonl(path)), 2)
            cfg['k_rule']['rate_per_minute'] = 12
            run_selection(cfg, run, [SimpleNamespace(video_id='v')], Path(d))
            self.assertEqual(io.read_jsonl(path)[-1]['k'], 12)
            self.assertEqual(io.read_jsonl(path)[-1]['times'], [10])
            context.ground_truth = [20]
            run_selection(cfg, run, [SimpleNamespace(video_id='v')], Path(d))
            self.assertEqual(io.read_jsonl(path)[-1]['times'], [20])


class GenerationCoverageTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.GenerationInputTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.records = [self.f.video, self.f.make_video('second')]
        for r in self.records:
            self.f.select(r)
        stage.build_moments(self.f.cfg, self.f.run.selection, self.records, self.f.exp)

    def evaluate(self):
        return stage.evaluate_generation(self.f.cfg, self.f.run, self.records, self.f.exp, scorers={})

    def test_partial_smoke_keeps_all_videos_and_reports_missing(self):
        f = self.f
        factory, backend = f.factory()
        stage.run_generation(f.cfg, f.run, f.exp, factory, self.records, limit=1)
        r = self.evaluate()
        self.assertEqual((r['n_videos'], r['n_videos_observed'], r['n_missing']), (2, 1, 1))
        self.assertEqual(r['macro_f1']['mean'], 0.5)
        self.assertEqual(r['status'], 'incomplete')
        report.generation_table([r], f.exp / 'tables')
        self.assertIn('incomplete', (f.exp / 'tables/generation_matrix.md').read_text())
        stage.run_generation(f.cfg, f.run, f.exp, factory, self.records)
        r = self.evaluate()
        self.assertEqual((r['status'], r['n_missing'], r['macro_f1']['mean']), ('complete', 0, 1.0))

    def test_empty_outputs_are_incomplete_with_fixed_denominator(self):
        r = self.evaluate()
        self.assertEqual((r['n_videos'], r['n_missing'], r['macro_f1']['mean']), (2, 2, 0.0))
        self.assertEqual(r['status'], 'incomplete')

    def test_unprepared_video_is_visible(self):
        f = self.f
        self.records.append(f.make_video('unprepared'))
        factory, _ = f.factory()
        stage.run_generation(f.cfg, f.run, f.exp, factory, self.records)
        r = self.evaluate()
        self.assertEqual(r['videos_without_prepared_moments'], ['unprepared'])
        self.assertEqual(r['n_videos'], 3)
        self.assertEqual(r['status'], 'incomplete')

    def test_recorded_failures_are_distinct_from_missing(self):
        f = self.f
        factory, _ = f.factory()
        stage.run_generation(f.cfg, f.run, f.exp, factory, self.records)
        path = stage.generations_path(f.exp, f.run)
        rows = io.read_jsonl(path)
        rows[0]['error'] = 'injected generation failure'
        path.unlink()
        io.append_jsonl(path, rows)
        r = self.evaluate()
        self.assertEqual((r['status'], r['n_error'], r['n_missing']), ('complete_with_errors', 1, 0))
        self.assertEqual(r['macro_f1']['mean'], 0.5)
