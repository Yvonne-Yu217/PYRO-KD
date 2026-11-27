"""Real Parquet/CLI integration using temporary synthetic data only.

Run with the data environment's Python; unittest avoids adding pytest there.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'requires data environment')
class VideoMMECLI(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.labels = self.root/'synthetic.parquet'
        rows = [{'question_id':q, 'video_id':v, 'answer':'B'}
                for q,v in [('q1','a'),('q2','a'),('q3','b')]]
        # Filtering must exclude this malformed answer outside membership.
        pq.write_table(pa.Table.from_pylist(rows + [
            {'question_id':'excluded','video_id':'excluded','answer':'INVALID'}]), self.labels)
        self.subset = self.root/'subset.json'
        self.subset.write_text(json.dumps(dict(dataset='synthetic',revision='fixture',official_split='test',
            annotations_sha256=sha(self.labels),videos=[{'video_id':'a','domain':'sports'},
                                                       {'video_id':'b','domain':'art'}])))
        self.membership = self.root/'membership.json'
        self.membership.write_text(json.dumps(dict(dataset='synthetic',revision='fixture',official_split='test',
            annotations_sha256=sha(self.labels),subset_sha256=sha(self.subset),expected_questions=3,
            questions=[{k:r[k] for k in ('question_id','video_id')} for r in rows])))
        self.predictions = [dict(id=r['question_id'],video_id=r['video_id'],dataset='video-mme',
            split='long-heldout',selector='U',selection_condition='U',audio_context='none',
            run_fingerprint='synthetic-fixture-only',output='B',predicted_index=1,
            source_sha256={'question_membership':sha(self.membership),'subset':sha(self.subset)}) for r in rows]
        self.output = self.root/'score.json'

    def run_cli(self, predictions):
        inputs = self.root/'synthetic-predictions.jsonl'
        inputs.write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        return subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_video_mme_qa.py'),
            '--input',str(inputs),'--annotations',str(self.labels),'--membership',str(self.membership),
            '--subset',str(self.subset),'--output',str(self.output)],
            cwd=ROOT,text=True,capture_output=True,timeout=30)

    def test_real_parquet_filter_and_complete_denominator(self):
        self.predictions[-1].update(output='A',predicted_index=0)
        run = self.run_cli(self.predictions)
        self.assertEqual(run.returncode,0,run.stderr)
        report = json.loads(self.output.read_text())
        self.assertEqual(report['accuracy'],2/3)
        self.assertTrue(report['complete'])
        self.assertEqual(report['annotations_sha256'],sha(self.labels))
        self.assertEqual(report['by_domain']['art']['accuracy'],0)

    def test_partial_predictions_exit_nonzero_and_keep_missing_denominator(self):
        run = self.run_cli(self.predictions[:1])
        self.assertEqual(run.returncode,1,run.stderr)
        report = json.loads(self.output.read_text())
        self.assertEqual(report['accuracy'],1/3)
        self.assertEqual(report['status_counts']['missing'],2)

    def test_source_mismatch_rejected_before_answer_file_access(self):
        self.predictions[0]['source_sha256']['question_membership'] = 'wrong'
        self.labels.unlink()
        run = self.run_cli(self.predictions)
        self.assertNotEqual(run.returncode,0)
        self.assertIn('prediction dataset/split/source mismatch',run.stderr)
        self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
