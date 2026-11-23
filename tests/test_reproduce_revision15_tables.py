import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.reproduce_revision15_tables import completion_gate, main, require_bound, sha

class ReproductionGuardTests(unittest.TestCase):
    def test_false_manifest_does_not_open_artifacts_or_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';out=Path(tmp)/'out'
            p.write_text(json.dumps({'local_experiments_complete':False,'artifact_sha256':{'never-read-qa':'bad'}}))
            with patch('scripts.reproduce_revision15_tables.sha',side_effect=AssertionError('artifact read before gate')):
                with self.assertRaisesRegex(ValueError,'not ready'):main(['--completion-manifest',str(p),'--output-root',str(out)])
            self.assertFalse(out.exists())
    def test_changed_artifact_refused_before_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);artifact=root/'metadata.json';artifact.write_text('{}')
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'local_experiments_complete':True,'errors':[],'artifact_sha256':{str(artifact):sha(artifact)}}))
            artifact.write_text('{"changed":true}')
            with patch('scripts.reproduce_revision15_tables.build_commands',side_effect=AssertionError('commands before guard')):
                with self.assertRaisesRegex(ValueError,'changed'):main(['--completion-manifest',str(manifest),'--output-root',str(root/'out')])
            self.assertFalse((root/'out').exists())
    def test_existing_output_and_unbound_input_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(ValueError,'already exist'):main(['--completion-manifest','unopened','--output-root',str(root)])
            with self.assertRaisesRegex(ValueError,'absent'):require_bound(root/'not-read',{'artifact_sha256':{}})
    def test_unchanged_metadata_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'metadata';p.write_text('metadata only')
            m=Path(tmp)/'manifest';m.write_text(json.dumps({'local_experiments_complete':True,'errors':[],'artifact_sha256':{str(p):sha(p)}}))
            self.assertTrue(completion_gate(m)['local_experiments_complete'])

if __name__=='__main__':unittest.main()
