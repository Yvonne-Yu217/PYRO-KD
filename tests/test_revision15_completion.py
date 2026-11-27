import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.verify_revision15_completion import coverage, exact, read_rows, Audit, audit_qa, check_bound_sources, check_family
from types import SimpleNamespace

class CompletionTests(unittest.TestCase):
    def test_failures_are_attempts_and_retries_are_enumerated(self):
        info=coverage([{'id':'a','error':'oom'}, {'id':'a','output':'ok'}, {'id':'b','hit_token_limit':True}],{'a','b'})
        self.assertTrue(exact(info));self.assertEqual(info['failed_ids'],['b'])
        self.assertEqual(info['superseded_error_attempts'],1)
    def test_missing_unexpected_duplicate_fail(self):
        for rows in ([{'id':'a'}], [{'id':'a'},{'id':'b'},{'id':'x'}], [{'id':'a'},{'id':'a'},{'id':'b'}]):
            self.assertFalse(exact(coverage(rows,{'a','b'})))
    def test_truncated_and_malformed_jsonl_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'rows.jsonl'
            for data in ('{"id":"a"}', '{bad}\n'):
                p.write_text(data)
                with self.assertRaises(ValueError):read_rows(p)
    def test_missing_and_stale_selection_or_contrast_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'selection.jsonl';p.write_text('{}\n')
            from scripts.verify_revision15_completion import sha
            digest=sha(p)
            with self.assertRaisesRegex(ValueError,'absent'):
                check_bound_sources(Audit(),{},[p])
            check_bound_sources(Audit(),{str(p):digest},[p])
            p.write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError,'stale'):
                check_bound_sources(Audit(),{str(p):digest},[p])
    def test_missing_or_wrong_contrast_family_refused(self):
        good=dict(condition_a='a',condition_b='b',excluded=[],units=[{'unit_id':str(x)} for x in range(10)],included_example_ids=[str(x) for x in range(30)],wilcoxon={'holm_pvalue':1.0})
        check_family(Audit(),[good],[('a','b')])
        with self.assertRaisesRegex(ValueError,'size'):
            check_family(Audit(),[],[('a','b')])
        with self.assertRaisesRegex(ValueError,'frozen plan'):
            check_family(Audit(),[good],[('b','a')])
        with self.assertRaisesRegex(ValueError,'cluster'):
            check_family(Audit(),[{**good,'units':good['units'][:-1]}],[('a','b')])
    def test_incomplete_qa_never_opens_scoring_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'configs').mkdir()
            (root/'configs/egoschema_pilot.json').write_text(json.dumps({'heldout_q_uids':[str(x) for x in range(490)]}))
            a=Audit();opened=[];original=a.obj
            def obj(p):
                opened.append(str(p));return original(p)
            with patch.object(a,'obj',side_effect=obj):
                audit_qa(a,SimpleNamespace(formal_repo=root,results_root=root),'egoschema')
            self.assertTrue(a.errors)
            self.assertFalse(any('/evaluation/' in x for x in opened))

if __name__=='__main__':unittest.main()
