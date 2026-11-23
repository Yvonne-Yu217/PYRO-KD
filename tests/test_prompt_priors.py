import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.generation.exemplars import check_resume_prior, load_exemplars
from src.generation.prompts import build_prompt


class PromptPriorTests(unittest.TestCase):
    def bundle(self):
        return {'schema':'reframed-train-exemplars-v1', 'source_sha256':{'train':'a'*64},
                'excluded_movie_ids':['heldout'],
                'examples':[{'split':'train','track':'us','movie_id':f'm{i}','video_id':f'v{i}',
                             'start':1.0,'end':3.0,'text':f'Example description {i}.'} for i in range(3)]}

    def load(self, bundle):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'examples.json'
            path.write_text(json.dumps(bundle))
            return load_exemplars(path)

    def test_prior_preserves_current_video_controls_and_literal_braces(self):
        p0=build_prompt('P0',75.5,4,2.0,12,'speech')
        p1=build_prompt('P1',75.5,4,2.0,12,'speech', ['A {person} turns.', 'A door opens.', 'A car stops.'])
        self.assertTrue(p1.endswith(p0))
        self.assertIn('A {person} turns.',p1)
        self.assertEqual(p1.count('Example '),3)
        self.assertNotIn('Professional descriptions',p0)

    def test_no_silent_fallback_or_prior_contamination(self):
        for examples in ([],['one'],['one','two',''],['one','two','three','four']):
            with self.subTest(examples=examples), self.assertRaises(ValueError):
                build_prompt('P1',1,4,2,10,exemplars=examples)
        with self.assertRaises(ValueError):
            build_prompt('P0',1,4,2,10,exemplars=['one','two','three'])

    def test_rejects_heldout_or_duplicate_movies_and_missing_provenance(self):
        for change in ({'split':'test'},{'split':'validation'},{'movie_id':'heldout'},
                       {'movie_id':'m1'},{'video_id':''},{'end':float('nan')},{'text':''}):
            bundle=self.bundle(); bundle['examples'][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(bundle)
        bundle=self.bundle(); del bundle['source_sha256']
        with self.assertRaises(ValueError): self.load(bundle)

    def test_content_hash_changes_and_resume_refuses_mixing(self):
        bundle=self.bundle(); texts,digest=self.load(bundle)
        self.assertEqual(len(texts),3)
        altered=copy.deepcopy(bundle); altered['examples'][0]['text']='A different professional description.'
        self.assertNotEqual(digest,self.load(altered)[1])
        row={'prior':'P1','exemplar_sha256':digest}
        check_resume_prior([row],'P1',digest)
        check_resume_prior([{'prior':'P0'}],'P0',None)
        for prior,sha in [('P0',None),('P1','different')]:
            with self.assertRaises(ValueError): check_resume_prior([row],prior,sha)
        with self.assertRaises(ValueError): check_resume_prior([{'prior':'P1'}],'P1',digest)
