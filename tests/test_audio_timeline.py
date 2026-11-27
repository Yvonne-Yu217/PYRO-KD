from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest
from PIL import Image
from src.generation.audio_timeline import contextual_timeline,render_timeline,timeline_hash,check_audio_resume
from src.generation.prompts import build_prompt
from scripts.run_ad_vlm import prepare


class TimelineTests(unittest.TestCase):
    def test_context_uses_only_contained_words_and_near_events(self):
        words=[{'start':0,'end':1,'text':' old'}, {'start':19,'end':20,'text':' hello'},
               {'start':20,'end':21,'text':' world'}, {'start':35,'end':36,'text':' beyond'}]
        events=[{'start':22,'end':22,'kind':'event','text':'Music'}, {'start':60,'end':60,'kind':'event','text':'Clap'}]
        result,omitted=contextual_timeline(words,events,20,window=15)
        self.assertEqual([r['text'] for r in result],['hello world','Music'])
        self.assertEqual(omitted,0)
        self.assertIn('[0:22.0 event: "Music"]',render_timeline(result))

    def test_missing_audio_not_treated_as_detected_empty(self):
        with self.assertRaises(ValueError): render_timeline(None)
        self.assertIn('no speech or selected events detected',render_timeline([]))
        with self.assertRaises(ValueError): render_timeline([{'start':-1,'end':1,'kind':'speech','text':'x'}])

    def test_prior_preserves_audio_context_and_disallows_duplicate_transcript(self):
        text=render_timeline([{'start':1,'end':2,'kind':'speech','text':'a "quote"\nand {braces}'}])
        p0=build_prompt('P0',2,4,2,10,audio_timeline_text=text)
        p1=build_prompt('P1',2,4,2,10,exemplars=['a','b','c'],audio_timeline_text=text)
        self.assertTrue(p1.endswith(p0))
        self.assertIn('\\"quote\\"',p0)
        with self.assertRaises(ValueError): build_prompt('P0',2,4,2,10,'legacy',audio_timeline_text=text)

    def test_generator_none_removes_transcript_and_timeline_reads_actual_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'frame.png';Image.new('RGB',(8,8)).save(path)
            row={'id':'x','frames':[str(path)],'time':1,'word_limit':10,'transcript':'OLD ANNOTATION',
                 'audio_timeline':[{'start':0,'end':0.5,'kind':'speech','text':'ASR WORDS'}]}
            args=SimpleNamespace(prior='P0',span=2,max_side=768,audio_context='none')
            _,prompt,_=prepare(row,args)
            self.assertNotIn('OLD ANNOTATION',prompt);self.assertNotIn('ASR WORDS',prompt)
            args.audio_context='timeline';_,prompt,_=prepare(row,args)
            self.assertIn('ASR WORDS',prompt);self.assertNotIn('OLD ANNOTATION',prompt)

    def test_resume_rejects_context_and_timeline_changes(self):
        row={'id':'x','audio_timeline':[]}
        old={'id':'x','audio_context':'timeline','audio_timeline_sha256':timeline_hash([])}
        check_audio_resume([old],[row],'timeline')
        with self.assertRaises(ValueError): check_audio_resume([old],[row],'none')
        row['audio_timeline']=[{'start':1,'end':1,'kind':'event','text':'Music'}]
        with self.assertRaises(ValueError): check_audio_resume([old],[row],'timeline')
