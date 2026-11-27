from types import SimpleNamespace
import unittest
import numpy as np
from src.utils.speech_features import speech_windows, transcribe_gated


class SpeechTests(unittest.TestCase):
    def test_windows_never_bridge_silence_and_split_long_regions(self):
        self.assertEqual(list(speech_windows([(0,2),(4,9)],100,10,3)),[(0,20),(40,70),(70,90)])
        self.assertEqual(list(speech_windows([(1,3),(2,4)],50,10)),[(10,40)])
        self.assertEqual(list(speech_windows([(0.11,0.29)],10,10)),[])
        for spans in ([(-1,2)],[(1,11)],[(2,1)],[(0,float('nan'))]):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                list(speech_windows(spans,100,10))

    def test_only_gated_samples_reach_model_and_times_are_restored(self):
        calls=[]
        class Model:
            def transcribe(self,wave,**kwargs):
                calls.append((wave.tolist(),kwargs))
                return iter([SimpleNamespace(words=[SimpleNamespace(start=0.0,end=0.5,word='hi',probability=0.8)])]),None
        result=transcribe_gated(np.arange(100),[(1,2),(5,7)],Model(),sample_rate=10)
        self.assertEqual([c[0] for c in calls],[list(range(10,20)),list(range(50,70))])
        self.assertTrue(all(c[1]['word_timestamps'] and not c[1]['condition_on_previous_text'] for c in calls))
        self.assertEqual([(w['start'],w['end']) for w in result['words']],[(1,1.5),(5,5.5)])
        self.assertEqual(result['speech_samples'],30)

    def test_outside_window_is_rejected_without_clamping(self):
        class Model:
            def transcribe(self,wave,**kwargs):
                words=[SimpleNamespace(start=0,end=2,word='outside',probability=0.5),
                       SimpleNamespace(start=0.2,end=0.8,word='inside',probability=0.7)]
                return [SimpleNamespace(words=words)],None
        result=transcribe_gated(np.zeros(100),[(2,3)],Model(),sample_rate=10)
        self.assertEqual([w['text'] for w in result['words']],['inside'])
        self.assertEqual(result['rejected_words'][0]['end'],4)
        self.assertEqual(result['rejected_words'][0]['gate'],[2,3])

    def test_no_speech_does_not_call_model(self):
        class Model:
            def transcribe(self,*args,**kwargs): raise AssertionError('must not call ASR')
        result=transcribe_gated(np.zeros(100),[],Model(),sample_rate=10)
        self.assertEqual(result,{'words':[],'rejected_words':[],'windows':[],'speech_samples':0})


class MultilingualSpeechTests(unittest.TestCase):
    def test_per_window_detection_retains_language_and_never_bridges_gaps(self):
        calls=[]
        class Model:
            def transcribe(self,wave,**kwargs):
                calls.append((len(wave),kwargs['language']))
                return [],SimpleNamespace(language='es' if len(calls)==1 else 'fr',language_probability=.8)
        result=transcribe_gated(np.zeros(100),[(1,2),(5,7)],Model(),sample_rate=10,language=None)
        self.assertEqual(calls,[(10,None),(20,None)])
        self.assertEqual([w['detected_language'] for w in result['windows']],['es','fr'])
        self.assertTrue(all(w['language_probability']==.8 for w in result['windows']))

    def test_forced_language_passes_through_without_fake_detection_metadata(self):
        class Model:
            def transcribe(self,wave,**kwargs):
                assert kwargs['language']=='zh'
                return [],None
        result=transcribe_gated(np.zeros(100),[(1,2)],Model(),sample_rate=10,language='zh')
        self.assertNotIn('detected_language',result['windows'][0])

    def test_invalid_language_metadata_fails(self):
        class Model:
            def transcribe(self,*args,**kwargs):
                return [],SimpleNamespace(language='es',language_probability=float('nan'))
        with self.assertRaises(ValueError):
            transcribe_gated(np.zeros(100),[(1,2)],Model(),sample_rate=10,language=None)
