import unittest
import numpy as np
from src.utils.event_features import centered_window,event_times,event_cues


class EventTests(unittest.TestCase):
    def test_grid_and_edges_preserve_audio_timing(self):
        np.testing.assert_array_equal(event_times(25,10),[0,1,2])
        wave=np.arange(6,dtype=np.float32)
        np.testing.assert_array_equal(centered_window(wave,2,0,seconds=2),[0,0,0,1])
        np.testing.assert_array_equal(centered_window(wave,2,2.5,seconds=2),[3,4,5,0])
        np.testing.assert_array_equal(centered_window(wave,2,1.5,seconds=2),[1,2,3,4])
        with self.assertRaises(ValueError): event_times(0,10)

    def test_speech_silence_suppressed_but_music_and_laughter_preserved(self):
        p=np.array([[0,0,0,0],[1,1,0.8,0.4],[0.9,0.9,0.2,0.2]])
        e=np.array([[2,0],[2,0],[0,7]])
        result=event_cues(p,e,[0.1,1,0.1],['Speech','Silence','Music','Laughter'])
        np.testing.assert_allclose(result['class_onsets'][1],[0,0,0.8,0.4])
        np.testing.assert_allclose(result['class_onsets'][0],0)
        np.testing.assert_allclose(result['class_onsets'][2],0)
        np.testing.assert_allclose(result['clap_change'],[0,0,1])
        np.testing.assert_allclose(np.linalg.norm(result['clap_embeddings'],axis=1),1)
        np.testing.assert_allclose(result['loudness_rise_db'],[0,20,0])

    def test_memory_uses_only_previous_embeddings(self):
        result=event_cues(np.zeros((3,1)),[[1,0],[0,1],[-1,0]],[1,1,1],['Music'],memory=1)
        np.testing.assert_allclose(result['clap_change'],[0,1,1])
        with self.assertRaises(ValueError): event_cues([[0]],[[0,0]],[1],['Music'])
        with self.assertRaises(ValueError): event_cues([[float('nan')]],[[1,0]],[1],['Music'])
