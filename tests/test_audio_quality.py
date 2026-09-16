from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from breath_cleaner.audio_io import WavAudio,read_audio_native,write_wav
from breath_cleaner.detect import BreathSegment
from breath_cleaner.labeler import _cut_segments
from breath_cleaner.process import duck_segments

def segment(start,end):
    return BreathSegment(start,end,.9,-30,.1,1500,3500)

class AudioQualityTests(unittest.TestCase):
    def test_stereo_duck_preserves_shape_and_unselected_samples(self):
        audio=np.column_stack([np.ones(4800,np.float32)*.2,np.ones(4800,np.float32)*-.1])
        result=duck_segments(audio,48000,[segment(.02,.06)],gain_db=-12,fade_ms=5)
        self.assertEqual(result.shape,audio.shape)
        np.testing.assert_array_equal(result[:800],audio[:800])
        np.testing.assert_array_equal(result[3200:],audio[3200:])
        self.assertLess(abs(float(result[2000,0])),abs(float(audio[2000,0])))
        self.assertGreater(float(result[2000,0]),0)
        self.assertLess(float(result[2000,1]),0)

    def test_uncertain_detection_is_attenuated_less_than_confident_detection(self):
        audio=np.ones(8000,dtype=np.float32)*.2
        uncertain=duck_segments(audio,16000,[segment(.1,.4)],gain_db=-16,fade_ms=10,
                                low_confidence_gain_db=-8,confidence_floor=.75)
        certain_segment=segment(.1,.4)
        certain_segment=type(certain_segment)(certain_segment.start,certain_segment.end,1.0,
            certain_segment.rms_db,certain_segment.zcr,certain_segment.centroid_hz,certain_segment.rolloff_hz)
        certain=duck_segments(audio,16000,[certain_segment],gain_db=-16,fade_ms=10,
                              low_confidence_gain_db=-8,confidence_floor=.75)
        self.assertGreater(abs(float(uncertain[3000])),abs(float(certain[3000])))
        np.testing.assert_array_equal(uncertain[:1000],audio[:1000])

    def test_stereo_cut_crossfade_retains_channels(self):
        audio=np.column_stack([np.linspace(-.5,.5,4800,dtype=np.float32),np.linspace(.4,-.4,4800,dtype=np.float32)])
        result=_cut_segments(audio,48000,[segment(.03,.05)],crossfade_ms=5)
        self.assertEqual(result.ndim,2)
        self.assertEqual(result.shape[1],2)
        self.assertLess(result.shape[0],audio.shape[0])
        self.assertTrue(np.isfinite(result).all())

    def test_wav_roundtrip_preserves_48khz_stereo(self):
        audio=np.column_stack([np.linspace(-.8,.8,1000,dtype=np.float32),np.linspace(.7,-.7,1000,dtype=np.float32)])
        with tempfile.TemporaryDirectory() as temp_dir:
            path=Path(temp_dir)/'audio-quality-test.wav'
            write_wav(path,WavAudio(48000,audio))
            decoded=read_audio_native(path)
            self.assertEqual(decoded.sample_rate,48000)
            self.assertEqual(decoded.samples.shape,(1000,2))
            np.testing.assert_allclose(decoded.samples,audio,atol=4e-5)

if __name__=='__main__':unittest.main()
