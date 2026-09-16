from __future__ import annotations
import json,os,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from breath_cleaner import labeler
from breath_cleaner.neural import load_cnn

class ModelDeliveryTests(unittest.TestCase):
    def test_safe_personal_default_metadata(self):
        model=json.loads((ROOT/'models/kabum_v2.json').read_text(encoding='utf-8'))
        self.assertEqual(model['recommended_threshold'],.85)
        self.assertEqual(model['calibration_version'],'speech-safe-20260915')

    def test_personal_cnn_candidate_checkpoint_integrity(self):
        path=ROOT/'models/breath_cnn_personal_v3.json';model=json.loads(path.read_text());model['_path']=str(path)
        network=load_cnn(model)
        self.assertFalse(model['accepted'])
        self.assertEqual(network.training,False)

    def test_model_page_loads_default_recommended_threshold(self):
        self.assertIn('const defaultModel = state.models.find',labeler.MODEL_HTML)
        self.assertIn("thresholdInput.value = String(defaultModel.recommended_threshold)",labeler.MODEL_HTML)
        self.assertIn('Akıllı Hafif (-8…-16 dB)',labeler.MODEL_HTML)
        self.assertIn('“Akıllı Hafif” ve 0,85 eşik varsayılandır',labeler.MODEL_HTML)
        self.assertNotIn('“Hafif azalt” ve %75 eşik varsayılandır',labeler.MODEL_HTML)

if __name__=='__main__':unittest.main()
