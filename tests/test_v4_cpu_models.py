import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tools")]
import kaggle_v4_external as trainer
import infer_v4

class CpuModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
    def test_native_model_forward_backward(self):
        for cls in (trainer.ResidualPhaseUNet,trainer.TemporalEnhancer,trainer.AdaptiveMixture):
            with self.subTest(model=cls.__name__):
                model=cls()
                x=torch.randn(1,4096)*.05
                y=model(x)
                self.assertEqual(y.shape,x.shape)
                self.assertTrue(torch.isfinite(y).all())
                y.square().mean().backward()
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
    def test_synthetic_validation_repeatable(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            c=path/"clean.wav"; n=path/"noise.wav"
            sf.write(c,np.sin(np.arange(48000)*.07).astype(np.float32)*.1,16000)
            sf.write(n,np.random.default_rng(1).normal(0,.02,48000).astype(np.float32),16000)
            data=trainer.SyntheticPairs([c],[n],2,42,False)
            x,y=data[0]; x2,y2=data[0]
            self.assertEqual(tuple(x.shape),(48000,))
            self.assertTrue(torch.equal(x,x2) and torch.equal(y,y2))
            self.assertTrue(torch.isfinite(x).all())
            self.assertLessEqual(float(x.abs().max()),1)
            self.assertEqual(trainer.quality_paths([c,c,n]),[c,n])
    def test_inference_lengths(self):
        for length in (1,24001,48000,48001,100003):
            x=np.random.default_rng(42).normal(0,.1,length).astype(np.float32)
            np.testing.assert_allclose(infer_v4.enhance(torch.nn.Identity(),x,"cpu"),x,atol=1e-7)
        self.assertEqual(len(infer_v4.resample(np.zeros(44100,dtype=np.float32),44100,16000)),16000)
    def test_wer(self):
        self.assertEqual(trainer.word_error_count("one two three","one four three"),(1,3))
        self.assertEqual(trainer.word_error_count("one two","one"),(1,2))

if __name__=="__main__":
    unittest.main()
