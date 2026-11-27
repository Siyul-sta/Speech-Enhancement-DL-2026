import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

submission=module("submit_v5","tools/submit_v5.py")
oracle=module("oracle_mask_audit","tools/oracle_mask_audit.py")
training=module("training_v5","src/kaggle_v4_external.py")

class V5Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
    def test_oracle(self):
        oracle.self_test()
    def test_relaxed_and_dual_axis(self):
        x=torch.randn(1,48000)*0.03
        for cls in (training.RelaxedPhaseUNet,training.DualAxisEnhancer):
            m=cls()
            y=m(x)
            self.assertEqual(y.shape,x.shape)
            self.assertTrue(torch.isfinite(y).all())
            y.square().mean().backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_critic_gradient(self):
        c=training.PESQCritic().eval().requires_grad_(False)
        y=torch.randn(1,48000,requires_grad=True)
        x=torch.randn_like(y)
        loss=(1-c(y,x)).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(y.grad).all())
    def test_fixed_ensemble_and_io(self):
        class Scale(torch.nn.Module):
            def __init__(self,scale): super().__init__();self.scale=scale
            def forward(self,x): return x*self.scale
        models=[(.25,Scale(1)),(.75,Scale(.5))]
        x=np.full(48000,.25,np.float32)
        np.testing.assert_allclose(submission.predict(models,x,torch.device("cpu")),x*.625)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp=root/"input";inp.mkdir();out=root/"output"
            sf.write(inp/"test.WAV",x,16000,subtype="PCM_16")
            submission.run_directory(models,inp,out,torch.device("cpu"))
            y,sr=sf.read(out/"test.WAV")
            self.assertEqual((sr,y.shape),(16000,(48000,)))
            with self.assertRaises(ValueError): submission.run_directory(models,inp,out,torch.device("cpu"))

if __name__=="__main__": unittest.main()
