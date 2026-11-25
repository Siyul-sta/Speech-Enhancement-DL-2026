"""CPU-only public model compatibility smoke; uses synthetic audio, never private data."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import kaggle_v4_external as trainer
import kaggle_v4_public as public

class PublicModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
    def test_metricgan_transfer_update(self):
        with tempfile.TemporaryDirectory() as d,patch.object(trainer,"WORK",Path(d)):
            model=trainer.MetricGANTransfer().train()
            t=torch.arange(16000,dtype=torch.float32)/16000
            clean=(.1*torch.sin(2*torch.pi*220*t))[None]
            noisy=clean+torch.randn_like(clean)*.02
            before=[p.detach().clone() for p in model.generator.parameters()]
            opt=torch.optim.AdamW(model.generator.parameters(),lr=1e-4)
            pred=model(noisy)
            self.assertEqual(pred.shape,noisy.shape)
            self.assertTrue(torch.isfinite(pred).all())
            (pred-clean).abs().mean().backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.generator.parameters()))
            opt.step()
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.generator.parameters())))
    def test_metricgan_public_inference(self):
        with tempfile.TemporaryDirectory() as d,patch.object(public,"WORK",Path(d)):
            model=public.load_model("metricgan",torch.device("cpu"))
            x=torch.randn(1,16000)*.05
            with torch.inference_mode():
                out=public.checked_output(model,x)
            self.assertEqual(out.shape,x.shape)
            self.assertTrue(torch.isfinite(out).all())


    def test_sepformer_dns_and_wham_inference(self):
        for name in ("sepformer_dns","sepformer_wham"):
            with self.subTest(name=name):
                fn=public.load_model(name,torch.device("cpu"))
                x=torch.randn(1,4096)*.05
                with torch.inference_mode():
                    y=public.checked_output(fn,x)
                self.assertEqual(y.shape,x.shape)
    def test_sepformer_decoder_update(self):
        with tempfile.TemporaryDirectory() as d,patch.object(trainer,"WORK",Path(d)),patch.dict(trainer.EXPERIMENT_CONFIG,{"architecture":"sepformer_dns_transfer"}):
            model=trainer.SepformerTransfer().train()
            x=torch.randn(1,4096)*.05
            before=[p.detach().clone() for p in model.decoder.parameters()]
            opt=torch.optim.AdamW(model.decoder.parameters(),lr=1e-4)
            y=model(x)
            self.assertEqual(y.shape,x.shape)
            (y-x).abs().mean().backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.decoder.parameters()))
            opt.step()
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.decoder.parameters())))
            self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
    def test_deepfilter_inference(self):
        fn=public.load_model("deepfilter",torch.device("cpu"))
        x=torch.randn(1,16000)*.05
        with torch.inference_mode():
            y=public.checked_output(fn,x)
        self.assertEqual(y.shape,x.shape)

if __name__=="__main__":
    unittest.main()
