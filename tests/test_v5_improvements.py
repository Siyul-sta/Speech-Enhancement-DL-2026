"""V5 offline contracts; no private data or GPU."""
import sys
import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tools")]
import kaggle_v4_external as trainer
import kaggle_v4_public as public
import oracle_mask_audit as oracle
import ralph_v4

class V5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_oracle(self):
        oracle.self_test()

    def test_new_models_forward_backward(self):
        for cls in (trainer.RelaxedPhaseUNet,trainer.DualAxisEnhancer):
            with self.subTest(model=cls.__name__):
                model=cls()
                x=torch.randn(1,4096)*.05
                out=model(x)
                self.assertEqual(x.shape,out.shape)
                self.assertTrue(torch.isfinite(out).all())
                (out-x*.6).square().mean().backward()
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertTrue(any(g.abs().sum()>0 for g in grads))

    def test_remix_deterministic_and_train_only(self):
        rng=np.random.default_rng(42)
        train=trainer.SpeechPairs.__new__(trainer.SpeechPairs)
        train.split="train"
        train.clean=rng.normal(0,.1,(1296,1024)).astype(np.float32)
        train.noisy=train.clean+rng.normal(0,.03,train.clean.shape).astype(np.float32)
        data=trainer.ChallengeRemix(train,16,42)
        x,c=data[0]
        self.assertEqual(x.shape,c.shape)
        self.assertTrue(torch.equal(x,data[0][0]))
        trainer.set_data_epoch(data,2)
        self.assertFalse(torch.equal(x,data[0][0]))
        self.assertTrue(torch.isfinite(x).all())
        self.assertLessEqual(float(x.abs().max()),1.)
        train.split="validation"
        with self.assertRaises(ValueError): trainer.ChallengeRemix(train,16,42)

    def test_pesq_critic_and_generator_update(self):
        model=trainer.TemporalEnhancer()
        critic=trainer.PESQCritic()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=1e-3)
        x=torch.randn(2,4096)*.05
        clean=x*.6
        # Fake scalar metric verifies gradient plumbing, not PESQ quality.
        def mock_pesq(sr,c,y,mode): return float(4.5-np.mean((c-y)**2))
        before=model.out.weight.detach().clone()
        before_critic=critic.net[-2].weight.detach().clone()
        loss=trainer.supervised_step(model,optimizer,x,clean,{"id":"V3_RESIDUAL_PHASE"},
            critic=critic,critic_optimizer=critic_optimizer,pesq_fn=mock_pesq,
            metric_weight=.05,update_critic=True)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before,model.out.weight))
        self.assertFalse(torch.equal(before_critic,critic.net[-2].weight))
        self.assertTrue(all(not p.requires_grad for p in critic.parameters()))

    def test_triple_blend_and_packaging(self):
        x=torch.randn(1,100)*.05
        scales={"cmgan":.2,"metricgan":.6,"deepfilter":1.}
        with patch.object(public,"load_model",lambda n,d:lambda x:x*scales[n]):
            torch.testing.assert_close(public.public_recipe("mean3:cmgan:metricgan:deepfilter","cpu")(x),x*.6)
        combined=ralph_v4.public_with_external((ROOT/"src/kaggle_v4_public.py").read_text(),(ROOT/"src/kaggle_v4_external.py").read_text())
        compile(combined,"combined.py","exec")
        self.assertIn("class CMGANTransfer",combined)
        self.assertIn("def cmgan_assets",combined)

    @unittest.skipUnless(os.environ.get("V5_TEST_PUBLIC_CMGAN")=="1","opt-in public network integration")
    def test_actual_cmgan(self):
        with tempfile.TemporaryDirectory() as d,patch.object(trainer,"WORK",Path(d)):
            model=trainer.CMGANTransfer().train()
            t=torch.arange(4096)/16000
            x=(.1*torch.sin(2*torch.pi*220*t)+.01*torch.randn_like(t))[None]
            y=model(x)
            self.assertEqual(x.shape,y.shape)
            self.assertTrue(torch.isfinite(y).all())
            (y-x*.7).abs().mean().backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.net.mask_decoder.parameters()))
            self.assertTrue(all(p.grad is None for p in model.net.dense_encoder.parameters()))

if __name__=="__main__":
    unittest.main()
