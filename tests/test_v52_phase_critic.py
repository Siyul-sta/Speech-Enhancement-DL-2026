"""V5.2 CPU contracts. Mock PESQ is gradient plumbing, not actual quality evidence."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tools")]
import kaggle_v4_external as t
import ralph_v4 as v4
import submit_v5

class V52Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(52)

    def test_phase_loss_and_silence(self):
        x=torch.randn(2,4096)*.03
        self.assertLess(float(t.phase_complex_loss(x,x)),float(t.phase_complex_loss(-x,x)))
        p=(-x).clone().requires_grad_(True)
        t.phase_complex_loss(p,x).backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum()),0)
        z=torch.zeros_like(x).requires_grad_(True)
        t.phase_complex_loss(z,z.detach()).backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_direct_complex_head(self):
        m=t.ComplexResidualUNet()
        x=torch.randn(1,4096)*.05
        y=m(x)
        torch.testing.assert_close(y,x,atol=1e-6,rtol=1e-5)
        (t.enhancement_loss(y,x*.5,"V3_RESIDUAL_PHASE")+.15*t.phase_complex_loss(y,x*.5)).backward()
        self.assertTrue(torch.isfinite(m.complex_head.weight.grad).all())
        self.assertGreater(float(m.complex_head.weight.grad.abs().sum()),0)
        self.assertEqual(submit_v5.FAMILIES["complex_residual"],"ComplexResidualUNet")
        # A zero input bin can be synthesized by the additive head.
        with torch.no_grad(): m.complex_head.bias[0]=1
        spec=torch.zeros(1,5,7,dtype=torch.complex64)
        features=torch.zeros(1,16,5,7)
        self.assertGreater(float(m.spectrum(spec,features,None,None).abs().sum()),0)

    def test_empirical_and_balanced_remix(self):
        rng=np.random.default_rng(42)
        train=t.SpeechPairs.__new__(t.SpeechPairs)
        train.split="train"
        train.clean=rng.normal(0,.05,(1296,1024)).astype(np.float32)
        noise=rng.normal(0,1,train.clean.shape).astype(np.float32)
        noise*=np.sqrt(np.mean(train.clean**2,axis=1,keepdims=True)/np.mean(noise**2,axis=1,keepdims=True))
        labels=np.resize([0.,5.,10.,15.],1296)
        train.noisy=train.clean+noise*10**(-labels[:,None]/20)
        train.noisy=train.noisy.astype(np.float32)
        train.manifest=pd.DataFrame({"noise_type":["common"]*1200+["rare"]*96})
        data=t.ChallengeRemix(train,64,52,empirical=True,balanced=True,exclude=list(range(32)))
        self.assertTrue(all(i not in data.pool for i in range(32)))
        self.assertGreater(data.noise_group_probs[1],96/1264)
        for i in range(16):
            x,c=data[i]
            value=float(10*torch.log10(c.square().mean()/(x-c).square().mean()))
            self.assertLess(min(abs(value-v) for v in (0,5,10,15)),.001)
            self.assertTrue(torch.equal(x,data[i][0]))
        train.split="validation"
        with self.assertRaises(ValueError): t.ChallengeRemix(train,64,52,empirical=True)

    def test_replay_rank_and_metric_gate(self):
        labels=torch.tensor([[.1,.4,.8,1.]])
        self.assertLess(float(t.rank_regression_loss(labels,labels)),float(t.rank_regression_loss(labels.flip(1),labels)))
        c=t.PESQCritic(); opt=torch.optim.AdamW(c.parameters(),lr=.001)
        replay=t.CriticReplay(capacity=2,seed=52)
        x=torch.randn(2,4096)*.05
        target=x*.7
        def fake(sr,ref,pred,mode): return float(4.5-10*np.mean((ref-pred)**2))
        for _ in range(3):
            self.assertTrue(np.isfinite(t.critic_replay_step(c,opt,replay,x*.9,x,target,fake)))
        self.assertEqual(len(replay.items),2)
        self.assertTrue(all(not z.requires_grad and z.device.type=="cpu" for group in replay.items for z in group))
        self.assertTrue(all(not p.requires_grad for p in c.parameters()))
        self.assertEqual(t.adaptive_metric_weight({"mae":1.111,"pearson":.769},5,50),0)
        self.assertEqual(t.adaptive_metric_weight({"mae":.2,"pearson":.8},2,50),0)
        self.assertEqual(t.adaptive_metric_weight({"mae":.2,"pearson":.8},5,4),0)
        self.assertAlmostEqual(t.adaptive_metric_weight({"mae":.2,"pearson":.8},5,50),.05)
        model=t.ComplexResidualUNet()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        probe={}
        v=t.supervised_step(model,optimizer,x,target,{"id":"V3_RESIDUAL_PHASE","phase_loss":True},
            critic=c,critic_optimizer=opt,pesq_fn=fake,metric_weight=.05,update_critic=True,metric_probe=probe,critic_replay=replay)
        self.assertTrue(np.isfinite(v));self.assertGreater(probe["gradient_norm"],0)

    def test_stage_sequence_and_no_transductive(self):
        records=[]
        names=[]
        for _ in range(7):
            plan=v4.v52_stage_plan(records,[])
            self.assertFalse(plan["transductive"])
            self.assertEqual(plan["challenge_remix"],1024)
            names.append(plan["v52_stage"])
            records.append(({"kind":"v4_external"},plan))
        self.assertEqual(names,["phase-control","phase-loss","complex-head","empirical-remix","balanced-remix","fixed-critic","adaptive-critic"])
        self.assertIsNone(v4.v52_stage_plan(records,[]))
        self.assertEqual(v4.v52_stage_plan([], [{"plan":{"v52_stage":"phase-control"}}])["v52_stage"],"phase-loss")

if __name__=="__main__": unittest.main()
