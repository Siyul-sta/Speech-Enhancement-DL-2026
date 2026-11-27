"""V5.3 transfer contracts; no private data or GPU. Optional real CMGAN."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tools")]
import kaggle_v4_external as t
import ralph_v4 as v4

def mock_cmgan():
    m=t.CMGANTransfer.__new__(t.CMGANTransfer)
    nn.Module.__init__(m)
    m.net=nn.Module()
    for name in ("dense_encoder","TSCB_1","TSCB_2","TSCB_3","TSCB_4","mask_decoder","complex_decoder"):
        setattr(m.net,name,nn.Linear(8,8))
    return m

class V53Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(53)

    def test_progressive_and_lr_partition(self):
        m=mock_cmgan()
        original=t.transfer_state_hash(m)
        schedule=t.TransferSchedule(m,"progressive",3e-5)
        opt=schedule.optimizer()
        self.assertEqual(t.transfer_state_hash(m),original)
        self.assertEqual(len({id(p) for g in opt.param_groups for p in g["params"]}),sum(len(g["params"]) for g in opt.param_groups))
        self.assertEqual(schedule.apply(1)["active_groups"],["decoder"])
        self.assertEqual(schedule.apply(4)["active_groups"],["decoder","late"])
        self.assertEqual(schedule.apply(7)["active_groups"],["decoder","late","early"])
        self.assertEqual(schedule.apply(10)["active_groups"],["decoder","late","early","encoder"])
        for i,g in enumerate(opt.param_groups):
            self.assertAlmostEqual(g["lr"],3e-5*.3**i)
        schedule.apply(1)
        x=torch.randn(2,8)
        opt.zero_grad(set_to_none=True)
        m.net.mask_decoder(m.net.dense_encoder(x)).square().mean().backward()
        self.assertIsNone(m.net.dense_encoder.weight.grad)
        before=m.net.dense_encoder.weight.detach().clone()
        opt.step()
        torch.testing.assert_close(before,m.net.dense_encoder.weight)
        schedule.apply(10)
        opt.zero_grad(set_to_none=True)
        m.net.mask_decoder(m.net.dense_encoder(x)).square().mean().backward()
        self.assertGreater(float(m.net.dense_encoder.weight.grad.abs().sum()),0)

    def test_lora_merge_ordinary_state(self):
        m=mock_cmgan()
        x=torch.randn(2,8)
        before=m.net.TSCB_4(x).detach()
        schedule=t.TransferSchedule(m,"lora",3e-5)
        self.assertEqual(schedule.apply(1)["active_groups"],["decoder","lora"])
        torch.testing.assert_close(m.net.TSCB_4(x),before)
        original=m.net.TSCB_4.parametrizations.weight.original
        self.assertFalse(original.requires_grad)
        m.net.TSCB_4(x).square().mean().backward()
        self.assertIsNone(original.grad)
        self.assertGreater(float(m.net.TSCB_4.parametrizations.weight[0].b.grad.abs().sum()),0)
        schedule.optimizer().step()
        expected=m.net.TSCB_4(x).detach()
        schedule.merge()
        self.assertFalse(any("parametrizations" in k for k in m.state_dict()))
        restored=mock_cmgan()
        restored.load_state_dict(m.state_dict(),strict=True)
        torch.testing.assert_close(restored.net.TSCB_4(x),expected)

    def test_metricgan_parameter_partition(self):
        m=t.MetricGANTransfer.__new__(t.MetricGANTransfer)
        nn.Module.__init__(m)
        m.generator=nn.Sequential(nn.LSTM(8,8,batch_first=True),nn.Linear(8,8),nn.PReLU())
        schedule=t.TransferSchedule(m,"progressive",3e-5)
        self.assertFalse(any(p.requires_grad for p in m.generator[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in m.generator[1].parameters()))
        schedule.apply(4)
        self.assertTrue(all(p.requires_grad for p in m.parameters()))

    def test_sepformer_gradient_path(self):
        class Mask(nn.Module):
            def __init__(self):
                super().__init__();self.tail=nn.Linear(8,8)
            def forward(self,x): return self.tail(x).sigmoid()[None]
        m=t.SepformerTransfer.__new__(t.SepformerTransfer)
        nn.Module.__init__(m)
        m.encoder=nn.Linear(8,8);m.masknet=Mask();m.decoder=nn.Linear(8,8)
        schedule=t.TransferSchedule(m,"progressive",3e-5)
        x=torch.randn(2,8)
        m(x).square().mean().backward()
        self.assertIsNone(m.masknet.tail.weight.grad)
        self.assertIsNone(m.encoder.weight.grad)
        schedule.apply(10)
        m(x).square().mean().backward()
        self.assertGreater(float(m.masknet.tail.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.encoder.weight.grad.abs().sum()),0)

    def test_guard_and_suite(self):
        self.assertTrue(t.transfer_candidate_allowed(3.,[2.8,2.5],[2.9,2.6]))
        self.assertFalse(t.transfer_candidate_allowed(4.,[2.5,2.6],[2.9,2.6]))
        self.assertFalse(t.transfer_candidate_allowed(3.,[2.8],[2.9,2.6]))
        records=[]
        for _ in range(11):
            p=v4.v53_stage_plan(records,[],12)
            self.assertFalse(p["transductive"]);self.assertFalse(p["pesq_aware"])
            records.append(({"kind":"v4_external"},{**p,"v53_requested_epochs":12,"model_family":p["architecture"],"v53_initial_hash":"a"*64}))
        self.assertIsNone(v4.v53_stage_plan(records,[],12))
        self.assertEqual(v4.v53_stage_plan(records,[],25)["v53_stage"],"cmgan-decoder")
        self.assertEqual(v4.v53_stage_plan(records[:1],[],12)["v53_expected_initial_hash"],"a"*64)

    @unittest.skipUnless(os.environ.get("V5_TEST_PUBLIC_CMGAN")=="1","opt-in real CMGAN")
    def test_actual_cmgan_progressive_and_lora(self):
        with tempfile.TemporaryDirectory() as d,patch.object(t,"WORK",Path(d)):
            m=t.CMGANTransfer().eval()
            x=(.1*torch.sin(torch.arange(4096)*.0864))[None]
            with torch.no_grad(): reference=m(x)
            s=t.TransferSchedule(m,"progressive",3e-5)
            s.apply(4)
            m(x).abs().mean().backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.net.TSCB_4.parameters()))
            self.assertTrue(all(p.grad is None for p in m.net.dense_encoder.parameters()))
            m.zero_grad(set_to_none=True)
            a=t.TransferSchedule(m,"lora",3e-5)
            with torch.no_grad(): torch.testing.assert_close(m(x),reference,atol=1e-6,rtol=1e-5)
            m(x).abs().mean().backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for layer in a.adapters for p in layer.parametrizations.weight[0].parameters()))
            a.optimizer().step()
            m.eval()
            with torch.no_grad(): expected=m(x)
            a.merge()
            self.assertFalse(any("parametrizations" in k for k in m.state_dict()))
            restored=t.CMGANTransfer().eval()
            restored.load_state_dict(m.state_dict(),strict=True)
            with torch.no_grad(): torch.testing.assert_close(restored(x),expected,atol=1e-5,rtol=1e-4)

if __name__=="__main__": unittest.main()
