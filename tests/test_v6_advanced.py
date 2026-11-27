"""V6.1 advanced adapter tests. Synthetic only; public assets opt-in per CI job."""
from __future__ import annotations
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import ralph_v6 as ctl
from v6_preflight import load_module

class Advanced(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(42)
        cls.tmp=tempfile.TemporaryDirectory();root=Path(cls.tmp.name)
        source=root/"source.py";source.write_text(ctl.source(),encoding="utf-8")
        cls.mod=load_module(source);cls.root=root
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_scan_exact_and_gradients(self):
        m=self.mod
        a=torch.rand(2,19,4,3,dtype=torch.float64,requires_grad=True)
        b=torch.randn_like(a,requires_grad=True);h=torch.randn(2,4,3,dtype=a.dtype,requires_grad=True)
        z=m.v6_affine_scan(a,b,h)
        ref=[];s=h
        for k in range(a.shape[1]):
            s=a[:,k]*s+b[:,k];ref.append(s)
        ref=torch.stack(ref,1)
        self.assertTrue(torch.allclose(z,ref,atol=1e-12,rtol=1e-12))
        gz=torch.autograd.grad(z.square().sum(),(a,b,h),retain_graph=True)
        gr=torch.autograd.grad(ref.square().sum(),(a,b,h))
        for x,y in zip(gz,gr): self.assertTrue(torch.allclose(x,y,atol=1e-10,rtol=1e-10))

    def test_mamba_layout_and_backward(self):
        m=self.mod.V6Mamba1(16,d_state=4,d_conv=3,expand=2)
        self.assertEqual(set(m.state_dict()),{"A_log","D","in_proj.weight","conv1d.weight","conv1d.bias","x_proj.weight","dt_proj.weight","dt_proj.bias","out_proj.weight"})
        x=torch.randn(2,53,16,requires_grad=True);y=m(x)
        self.assertEqual(x.shape,y.shape);self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_rose_config_rejects_wrong_model(self):
        m=self.mod
        cfg=dict(backbone="ncsnpp_v2",sde="ouve",loss_type="precond_denoiser",theta=1.5,sigma_min=.05,sigma_max=.5)
        self.assertEqual(m.v6_rose_config(cfg)["sr"],16000)
        with self.assertRaises(ValueError): m.v6_rose_config({**cfg,"loss_type":"score_matching"})
        with self.assertRaises(ValueError): m.v6_rose_config({**cfg,"sr":48000})

    def test_real_public_model(self):
        family=os.environ.get("V6_TEST_ADVANCED")
        if family not in ctl.ADVANCED: self.skipTest("Set V6_TEST_ADVANCED for real upstream assets")
        m=self.mod;root=self.root/family
        cfg=ctl.config(family,"train","a"*64,{},epochs=2)
        m.EXPERIMENT_CONFIG=cfg
        print("ACTUAL PUBLIC MODEL START",family,flush=True)
        model=m.v6_make_model(cfg,mp_root=root).eval()
        t=torch.arange(4096,dtype=torch.float32)/16000
        x=(.04*torch.sin(2*torch.pi*220*t)+.005*torch.randn_like(t))[None]
        with torch.no_grad(): y=model(x)
        self.assertEqual(y.shape,x.shape);self.assertTrue(torch.isfinite(y).all())
        self.assertGreater(float(y.abs().max()),1e-8)
        other=m.v6_make_model(cfg,mp_root=root,pretrained=False).eval()
        other.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad(): z=other(x)
        self.assertTrue(torch.allclose(y,z,atol=2e-5,rtol=2e-4))
        del other
        model.train();schedule=m.TransferSchedule(model,"progressive",3e-5);schedule.apply(1)
        pred=model(x);loss=m.v6_loss(pred,x,cfg);loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(float(g.abs().max())>0 for g in grads))
        print("ACTUAL PUBLIC MODEL PASS",family,flush=True)

if __name__=="__main__": unittest.main()
