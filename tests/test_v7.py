"""V7 offline tests: reviewed source contracts, objectives, search, quota wait and recovery."""
import ast
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"src")]
import ralph_v6 as ctl
import ralph_v7 as v7
import v7_search as search
import v7_objectives as obj
import ralph_v610 as focus

class V7Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.sha=search.digest(search.source())
        self.old=ctl.config("mp_dns","train",search.digest(ctl.source()),{},epochs=3)
        self.r={"best_validation_pesq_wb":3.25,"smoke":False,"best_epoch":2,"v56_source_sha":search.digest(ctl.source()),
            "domain_scores":{"external_normal":3.5,"external_stress":2.5},
            "v6":{"checkpoint_sha256":"b"*64,"attempted_epochs":3,"checkpoint_epoch":2,"root_external_domains":[3.5,2.5],"lr":2e-5,"gain":.07}}
        self.parent=({"ref":"owner/job","status":"COMPLETE"},self.r,self.old,Path("dummy"))
        self.diag={"phase_gain":.5,"magnitude_gain":.2}
    def test_source_contracts(self):
        a,b=focus.source(),search.source()
        def body(code,name):
            return ast.dump(next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name==name))
        for name in ("evaluate","score_waveforms","v6_eval","v6_loaders","v6_make_model","v6_parent_path"):
            self.assertEqual(body(a,name),body(b,name))
        self.assertNotEqual(search.digest(a),self.sha)
        self.assertEqual(b.count("V7 CRITIC WARMUP"),1)
    def test_spectral_loss_and_silence_gradients(self):
        torch.manual_seed(7);t=torch.arange(4096)/16000
        s=(.05*torch.sin(t*2000))[None];x=(s+.03*torch.randn_like(s)).requires_grad_()
        loss=obj.v7_spectral_balance(x,s)
        self.assertGreater(float(loss.detach()),0)
        self.assertAlmostEqual(float(obj.v7_spectral_balance(s,s)),0.,places=6)
        loss.backward();self.assertTrue(torch.isfinite(x.grad).all())
        a=torch.zeros_like(s,requires_grad=True);b=obj.v7_spectral_balance(a,a.detach());b.backward()
        self.assertTrue(torch.isfinite(a.grad).all())
    def test_residual_trust_no_coarse_gradient(self):
        x=torch.randn(1,4096)*.01;coarse=x.clone().requires_grad_();p=(x+.01).requires_grad_()
        self.assertAlmostEqual(float(obj.v7_residual_trust(x,x,x)),0.)
        loss=obj.v7_residual_trust(p,coarse,x);loss.backward()
        self.assertIsNone(coarse.grad);self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(loss.detach()),0)
    def test_critic_gate(self):
        good={"pearson":.8,"mae":.3}
        self.assertEqual(obj.v7_critic_gate(good,24),.025)
        for bad,n in [(None,24),({"pearson":.2,"mae":.3},24),({"pearson":.8,"mae":.8},24),(good,4)]:
            self.assertEqual(obj.v7_critic_gate(bad,n),0.)
    def test_critic_warmup_disjoint(self):
        fit=list(range(40));calids=[40,41]
        class Train:
            split="train"
            def __getitem__(self,i):return torch.zeros(48000),torch.zeros(48000)
        model=torch.nn.Identity();seen=[]
        with patch.object(obj,"v6_split",lambda tr:(fit,calids,[42],[43]),create=True),patch.object(obj,"critic_replay_step",lambda *a:seen.append(1),create=True),patch.object(obj,"critic_calibration",lambda *a:{"count":32,"pearson":.7,"mae":.4},create=True):
            n,c=obj.v7_critic_warmup(model,None,None,None,Train(),calids,[],torch.device("cpu"),None,42,24)
            self.assertEqual(n,24);self.assertEqual(len(seen),24);self.assertEqual(c["count"],32)
            with self.assertRaises(ValueError):obj.v7_critic_warmup(model,None,None,None,Train(),[0],[],torch.device("cpu"),None,42,24)
    def test_config_mutations_inherit(self):
        c=search.build(self.parent,"phase_light",self.sha,None,1,2048,512)
        self.assertEqual(c["v7_phase_weight"],.04);self.assertTrue(c["v610_phase_derivatives"])
        self.assertEqual(c["v6_parent"]["sha256"],"b"*64)
        self.old.update(v610_phase_derivatives=True,v7_phase_weight=.04)
        s=search.build(self.parent,"spectral_balance",self.sha,None,1,2048,512)
        self.assertEqual(s["v7_phase_weight"],.04);self.assertEqual(s["v7_spectral_weight"],.04)
        with self.assertRaises(ValueError):search.build(self.parent,"phase_light",self.sha,None,1,2048,512)
        with self.assertRaises(ValueError):search.build(self.parent,"arbitrary_code",self.sha,None,1,2048,512)
    def test_residual_and_critic(self):
        r=search.build(self.parent,"residual",self.sha,None,1,2048,512)
        self.assertEqual(r["v6_depth"],2);self.assertTrue(r["v6_parent_coarse"])
        self.assertFalse(search.available(self.old,"residual_trust"))
        c=search.build(self.parent,"critic_warm24",self.sha,None,1,2048,512)
        self.assertTrue(c["v6_critic"]);self.assertEqual(c["v6_crop"],48000);self.assertEqual(c["v7_critic_warm_count"],24)
        sm=search.build(self.parent,"critic_warm24",self.sha,None,1,2048,512,True)
        self.assertIsNone(sm["v6_parent"]);self.assertEqual(sm["v7_critic_warm_count"],24)
    def test_evidence_and_failure_memory(self):
        self.r["v6"]["gain"]=0.
        c=search.build(self.parent,"phase_derivatives",self.sha,None,1,2048,512)
        e={"f":({"status":"ERROR_TERMINAL"},None,c,Path("x"))}
        self.assertNotIn("phase_derivatives",[n for n,w in search.rank(self.parent,self.diag,e)])
        out,attempts=search.history(e)
        self.assertEqual(out["phase"],[0.]);self.assertIn("phase_derivatives",attempts["b"*64])
        self.assertNotIn("korean128",[n for n,w in search.rank(self.parent,self.diag,{})])
        self.assertIn("korean128",[n for n,w in search.rank(self.parent,self.diag,{},True)])
    def test_result_provenance(self):
        c=search.build(self.parent,"critic_warm24",self.sha,None,1,2048,512)
        r={"v7":{"protocol":search.VERSION,"recipe":c["v7_recipe"],"mutation":c["v7_mutation"],"parent":c["v6_parent"],"warmup_updates":24,"warmup_calibration":{"count":32}}}
        self.assertIs(search.verify_result(r,c),r)
        r["v7"]["warmup_updates"]=0
        with self.assertRaises(ValueError):search.verify_result(r,c)
    def test_budget_smoke_and_trial(self):
        def cost(c,e):return .2 if c["smoke"] else c["epochs"]*(.2+c["v6_mix"]/2048)
        with patch.object(search,"cost",cost):
            self.assertIsNone(search.pick(self.parent,self.diag,{},self.sha,None,.1,6,200,3,True))
            p=search.pick(self.parent,self.diag,{},self.sha,None,.8,6,200,3,True)
            self.assertLessEqual(p["hours"],.8);self.assertTrue(p["compound_data_change"])
            self.assertIsNone(search.pick(self.parent,self.diag,{},self.sha,None,.8,6,3,3,True))
    def test_quota_parser_and_actual_refresh(self):
        q=v7.parse_quota("Type Used Remaining Total RefreshAt\nGPU 10.74h 19.26h 30h 2026-11-28T00:00:00Z\nTPU 0h 20h 20h")
        self.assertEqual(q["remaining"],19.26)
        self.assertFalse(v7.refreshed(q,{**q,"refresh":"2026-12-05T00:00:00+00:00"}))
        self.assertTrue(v7.refreshed(q,{**q,"remaining":30.,"used":0.}))
        for bad in ("TPU 0h 20h 20h","GPU 2h nan 30h","GPU 1h 31h 30h"):
            with self.assertRaises(ValueError):v7.parse_quota(bad)
    def test_wait_refresh_never_submits(self):
        q={"used":29.,"remaining":1.,"total":30.,"refresh":"2026-11-28T00:00:00+00:00"}
        new={**q,"used":0.,"remaining":30.};t=[0.]
        def sleep(n):t[0]+=n
        with tempfile.TemporaryDirectory() as d,patch.object(v7.core,"OUTPUT",Path(d)),patch.object(v7.core,"no_uncertain_prior_jobs") as check:
            r=v7.await_refresh("dummy",q,180,2,poll_minutes=1,clock=lambda:t[0],sleep=sleep,reader=lambda:new)
            self.assertEqual(r,new);check.assert_called_once()
            self.assertIsNone(v7.await_refresh("dummy",new,180,2,poll_minutes=1,clock=lambda:t[0],sleep=sleep,reader=lambda:new))
if __name__=="__main__":unittest.main()
