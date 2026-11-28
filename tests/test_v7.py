"""V7 offline tests: reviewed source contracts, objectives, search, quota wait and recovery."""
import ast
import json
from contextlib import ExitStack
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
    def test_controller_one_round_and_recovery_order(self):
        # Integration fixture: real recipe construction and journal, fake remote/CPU.
        cfg=search.build(self.parent,"spectral_balance",self.sha,None,1,2048,512)
        smoke=search.build(self.parent,"spectral_balance",self.sha,None,1,2048,512,True)
        plan={"config":cfg,"smoke":smoke,"reuse_smoke":False,"hours":.5,"reason":{"family":"spectral"},"compound_data_change":False}
        q={"remaining":10.,"used":20.,"total":30.,"refresh":None}
        calls=[]
        def submit(kaggle,code,tag,num,c,wait):
            calls.append("smoke" if c["smoke"] else "train")
            result={"best_validation_pesq_wb":3.3,"v6":{"metric_steps":0,"critic_calibration":None},
                "v7":{"protocol":search.VERSION,"recipe":c["v7_recipe"],"mutation":c["v7_mutation"],"parent":c["v6_parent"],
                    "warmup_updates":0,"warmup_calibration":None}}
            return {"ref":"owner/new"},result
        with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
            root=Path(d);remote=root/"remote";output=root/"v7"
            patches=[(v7.core,"OUTPUT",remote),(v7,"ROOT",output),(v7.shutil,"which",lambda _: "kaggle"),
                (v7.refine,"wait_idle",lambda _:calls.append("idle")),(v7.ctl,"resume",lambda *a:calls.append("resume")),
                (v7,"quota",lambda *a,**k:q),(v7,"verified_entries",lambda:{"parent":self.parent}),
                (v7.oldloop,"train_diagnostic",lambda *a:self.diag),(v7.v4,"find_public_cache",lambda:None),
                (v7.search,"pick",lambda *a:plan),(v7.search,"cost",lambda *a:.25),
                (v7.focus,"preflight",lambda *a:calls.append("preflight")),(v7.ctl,"submit",submit)]
            for module,key,val in patches:stack.enter_context(patch.object(module,key,val))
            import v6_preflight
            stack.enter_context(patch.object(v6_preflight,"require",lambda *a:None))
            a=SimpleNamespace(data_root=root,execute=True,allow_public_assets=True,resume=True,max_job_minutes=360,
                budget_hours=3.,reserve_hours=2.,wait_for_quota=False,max_quota_cycles=2,max_wait_hours=168.,
                poll_minutes=10,max_rounds=1,max_epochs=200,chunk_epochs=3,allow_reduce_mix=True,enable_korean=False,
                max_stale=0,skip_final=True,top=3)
            v7.run(a)
            self.assertEqual(calls,["idle","resume","preflight","smoke","train"])
            self.assertFalse((remote/"active.lock").exists())
            j=json.loads(next(output.glob("run_*.json")).read_text())
            self.assertEqual(j["stop_reason"],"round_limit");self.assertEqual(j["jobs"],2)
            self.assertEqual(j["events"][0]["status"],"complete")
            self.assertAlmostEqual(j["events"][0]["gain"],.05)
            calls.clear()
            with patch.object(v7.ctl,"resume",side_effect=TimeoutError("unresolved")):
                with self.assertRaises(TimeoutError):v7.run(a)
            self.assertEqual(calls,["idle"]);self.assertFalse((remote/"active.lock").exists())

    def test_actual_mp_public_checkpoint(self):
        import os
        if os.environ.get("V7_TEST_MP")!="1":self.skipTest("Optional public MP-DNS assets")
        from v6_preflight import load_module
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runner=root/"runner.py";runner.write_bytes(search.source().encode("utf-8"))
            m=load_module(runner);assets=root/"assets"
            model=m.V6MPNet("mp_dns",assets,allow_download=True).eval()
            torch.manual_seed(7)
            x=torch.randn(1,48000)*.03
            with torch.no_grad():y=model(x)
            self.assertEqual(tuple(y.shape),(1,48000));self.assertTrue(torch.isfinite(y).all())
            other=m.V6MPNet("mp_dns",assets,False,False).eval()
            other.load_state_dict(model.state_dict(),strict=True)
            with torch.no_grad():z=other(x)
            self.assertTrue(torch.allclose(y,z,atol=2e-5,rtol=2e-4))
            crop=x[:,:4096];clean=crop*.8
            cfg={"id":"V3_RESIDUAL_PHASE","v610_phase_derivatives":True,"v7_phase_weight":.08,"v7_spectral_weight":.04}
            model.train();m.TransferSchedule(model,"progressive",3e-5)
            loss=m.v6_loss(model(crop),clean,cfg);loss.backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in model.parameters()))
            refiner=m.V6Refiner(other,2).eval()
            with torch.no_grad():self.assertTrue(torch.allclose(refiner(crop),other(crop),atol=2e-5,rtol=2e-4))
            refiner.train();stages=refiner.forward_stages(crop)
            loss=m.v6_loss(stages[-1],clean,cfg)+.02*m.v7_residual_trust(stages[-1],stages[0],crop)
            loss.backward()
            self.assertFalse(any(p.grad is not None for p in refiner.coarse.parameters()))
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in refiner.phase.parameters()))

if __name__=="__main__":unittest.main()
