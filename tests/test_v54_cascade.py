"""V5.4 CPU contracts; actual pinned CMGAN test optional, no private audio."""
import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tools")]
import kaggle_v4_external as t
import ralph_v4 as v4
import submit_v5

CFG={"id":"V3_RESIDUAL_PHASE","phase_loss":True,"cascade_aux":True}

class V54Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(54)

    def test_identity_depth_and_gradient_schedule(self):
        x=torch.randn(1,4096)*.05
        for cls,depth in ((t.CascadeNativeTwo,2),(t.CascadeNativeThree,3),(t.CascadeEnhancer,4)):
            with self.subTest(depth=depth):
                m=cls()
                self.assertEqual(m.set_epoch(1),["coarse","residual"])
                stages=m.forward_stages(x)
                self.assertEqual(list(stages),["coarse","residual","complex","refine"][:depth])
                for y in stages.values():
                    self.assertEqual(y.shape,x.shape)
                    torch.testing.assert_close(y,x,atol=2e-6,rtol=1e-5)
                opt=torch.optim.AdamW(m.parameters(),lr=.001)
                loss=t.supervised_step(m,opt,x,x*.6,CFG)
                self.assertTrue(np.isfinite(loss))
                self.assertGreater(float(m.residual.out.weight.grad.abs().sum()),0)
                if depth>=3: self.assertIsNone(m.complex.head.weight.grad)
                if depth>=4: self.assertIsNone(m.refine.out.weight.grad)
                m.set_epoch(5)
                loss=t.supervised_step(m,opt,x,x*.6,CFG)
                self.assertTrue(np.isfinite(loss))
                if depth>=3: self.assertGreater(float(m.complex.head.weight.grad.abs().sum()),0)
                if depth>=4: self.assertGreater(float(m.refine.out.weight.grad.abs().sum()),0)

    def test_original_input_skip(self):
        noisy=torch.randn(1,4096)*.05
        prev=torch.zeros_like(noisy)
        m=t.ConditionedWaveRefiner()
        with torch.no_grad():
            m.out.weight.fill_(.001)
        a=m(noisy,prev)
        b=m(-noisy,prev)
        self.assertGreater(float((a-b).abs().sum()),0)
        # Complex refiner can synthesize a nonzero spectrum from a zero prior.
        z=t.ConditionedComplexRefiner()
        with torch.no_grad(): z.head.bias[0]=.1
        self.assertGreater(float(z(noisy,prev).abs().sum()),0)
        self.assertTrue(torch.isfinite(z(torch.zeros_like(noisy),prev)).all())

    def test_native_strict_checkpoint(self):
        self.assertEqual(submit_v5.FAMILIES["cascade_native"],"CascadeEnhancer")
        self.assertNotIn("cascade_cmgan",submit_v5.FAMILIES)
        m=t.CascadeEnhancer().eval()
        x=torch.randn(1,4096)*.03
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"best_model.pt"
            torch.save(m.state_dict(),path)
            restored=t.CascadeEnhancer().eval()
            restored.load_state_dict(torch.load(path,weights_only=True),strict=True)
            with torch.no_grad(): torch.testing.assert_close(m(x),restored(x))

    def test_report_order_and_noisy_only(self):
        m=t.CascadeNativeTwo().eval()
        x=torch.randn(3,4096)*.03
        def fake(sr,c,y,mode): return float(4.5-np.mean((c-y)**2))
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,x*.6),batch_size=2)
        scores=t.evaluate_cascade(m,loader,"cpu",fake)
        self.assertEqual(list(scores),["coarse","residual"])
        self.assertEqual(len(scores["residual"]),3)
        self.assertAlmostEqual(scores["residual"][0],fake(16000,x[0].numpy()*.6,x[0].numpy(),"wb"),places=5)
        with torch.no_grad(): torch.testing.assert_close(m(x),m.forward_stages(x)["residual"])

    def test_artifact_gate_and_suite(self):
        records=[]
        for _ in range(8):
            p=v4.v54_stage_plan(records,[],8)
            self.assertEqual(p["batch_size"],1)
            self.assertFalse(p["transductive"])
            self.assertEqual(p["pesq_aware"],p["adaptive_critic"])
            records.append(({"kind":"v4_external"},{**p,"v54_requested_epochs":8}))
        self.assertIsNone(v4.v54_stage_plan(records,[],8))
        self.assertEqual(v4.v54_stage_plan(records,[],12)["v54_stage"],"native-two")
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            def scores(name,n,value):
                with (p/name).open("w",newline="",encoding="utf-8") as f:
                    w=csv.writer(f);w.writerow(["index","pesq_wb"]);w.writerows((i,value) for i in range(n))
            scores("zero_shot_validation.csv",164,1.4)
            scores("zero_shot_external.csv",64,1.4)
            for name,value in (("coarse",1.8),("residual",2.)):
                scores("cascade_"+name+"_validation.csv",164,value)
                scores("cascade_"+name+"_external.csv",64,value)
            h=[{"epoch":i,"active_stages":["coarse","residual"],"trainable_parameters":100} for i in range(1,9)]
            (p/"cascade_history.json").write_text(json.dumps(h))
            r={"v54_stage":"native-two","v54_requested_epochs":8,"epochs_completed":8,
                "model_family":"cascade_native_two","cascade_depth":2,"cascade_pretrained":False,
                "critic_reserved_train_files":32,"zero_shot":{"challenge":1.4,"external_domains":[1.4,1.4]},
                "cascade_scores":{"coarse":{"challenge":1.8,"external_domains":[1.8,1.8]},"residual":{"challenge":2.,"external_domains":[2.,2.]}},
                "best_validation_pesq_wb":2.,"domain_scores":{"external_normal":2.,"external_stress":2.}}
            v4.verify_cascade_result(p,r,"v4_external")
            r["cascade_scores"]["residual"]["challenge"]=2.1
            with self.assertRaises(ValueError): v4.verify_cascade_result(p,r,"v4_external")

    @unittest.skipUnless(os.environ.get("V5_TEST_PUBLIC_CMGAN")=="1","opt-in real CMGAN")
    def test_actual_cmgan_frozen_cascade(self):
        with tempfile.TemporaryDirectory() as d,patch.object(t,"WORK",Path(d)):
            m=t.CascadeCMGANFour()
            x=(.1*torch.sin(torch.arange(4096)*.0864)+.01*torch.randn(4096))[None]
            m.eval()
            with torch.no_grad():
                stages=m.forward_stages(x)
                torch.testing.assert_close(stages["coarse"],stages["refine"],atol=2e-6,rtol=1e-5)
            original=t.transfer_state_hash(m.coarse)
            self.assertEqual(m.set_epoch(5),["residual","complex","refine"])
            m.train()
            opt=torch.optim.AdamW(m.parameters(),lr=.001)
            self.assertTrue(np.isfinite(t.supervised_step(m,opt,x,x*.7,CFG)))
            self.assertTrue(all(p.grad is None for p in m.coarse.parameters()))
            self.assertEqual(t.transfer_state_hash(m.coarse),original)
            for head in (m.residual.out,m.complex.head,m.refine.out):
                self.assertTrue(torch.isfinite(head.weight.grad).all())
                self.assertGreater(float(head.weight.grad.abs().sum()),0)

if __name__=="__main__": unittest.main()
