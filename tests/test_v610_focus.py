"""V6.10 CPU tests for new training-only objectives and controlled recipe lineage."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"src")]
import ralph_v610 as focus
import ralph_v6 as ctl
from v610_focus import v610_phase_derivatives,v610_noise_weights

class FocusTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.sha=focus.digest(focus.source())
        self.old=ctl.config("mp_dns","train",focus.digest(ctl.source()),{},epochs=3)
        self.result={"best_validation_pesq_wb":3.2,"domain_scores":{"external_normal":3.5,"external_stress":2.5},
            "v6":{"checkpoint_sha256":"b"*64,"attempted_epochs":3,"checkpoint_epoch":2,"root_external_domains":[3.5,2.5],"lr":2e-5}}
        self.parent=({"ref":"owner/job"},self.result,self.old,Path("dummy"))
    def test_overlay_keeps_base_evaluator(self):
        base=ctl.source();new=focus.source()
        for name in ("def evaluate(","def score_waveforms(","def v6_eval(","def v6_make_model("):
            def body(code):
                start=code.index(name);end=code.find("\ndef ",start+1)
                return code[start:end]
            self.assertEqual(body(base),body(new))
        self.assertNotEqual(focus.digest(base),self.sha)
        self.assertEqual(new.count("V610_HARD_INIT"),1)
    def test_phase_finite_and_sensitive(self):
        torch.manual_seed(6);t=torch.arange(4096)/16000.
        clean=(.1*torch.sin(2*torch.pi*350*t)+.03*torch.sin(2*torch.pi*730*t))[None]
        pred=(torch.roll(clean,11,dims=-1)+.02*torch.randn_like(clean)).requires_grad_()
        a=v610_phase_derivatives(clean,clean)
        b=v610_phase_derivatives(pred,clean)
        self.assertGreater(float(b),float(a)+.01)
        b.backward()
        self.assertTrue(torch.isfinite(pred.grad).all());self.assertGreater(float(pred.grad.abs().max()),0)
        silent=torch.zeros_like(clean,requires_grad=True)
        s=v610_phase_derivatives(silent,torch.zeros_like(clean));s.backward()
        self.assertTrue(torch.isfinite(silent.grad).all())
    def test_initial_hard_weight_train_only(self):
        train=SimpleNamespace(split="train",manifest=pd.DataFrame({"noise_type":["a","b","a","a","b","b"]}))
        class Mix:
            fit=[0,1];hard=True;group_names=["a","b"];probs=np.array([.5,.5])
            def update_hard(self,scores):
                self.scores=scores
                w=np.array([3.5-scores[k] for k in self.group_names]);self.probs=w/w.sum()
        mix=Mix()
        d=v610_noise_weights(mix,train,[2,3,4,5],[(0,1.),(1,1.),(2,3.),(3,3.)])
        self.assertGreater(d["probabilities"]["a"],d["probabilities"]["b"])
        self.assertEqual(d["counts"],{"a":2,"b":2})
        with self.assertRaises(ValueError):v610_noise_weights(mix,train,[0],[(0,1.)])
        train.split="validation"
        with self.assertRaises(ValueError):v610_noise_weights(mix,train,[2],[(0,1.)])
    def test_controlled_phase_trial(self):
        c=focus.trial(self.parent,"phase_derivatives",self.sha,None,1,2048,512)
        self.assertEqual(c["v6_parent"]["sha256"],"b"*64)
        self.assertTrue(c["v610_phase_derivatives"]);self.assertFalse(c["v610_hard_first"])
        self.assertEqual(c["v6_crop"],self.old["v6_crop"])
        self.assertEqual(c["v6_mix"],self.old["v6_mix"])
        self.assertFalse(c["v6_parent_coarse"])
        self.assertEqual(c["v56_source_sha"],self.sha)
        sm=focus.trial(self.parent,"phase_derivatives",self.sha,None,1,2048,512,True)
        self.assertTrue(sm["smoke"]);self.assertIsNone(sm["warm_ref"])
    def test_residual_parent_and_phase_inheritance(self):
        c=focus.trial(self.parent,"residual",self.sha,None,1,2048,512)
        self.assertTrue(c["v6_parent_coarse"]);self.assertEqual(c["v6_depth"],2)
        self.assertEqual(c["v6_base_family"],"mp_dns");self.assertEqual(c["v6_mode"],"cascade")
        self.old["v610_phase_derivatives"]=True
        h=focus.trial(self.parent,"hard_noise_first",self.sha,None,1,2048,512)
        self.assertTrue(h["v610_phase_derivatives"]);self.assertTrue(h["v610_hard_first"]);self.assertTrue(h["v6_hard"])
    def test_hypothesis_uses_exact_parent(self):
        report={"snapshot_champion":ctl.parent_info(self.parent[0],self.result),"train_oracle_diagnostic":{"phase_gain":.5,"magnitude_gain":.2}}
        self.assertEqual(focus.select_recipe(self.parent,report,{}),"phase_derivatives")
        report["train_oracle_diagnostic"]={"phase_gain":.1,"magnitude_gain":.5}
        self.assertEqual(focus.select_recipe(self.parent,report,{}),"hard_noise_first")
        report["snapshot_champion"]={}
        with self.assertRaises(ValueError):focus.select_recipe(self.parent,report,{})
if __name__=="__main__":unittest.main()
