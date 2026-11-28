"""V6 refinement tests: synthetic scores/audio, no GPU/network/private data."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import v6_refine_analysis as a
import ralph_v6_refine as r
import ralph_v6 as ctl

class Refine(unittest.TestCase):
    def test_score_matrix_and_group_diagnostics(self):
        m=np.array([[2.,4.],[4.,2.],[3.,3.],[4.,3.]])
        meta=[{"noise":"x","snr_group":"0–<5"},{"noise":"x","snr_group":"0–<5"},
              {"noise":"y","snr_group":"10–<15"},{"noise":"y","snr_group":"10–<15"}]
        s=a.matrix_summary(m,["A","B"])
        self.assertEqual(s["best"],"A")
        self.assertAlmostEqual(s["best_mean"],3.25)
        self.assertAlmostEqual(s["oracle_per_clip"],3.75)
        self.assertAlmostEqual(s["oracle_gap"],.5)
        g=a.group_summary(m,meta,["A","B"])
        self.assertEqual(g["noise"]["x"]["count"],2)
        self.assertAlmostEqual(g["noise"]["y"]["mean"]["A"],3.5)
    def test_hypothesis(self):
        g={"snr_group":{"0–<5":{"count":4,"mean":{"A":2.5}},"10–<15":{"count":8,"mean":{"A":3.5}}}}
        self.assertEqual(a.hypothesis({"magnitude_gain":.5,"phase_gain":.1},g,"A")["name"],"hard_noise")
        self.assertEqual(a.hypothesis({"magnitude_gain":.1,"phase_gain":.5},g,"A")["name"],"full_length")
    def test_cache_identity_and_tamper(self):
        class Counter(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=0
            def forward(self,x):self.calls+=1;return .8*x
        model=Counter();x=torch.randn(2,48000)*.01
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ident={"ref":"test","checkpoint":"hash"}
            y=a.cached_predictions(root,model,x,ident)
            self.assertEqual(model.calls,2)
            self.assertTrue(torch.equal(y,a.cached_predictions(root,model,x,ident)))
            self.assertEqual(model.calls,2)
            next(root.glob("*.npy")).write_bytes(b"tampered")
            self.assertTrue(torch.equal(y,a.cached_predictions(root,model,x,ident)))
            self.assertEqual(model.calls,4)
            a.cached_predictions(root,model,x,{"ref":"test","checkpoint":"different"})
            self.assertEqual(model.calls,6)
    def test_recipe_without_clean_input(self):
        x=torch.randn(3,4096);p=torch.stack((x,x*.5),1)
        y=a.predict_recipe(x,p,{"alignment":"none","weights":[.25,.75]},None)
        self.assertTrue(torch.allclose(y,x*.625,atol=1e-6))
    def test_trial_preserves_parent_and_recipe(self):
        sha="a"*64;old=ctl.config("mp_dns","train",sha,{},epochs=3)
        result={"best_validation_pesq_wb":3.2,"domain_scores":{"external_normal":3.5,"external_stress":2.5},
            "v6":{"checkpoint_sha256":"b"*64,"attempted_epochs":3,"checkpoint_epoch":2,"root_external_domains":[3.5,2.5],"lr":2e-5}}
        parent=({"ref":"owner/job"},result,old,Path("dummy"))
        cfg=r.build_trial(parent,{"hypothesis":{"name":"full_length"}},sha,{}, {},None,1)
        self.assertEqual(cfg["warm_ref"],"owner/job")
        self.assertEqual(cfg["v6_parent"]["sha256"],"b"*64)
        self.assertEqual(cfg["v6_crop"],48000)
        self.assertEqual(cfg["v6_mix"],old["v6_mix"])
        self.assertEqual(cfg["v6_public_mix"],old["v6_public_mix"])
        self.assertLessEqual(cfg["lr"],old["lr"]*.5)
        self.assertEqual(r.resign(cfg,sha),cfg)
        smoke=r.build_trial(parent,{"hypothesis":{"name":"full_length"}},sha,{}, {},None,1,smoke=True)
        self.assertTrue(smoke["smoke"])
        self.assertIsNone(smoke["warm_ref"])
        self.assertEqual(smoke["v6_crop"],48000)
        self.assertNotEqual(smoke["v56_key"],cfg["v56_key"])
    def test_wait_does_not_remove_active_lock(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);lock=p/"active.lock";lock.write_text("123")
            with patch.object(r.core,"OUTPUT",p):
                with self.assertRaises(TimeoutError):r.wait_idle(0,clock=lambda:0,sleep=lambda _:None)
            self.assertEqual(lock.read_text(),"123")
    def test_idle_window(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);t=[0];lock=p/"active.lock";lock.write_text("123")
            def sleep(n):
                t[0]+=n
                if t[0]>=30:lock.unlink(missing_ok=True)
            with patch.object(r.core,"OUTPUT",p):
                r.wait_idle(5,quiet_seconds=90,clock=lambda:t[0],sleep=sleep)
            self.assertEqual(t[0],120)
    def test_snapshot_excludes_unfinished(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"run_v4dummy"/"round_0001"/"v4_external";p.mkdir(parents=True)
            (p/"remote_state.json").write_text(json.dumps({"status":"SUBMIT_PENDING","v6":True}))
            with patch.object(a.core,"OUTPUT",Path(d)),patch.object(a.ctl,"verify") as verify:
                entries,rejected=a.snapshot()
                self.assertEqual(entries,[]);verify.assert_not_called()

if __name__=="__main__":unittest.main()
