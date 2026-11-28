"""V6.11 scheduler tests with synthetic ledger, no GPU or private audio."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import ralph_v611 as loop
import ralph_v6 as ctl
import ralph_v610 as focus
import v6_refine_analysis as analysis

class UnifiedTests(unittest.TestCase):
    def setUp(self):
        self.sha=focus.digest(focus.source())
        self.cfg=ctl.config("mp_dns","train",focus.digest(ctl.source()),{},epochs=3)
        self.r={"best_validation_pesq_wb":3.25,"smoke":False,"best_epoch":2,
            "domain_scores":{"external_normal":3.5,"external_stress":2.5},
            "v6":{"checkpoint_sha256":"b"*64,"attempted_epochs":3,"checkpoint_epoch":2,
                  "root_external_domains":[3.5,2.5],"lr":2e-5,"gain":.07}}
        self.st={"ref":"owner/job","status":"COMPLETE"}
        self.parent=(self.st,self.r,self.cfg,Path("dummy"))
        self.diag={"phase_gain":.5,"magnitude_gain":.2}
    def test_success_exploits_and_fallback_tie_preserves_parent(self):
        self.assertEqual(loop.ranked_recipes(self.parent,self.diag,{},200)[0],("continue",0))
        fallback={**self.r,"best_epoch":0,"v6":{**self.r["v6"],"attempted_epochs":4,"gain":0}}
        e={"a":self.parent,"b":({"ref":"owner/new","status":"COMPLETE"},fallback,self.cfg,Path("new"))}
        self.assertEqual(loop.champion(e)[0],self.st)
    def test_stagnation_explores_phase_and_skips_completed_hypothesis(self):
        self.r["v6"]["gain"]=0
        self.assertEqual(loop.ranked_recipes(self.parent,self.diag,{},200)[0],("phase_derivatives",0))
        c=loop.build(self.parent,"phase_derivatives",0,self.sha,None,1,2048,512)
        e={c["v56_key"]:({"status":"COMPLETE"},self.r,c,Path("new"))}
        self.assertEqual(loop.ranked_recipes(self.parent,self.diag,e,200)[0],("hard_noise_first",0))
        self.assertEqual(loop.ranked_recipes(self.parent,self.diag,e,3),[])
    def test_terminal_failure_counts_and_lr_retry_is_bounded(self):
        for i in range(2):
            c=loop.build(self.parent,"continue",i,self.sha,None,1,2048,512)
            if i==0:e={}
            e[str(i)]=({"status":"ERROR_TERMINAL"},None,c,Path("x"))
        self.assertNotIn("continue",[x[0] for x in loop.ranked_recipes(self.parent,self.diag,e,200)])
    def test_critic_three_epochs_and_exact_parent(self):
        with self.assertRaises(ValueError):loop.build(self.parent,"critic",0,self.sha,None,1,2048,512)
        c=loop.build(self.parent,"critic",0,self.sha,None,3,2048,512)
        self.assertTrue(c["v6_critic"]);self.assertEqual(c["v6_crop"],48000)
        self.assertEqual(c["v6_parent"]["sha256"],"b"*64)
        s=loop.build(self.parent,"critic",0,self.sha,None,3,2048,512,True)
        self.assertEqual(s["epochs"],1);self.assertIsNone(s["warm_ref"])
    def test_budget_checks_smoke_plus_trial_and_explicit_mix_reduction(self):
        def cost(c,e):return .2 if c["smoke"] else c["epochs"]*(.2+c["v6_mix"]/2048)
        with patch.object(loop,"estimate",cost):
            self.assertIsNone(loop.pick(self.parent,self.diag,{},self.sha,None,.1,6,200,3,True))
            self.assertIsNone(loop.pick(self.parent,self.diag,{},self.sha,None,.8,6,200,3,False))
            p=loop.pick(self.parent,self.diag,{},self.sha,None,.8,6,200,3,True)
            self.assertLessEqual(p["hours"],.8)
            self.assertTrue(p["compound_data_change"])
            self.assertLess(p["config"]["v6_mix"],2048)
    def test_existing_smoke_reused_and_failed_smoke_not_retried(self):
        with patch.object(loop,"estimate",lambda c,e:.3 if c["smoke"] else .4):
            p=loop.pick(self.parent,self.diag,{},self.sha,None,.7,6,200,1,False)
            s=p["smoke"]
            e={s["v56_key"]:({"status":"COMPLETE"},{"smoke":True},s,Path("s"))}
            q=loop.pick(self.parent,self.diag,e,self.sha,None,.4,6,200,1,False)
            self.assertTrue(q["reuse_smoke"])
            e={s["v56_key"]:({"status":"ERROR_TERMINAL"},None,None,Path("s"))}
            self.assertIsNone(loop.pick(self.parent,self.diag,e,self.sha,None,.4,6,200,1,False))
    def test_same_family_final_shortlist(self):
        a=self.parent
        b=({"ref":"second"}, {**self.r,"v6":{**self.r["v6"],"checkpoint_sha256":"c"*64}},self.cfg,Path("b"))
        cfg={**self.cfg,"architecture":"rose_cd"}
        c=({"ref":"third"}, {**self.r,"v6":{**self.r["v6"],"checkpoint_sha256":"d"*64}},cfg,Path("c"))
        self.assertEqual(analysis.shortlist([a,b,c],2,False),[a,b])
        self.assertEqual(analysis.shortlist([a,b,c],2,True),[a,c])
if __name__=="__main__":unittest.main()
