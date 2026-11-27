"""V5.6 scheduler CPU tests; mock all Kaggle submissions and metric values."""
import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import ralph_v56 as v
import ralph_v4 as v4

SHA="a"*64
def catalog():
    return v.stage_catalog([],25)

def fake_pilot(name,p,low,high,i):
    cfg=v.trial_cfg(p,name,low,42,None,SHA,"pilot")
    st={"kind":"v4_external","status":"COMPLETE","ref":f"owner/job-{i}","v56_key":cfg["v56_key"]}
    fields={k:p[k] for k in ("v52_stage","v53_stage","v54_stage") if k in p}
    r={**fields,"v56_key":cfg["v56_key"],"v56_source_sha":SHA,"v56_action":"pilot","seed":42,"model_family":p["architecture"],"best_validation_pesq_wb":2.+i*.1,"evaluation_protocol":"v4.2-fixed-dev-v2","epoch_seconds":[60.]*low}
    return st,r,cfg

class V56Tests(unittest.TestCase):
    def test_catalog_and_identity(self):
        rows=catalog()
        self.assertEqual(len(rows),26)
        self.assertEqual(len({n for n,*_ in rows}),26)
        self.assertEqual({a for _,_,a,_ in rows},{5,8,12})
        name,p,low,high=rows[0]
        a=v.trial_cfg(p,name,low,42,None,SHA,"pilot")
        b=v.trial_cfg(p,name,low,43,None,SHA,"pilot")
        self.assertNotEqual(a["v56_key"],b["v56_key"])
        self.assertFalse(a["transductive"]);self.assertFalse(a["amp"])

    def test_smoke_then_trial_and_failure_exclusion(self):
        a=v.next_trial([],{},SHA,None,25,25,10)
        self.assertEqual(a["v56_action"],"smoke")
        ledger={a["v56_key"]:{"status":"COMPLETE","kind":"v4_smoke"}}
        b=v.next_trial([],ledger,SHA,None,25,25,10)
        self.assertEqual(b["v56_action"],"pilot")
        self.assertEqual(b["architecture"],a["architecture"])
        ledger[a["v56_key"]]["status"]="ERROR_TERMINAL"
        c=v.next_trial([],ledger,SHA,None,25,25,10)
        self.assertNotEqual(c["architecture"],a["architecture"])
        self.assertIsNone(v.next_trial([],{},SHA,None,25,.01,10))

    def test_promotion_warm_and_ranking(self):
        rows=catalog();records=[];ledger={}
        for i,(name,p,low,high) in enumerate(rows[:4]):
            st,r,cfg=fake_pilot(name,p,low,high,i)
            records.append((st,r));ledger[cfg["v56_key"]]=st
            s=v.smoke_cfg(p["architecture"],SHA)
            ledger[s["v56_key"]]={"status":"COMPLETE","kind":"v4_smoke"}
        result=v.next_trial(records,ledger,SHA,None,25,25,10)
        self.assertEqual(result["v56_action"],"promote")
        self.assertIn(result["architecture"],[rows[2][1]["architecture"],rows[3][1]["architecture"]])
        # Native no-critic pilot can warm start only from its exact checkpoint.
        name,p,low,high=rows[2]
        parent=records[2][0]
        cfg=v.trial_cfg(p,name,high,42,None,SHA,"promote",parent)
        self.assertEqual(cfg["warm_ref"],parent["ref"])
        self.assertAlmostEqual(cfg["lr"],p["lr"]*.5)
        self.assertIsNone(v.trial_cfg(rows[0][1],rows[0][0],25,42,None,SHA,"promote",records[0][0])["warm_ref"])

    def test_quota_and_cost(self):
        with patch.object(v.legacy,"command",return_value="GPU 6.00h 23.62h 30.00h"):
            self.assertAlmostEqual(v.quota_hours("mock"),23.62)
        with patch.object(v.legacy,"command",return_value="GPU 0h nanh 30h"):
            with self.assertRaises(RuntimeError): v.quota_hours("mock")
        name,p,low,high=catalog()[0]
        cfg=v.trial_cfg(p,name,low,42,None,SHA,"pilot")
        self.assertGreater(v.cost_hours(cfg,[]),0)
        self.assertGreater(v.cost_hours(cfg,[]),v.cost_hours(cfg,[({"kind":"v4_external"},{"model_family":p["architecture"],"evaluation_protocol":"v4.2-fixed-dev-v2","epoch_seconds":[60.]})]))

    def test_recovery_identity(self):
        state={"v56_key":"key","v56_source_sha":SHA,"v56_action":"smoke","architecture":"temporal","epochs":2}
        r={"v56_key":"key","v56_source_sha":SHA,"v56_action":"smoke","model_family":"temporal","v56_requested_epochs":2,"smoke":True}
        v4.verify_v56_metadata(r,state)
        with self.assertRaises(ValueError): v4.verify_v56_metadata({**r,"v56_key":"wrong"},state)
        with self.assertRaises(ValueError): v4.verify_v56_metadata({**r,"smoke":False},state)

    def test_no_gpu_dry_run_and_permission_gate(self):
        args=argparse.Namespace(execute=False,epochs=25,allow_public_assets=False)
        with patch.object(v.core,"no_uncertain_prior_jobs"),patch.object(v4,"history",return_value=[]),patch.object(v4,"submit",side_effect=AssertionError("No GPU")):
            v.run(args)
            args.execute=True
            with self.assertRaises(RuntimeError): v.run(args)

    def test_round_limit_serial_and_cleanup(self):
        args=argparse.Namespace(execute=True,allow_public_assets=True,epochs=25,max_rounds=1,budget_hours=4.,reserve_hours=2.,max_job_minutes=180,public_cache_ref=None,data_root=None)
        with tempfile.TemporaryDirectory() as d:
            output=Path(d);calls=[]
            def submit(kaggle,source,tag,i,kind,cfg):
                calls.append((kind,cfg))
                self.assertTrue((output/"active.lock").exists())
                return {"ref":"owner/job-1","kind":kind},{"best_validation_pesq_wb":2.5}
            with patch("v572_scale_audit.require_report",return_value={}),patch("v571_preflight.require_report",return_value={"families":{},"cmgan_root":str(output),"sepformer_scale_policies":{}}),patch.object(v.core,"OUTPUT",output),patch.object(v.core,"no_uncertain_prior_jobs"),patch.object(v4,"history",return_value=[]),patch.object(v4,"find_public_cache",return_value=None),patch.object(v.shutil,"which",return_value="mock"),patch.object(v,"quota_hours",return_value=23.),patch.object(v4,"submit",submit):
                v.run(args)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][0],"v4_public")
            self.assertFalse((output/"active.lock").exists())
            self.assertEqual(len(list(output.glob("run_*/v56_efficiency.jsonl"))),1)

if __name__=="__main__": unittest.main()
