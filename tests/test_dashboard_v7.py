"""Read-only dashboard fixtures: no torch, Kaggle or private data needed."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import dashboard_v7 as d

class DashboardV7Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.remote=self.root/"remote";self.research=self.root/"research"
        self.remote.mkdir();self.research.mkdir()
    def tearDown(self):self.tmp.cleanup()
    def make(self,tag,number,score=None,smoke=False,recipe=None,status="COMPLETE",parent=3.1):
        p=self.remote/("run_v4"+tag)/f"round_{number:04d}"/("v4_smoke" if smoke else "v4_external");p.mkdir(parents=True)
        ref=f"owner/job-{tag}-{number}"
        state={"ref":ref,"kind":p.name,"status":status,"v6":True}
        c={"architecture":"mp_dns","epochs":3,"smoke":smoke,"v6_parent":{"score":parent},"v6_crop":16000,"v6_mix":2048,"v6_public_mix":512}
        if recipe:c["v7_recipe"]=recipe
        (p/"remote_state.json").write_text(json.dumps(state))
        (p/"source.py").write_text("EXPERIMENT_CONFIG = "+repr(c))
        if score is not None:
            r={"model_family":"mp_dns","smoke":smoke,"best_validation_pesq_wb":score,
               "evaluation_protocol":"v4-smoke-fixture-v1" if smoke else "v4.2-fixed-dev-v2",
               "v6":{"checkpoint_sha256":"a"*64,"metric_steps":0},"epoch_seconds":[300],"epochs_completed":1}
            (p/"result.json").write_text(json.dumps(r))
            with (p/"validation_per_file.csv").open("w",newline="") as f:
                w=csv.writer(f);w.writerow(["index","pesq_wb"]);w.writerows((i,score) for i in range(164))
        return p,ref
    def test_empty_and_catalog(self):
        x=d.snapshot(self.remote,self.research)
        self.assertIsNone(x["best"]);self.assertIsNone(x["active"])
        self.assertEqual(len(x["strategies"]),17)
        json.dumps(x,allow_nan=False)
    def test_exclude_smoke_and_invalid_rows(self):
        self.make("001",1,3.1757);self.make("002",1,4.7,True,"continue")
        p,_=self.make("003",1,4.2,False,"continue")
        (p/"validation_per_file.csv").write_text("index,pesq_wb\n0,4.2\n")
        x=d.snapshot(self.remote,self.research)
        self.assertEqual(x["best"]["score"],3.1757);self.assertEqual(len(x["history"]),1)
        self.assertEqual(x["counts"]["v7_trials"],1)
        a=next(s for s in x["strategies"] if s["name"]=="continue")
        self.assertEqual(a["unverified"],1);self.assertEqual(a["completed"],0)
    def test_cross_run_pending_and_strategy(self):
        p,ref=self.make("001",7,status="SUBMIT_PENDING")
        self.make("002",1,3.3,recipe="spectral_balance")
        self.make("002",2,3.1,recipe="spectral_balance")
        self.make("002",3,recipe="spectral_balance",status="ERROR_TERMINAL")
        with patch.object(d.old,"remote_status",return_value={"status":"RUNNING","error":None}) as status:
            x=d.snapshot(self.remote,self.research,True)
            status.assert_called_once_with(ref)
        self.assertEqual(x["active"]["ref"],ref);self.assertEqual(x["phase"],"legacy_pending")
        self.assertEqual(x["best"]["score"],3.3);self.assertEqual(x["history"][-1]["best"],3.3)
        a=next(s for s in x["strategies"] if s["name"]=="spectral_balance")
        self.assertEqual((a["attempts"],a["completed"],a["improved"],a["failed"]),(3,2,1,1))
    def test_journal_and_final(self):
        r=self.research/"v7";r.mkdir()
        (r/"run_v4001.json").write_text(json.dumps({"protocol":"v7-controller-v1","rounds":1,"cycle":1,"jobs":2,
            "stop_reason":"budget","events":[{"recipe":"continue","selection":{"priority":.1}}],"cycles":[{"end_quota":{"remaining":1}}]}))
        (self.remote/"active.lock").write_text("123")
        q=self.research/"v7_final_001";q.mkdir();(q/"report.json").write_text(json.dumps({"protocol":"v6-refine-v1","count":164,"winner":"single","winner_pesq":3.2}))
        x=d.snapshot(self.remote,self.research)
        self.assertEqual(x["phase"],"possible_quota_wait");self.assertEqual(x["event"]["recipe"],"continue")
        self.assertEqual(x["final"]["pesq"],3.2)
        (self.remote/"active.lock").unlink()
        self.assertEqual(d.snapshot(self.remote,self.research)["phase"],"journal_idle")
    def test_quota_parser_and_stale_cache(self):
        raw="resource used remaining total refreshAt\nGPU 4.73h 25.27h 30.00h 2026-12-05T00:00:00+00:00\n"
        self.assertEqual(d.parse_quota(raw)["remaining"],25.27)
        with self.assertRaises(ValueError):d.parse_quota("GPU 3h nan 30h")
        with patch.object(d,"QUOTA_CACHE",None),patch.object(d.shutil,"which",return_value="kaggle"),patch.object(d.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout=raw)) as run:
            self.assertFalse(d.live_quota()["stale"]);self.assertFalse(d.live_quota()["stale"]);run.assert_called_once()
            d.QUOTA_CACHE=(d.time.monotonic()-61,d.QUOTA_CACHE[1])
            run.side_effect=OSError("offline")
            q=d.live_quota();self.assertTrue(q["stale"]);self.assertEqual(q["remaining"],25.27)
    def test_log_uses_unresolved_older_job(self):
        _,ref=self.make("001",7,status="SUBMIT_PENDING");self.make("002",1,3.3)
        with patch.object(d.old,"remote_logs",return_value={"ref":ref}) as call:
            self.assertEqual(d.log_payload(self.remote)["ref"],ref);call.assert_called_once_with(ref)
    def test_page_read_only_endpoints(self):
        html=(ROOT/"tools/dashboard_v7.html").read_text(encoding="utf-8")
        for path in ("/api","/quota","/logs"):self.assertIn("'"+path+"'",html)
        self.assertNotIn("innerHTML",html)
        self.assertNotIn("method:'POST'",html)
if __name__=="__main__":unittest.main()
