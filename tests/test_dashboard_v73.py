"""V7.3 read-only fixture checks; no GPU, Drive or credentials."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import dashboard_v73 as d

class ParallelDashboardTests(unittest.TestCase):
    def test_missing_and_partial(self):
        self.assertEqual(d.colab_snapshot(None)["status"],"not_configured")
        with tempfile.TemporaryDirectory() as t:
            self.assertEqual(d.colab_snapshot(Path(t))["status"],"waiting_files")
    def test_history_and_score_gating(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            p={"protocol":"v71-plan-v1","hours":5,"parent":{"score":3.17},"candidates":[{"config":{"v7_recipe":"lr_half","v56_key":"test","epochs":3,"lr":1.5e-5},"smoke":{"epochs":1}}]}
            r={"protocol":"v71-session-v1","plan_sha":d.digest(p),"reason":"running","records":[{"index":0,"smoke":"complete","train":"not_started"}]}
            (root/"v71_plan.json").write_text(json.dumps(p))
            (root/"v71_report.json").write_text(json.dumps(r))
            f=root/"v71_trials/trial_00/train";f.mkdir(parents=True)
            (f/"history.csv").write_text("epoch,train_loss,validation_pesq_wb,best_pesq_wb,stale_epochs\n1,.1,3.4,3.4,0\n")
            (f/"result.json").write_text(json.dumps({"evaluation_protocol":"v4.2-fixed-dev-v2","smoke":False,"best_validation_pesq_wb":3.4,"v56_key":"test","v7":{"recipe":"lr_half"}}))
            (f/"checkpoint.pt").write_bytes(b"fixture")
            with (f/"validation_per_file.csv").open("w",newline="") as file:
                w=csv.writer(file);w.writerow(["index","pesq_wb"]);w.writerows((i,3.4) for i in range(164))
            a=d.colab_snapshot(root)
            self.assertEqual(a["status"],"saved_running");self.assertEqual(a["scores"],[])
            self.assertEqual(a["candidates"][0]["stages"]["train"]["history"][0]["challenge"],3.4)
            r["records"][0].update(train="complete",score=3.4)
            (root/"v71_report.json").write_text(json.dumps(r))
            self.assertEqual(d.colab_snapshot(root)["scores"][0]["score"],3.4)
            (f/"validation_per_file.csv").write_text("index,pesq_wb\n0,3.4\n")
            self.assertEqual(d.colab_snapshot(root)["scores"],[])
            r["plan_sha"]="wrong";(root/"v71_report.json").write_text(json.dumps(r))
            self.assertEqual(d.colab_snapshot(root)["status"],"inconsistent")
    def test_scoped_logs(self):
        lines=['V71_STAGE {"recipe":"a","stage":"smoke"}','V4_TELEMETRY {"event":"epoch","epoch":1,"challenge":4.8}',
               'V71_STAGE {"recipe":"a","stage":"train"}','V4_TELEMETRY {"event":"batch","epoch":1,"batch":2,"batches":10,"loss":0.2}']
        x=d.scoped_logs({"lines":lines})
        self.assertEqual(x["stage"]["stage"],"train");self.assertEqual(x["telemetry"]["epochs"],[])
        self.assertEqual(x["telemetry"]["batch"]["batch"],2)
    def test_page(self):
        html=d.page();self.assertIn("id=\"colabChart\"",html)
        self.assertIn("poll('/parallel'",html);self.assertIn("poll('/api'",html)
        self.assertNotIn("innerHTML",html);self.assertNotIn("method:'POST'",html)
        self.assertLess(html.index("const originalRenderLogs"),html.index("poll('/logs',"))
if __name__=="__main__":unittest.main()
