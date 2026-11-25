import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import ralph_v4 as v4
import v4_research_catalog as catalog
import v4_model_policy as policy

class ControllerTests(unittest.TestCase):
    def rows(self,path,count,value):
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["index","pesq_wb"])
            w.writerows((i,value) for i in range(count))
    def test_plan_rotation(self):
        plans=[v4.propose_config([],n,2,None) for n in range(24)]
        self.assertEqual({p["architecture"] for p in plans},{"residual","temporal","metricgan","moe","composed"})
        self.assertEqual({p["data_profile"] for p in plans},set(catalog.DATASETS))
        self.assertTrue(any(p["distill"] for p in plans))
        self.assertTrue(any(p["loss_mode"]=="composed" for p in plans))
    def test_network_grammar(self):
        spec=catalog.default_network(1)
        self.assertEqual(catalog.validate_network(spec),spec)
        spec["blocks"][0]["type"]="arbitrary"
        with self.assertRaises(ValueError): catalog.validate_network(spec)
    def test_robust_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            self.rows(path/"validation_per_file.csv",164,3.0)
            self.rows(path/"external_validation.csv",64,2.0)
            (path/"best_model.pt").write_bytes(b"test")
            r={"validation_files":164,"best_validation_pesq_wb":3.0,"external_validation_files":64,"external_validation_pesq_wb":2.0,"joint_score":2.5,"evaluation_protocol":"v4.2-fixed-dev-v2","domain_scores":{"challenge":3.0,"external_normal":2.0,"external_stress":2.0},"robust_score":0.5*7/3+1}
            (path/"result.json").write_text(json.dumps(r))
            self.assertAlmostEqual(v4.rank_score(v4.verify(path,"v4_external")),r["robust_score"])
            r["robust_score"]+=.1
            (path/"result.json").write_text(json.dumps(r))
            with self.assertRaises(ValueError): v4.verify(path,"v4_external")
    def test_loss_bounds(self):
        self.assertEqual(catalog.validate_loss({"wave":1,"spectral":0,"sisdr":.2})["wave"],1)
        for bad in ({"wave":0,"spectral":0,"sisdr":1},{"wave":float("nan"),"spectral":1,"sisdr":0},{"wave":True,"spectral":1,"sisdr":0}):
            with self.assertRaises(ValueError): catalog.validate_loss(bad)
    def test_discovery(self):
        def get(url):
            if "arxiv" in url:
                return b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>paper</id><title>Test</title><summary>Abstract</summary></entry></feed>'
            return json.dumps([{"id":"speechbrain/metricgan-plus-voicebank","tags":["license:apache-2.0"]},{"id":"secret","private":True}]).encode()
        result=catalog.discover(get)
        self.assertEqual(len(result["models"]),1)
        self.assertTrue(result["models"][0]["admitted"])
        self.assertEqual(result["papers"][0]["title"],"Test")
    def test_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            self.rows(path/"validation_per_file.csv",164,2.5)
            self.rows(path/"external_validation.csv",64,3.0)
            (path/"best_model.pt").write_bytes(b"test")
            r={"validation_files":164,"best_validation_pesq_wb":2.5,"external_validation_files":64,"external_validation_pesq_wb":3.0,"joint_score":2.75}
            (path/"result.json").write_text(json.dumps(r))
            self.assertEqual(v4.verify(path,"v4_external")["joint_score"],2.75)
            self.rows(path/"external_validation.csv",64,4.0)
            with self.assertRaises(ValueError): v4.verify(path,"v4_external")
    def test_audit_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            self.rows(path/"audit_per_file.csv",128,2.1)
            (path/"result.json").write_text(json.dumps({"audit_files":128,"audit_pesq":2.1}))
            self.assertEqual(v4.verify(path,"v4_audit")["audit_files"],128)
    def test_submit_mock_no_network(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def fake(kaggle,job,ref,folder,wait,status_hook=None):
                metadata=json.loads((job/"kernel-metadata.json").read_text())
                self.assertEqual(metadata["is_private"],"true")
                self.assertEqual(metadata["dataset_sources"],[v4.DATASET])
                self.rows(folder/"validation_per_file.csv",164,2.5)
                (folder/"result.json").write_text(json.dumps({"validation_files":164,"best_validation_pesq_wb":2.5}))
            with patch.object(v4,"ROOT",root),patch.object(v4.core,"OUTPUT",root/"runs"),patch.object(v4.core,"poll_adaptive_kernel",fake):
                st,r=v4.submit("mock",'DATASET_SLUG = "PRIVATE_DATASET_SLUG"\n',"v4test",1,"v4_public")
                self.assertEqual(st["status"],"COMPLETE")
                self.assertEqual(r["best_validation_pesq_wb"],2.5)
    def test_source_compiles(self):
        for p in (ROOT/"src").glob("kaggle_v4*.py"):
            compile(p.read_text(encoding="utf-8"),str(p),"exec")
        for p in ("infer_v4.py","ralph_v4.py","v4_model_policy.py","v4_research_catalog.py"):
            f=ROOT/"tools"/p
            compile(f.read_text(encoding="utf-8"),str(f),"exec")
    def test_generated_policy(self):
        code="class GeneratedEnhancer(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.conv=nn.Conv1d(1,1,3,padding=1)\n    def forward(self,x):\n        return self.conv(x.unsqueeze(1)).squeeze(1)\n"
        self.assertEqual(len(policy.validate_model(code)),64)
        with self.assertRaises(ValueError): policy.validate_model(code.replace("return self.conv(x.unsqueeze(1)).squeeze(1)","return torch.load('bad')"))

if __name__=="__main__":
    unittest.main()
