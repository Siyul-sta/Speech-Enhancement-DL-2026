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
import dashboard_v4

class ControllerTests(unittest.TestCase):
    def rows(self,path,count,value):
        with path.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["index","pesq_wb"])
            w.writerows((i,value) for i in range(count))
    def test_plan_rotation(self):
        plans=[v4.propose_config([],n,2,None) for n in range(24)]
        self.assertEqual({p["architecture"] for p in plans},{"residual","temporal","metricgan","moe","composed","sepformer_dns_transfer","sepformer_wham_transfer"})
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
    def test_public_shared_evaluator(self):
        public=(ROOT/"src/kaggle_v4_public.py").read_text(encoding="utf-8")
        external=(ROOT/"src/kaggle_v4_external.py").read_text(encoding="utf-8")
        combined=v4.public_with_external(public,external)
        self.assertIn("class SyntheticPairs",combined)
        self.assertIn("def public_assets",combined)
        compile(combined,"combined.py","exec")
    def test_empty_dashboard(self):
        with tempfile.TemporaryDirectory() as d:
            state=dashboard_v4.snapshot(Path(d))
            self.assertIsNone(state["best"])
            self.assertEqual(state["experiments"],[])


    def test_dashboard_remote_status_and_ranking(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); run=root/"run_v4test"
            for i,kind,score,transductive in [(1,"v4_public",2.2,False),(2,"v4_external",2.1,False),(3,"v4_smoke",4.0,False),(4,"v4_external",4.1,True),(5,"v4_cache",None,False)]:
                path=run/f"round_{i:04d}"/kind
                path.mkdir(parents=True)
                state={"status":"SUBMIT_PENDING" if score is None else "COMPLETE","kind":kind,"ref":f"owner/job-{i}"}
                (path/"remote_state.json").write_text(json.dumps(state),encoding="utf-8")
                if score is not None:
                    (path/"result.json").write_text(json.dumps({"robust_score":score,"transductive_noisy":transductive,"evaluation_protocol":"v4.2-fixed-dev-v2"}),encoding="utf-8")
            with patch.object(dashboard_v4,"remote_status",return_value={"status":"RUNNING","error":None}) as remote:
                state=dashboard_v4.snapshot(root,poll_remote=True)
            remote.assert_called_once_with("owner/job-5")
            self.assertEqual(state["remote"]["status"],"RUNNING")
            self.assertEqual(state["active"]["status"],"SUBMIT_PENDING")
            self.assertEqual(state["best"]["kind"],"v4_public")
            self.assertEqual(state["counts"]["ranked"],2)
            self.assertEqual(json.loads((run/"round_0005"/"v4_cache"/"remote_state.json").read_text())["status"],"SUBMIT_PENDING")


    def test_dashboard_client_disconnect(self):
        from unittest.mock import Mock
        for error in (ConnectionAbortedError(10053,"aborted"),ConnectionResetError(10054,"reset"),BrokenPipeError("closed")):
            with self.subTest(error=type(error).__name__):
                handler=Mock()
                handler.path="/"
                handler.close_connection=False
                handler.wfile.write.side_effect=error
                dashboard_v4.Handler.do_GET(handler)
                self.assertTrue(handler.close_connection)
        handler=Mock()
        handler.path="/"
        handler.wfile.write.side_effect=ValueError("unexpected")
        with self.assertRaises(ValueError):
            dashboard_v4.Handler.do_GET(handler)


    def test_dashboard_observed_telemetry(self):
        logs='\n'.join([
            'epoch=1 loss=0.12 val_pesq=2.1 best=2.1',
            'V4_TELEMETRY '+json.dumps({"event":"data","dataset":"train","stage":"download","bytes":100,"total_bytes":200,"elapsed_s":10}),
            'V4_TELEMETRY '+json.dumps({"event":"batch","epoch":2,"epochs":3,"batch":10,"batches":20,"loss":0.04,"gpu_alloc_mib":512,"gpu_total_mib":15360}),
            'epoch=2 loss=0.05 val_pesq=2.2 best=2.2',
            'V4_TELEMETRY '+json.dumps({"event":"epoch","epoch":2,"loss":0.05,"challenge":2.2,"external":1.9,"robust":1.8}),
            'V4_TELEMETRY malformed',
        ])
        t=dashboard_v4.parse_logs(logs)
        self.assertEqual(t["download"]["bytes"],100)
        self.assertEqual(t["batch"]["batch"],10)
        self.assertEqual(t["epochs"][0]["challenge"],2.1)
        self.assertEqual(t["epochs"][1]["external"],1.9)
        self.assertEqual(t["activity"],"epoch")
        self.assertIsNone(dashboard_v4.parse_logs("unrelated log")["download"])

    def test_dashboard_log_read_only_cache_and_failure(self):
        from types import SimpleNamespace
        with patch.dict(dashboard_v4.LOG_CACHE,{},clear=True),patch.object(dashboard_v4.shutil,"which",return_value="mock"),patch.object(dashboard_v4.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout="epoch=1 loss=0.1 val_pesq=2.2 best=2.2",stderr="")) as call:
            first=dashboard_v4.remote_logs("owner/test")
            self.assertEqual(first["telemetry"]["epochs"][0]["challenge"],2.2)
            self.assertEqual(dashboard_v4.remote_logs("owner/test"),first)
            call.assert_called_once()
            self.assertEqual(call.call_args.args[0],["mock","kernels","logs","owner/test"])
            dashboard_v4.LOG_CACHE["owner/test"]=(float("-inf"),first)
            call.return_value=SimpleNamespace(returncode=1,stdout="",stderr="network")
            failed=dashboard_v4.remote_logs("owner/test")
            self.assertIsNotNone(failed["error"])
            self.assertEqual(failed["lines"],first["lines"])
            self.assertEqual(dashboard_v4.remote_logs("bad ref")["lines"],[])

    def test_dashboard_remote_query_is_read_only(self):
        from types import SimpleNamespace
        with patch.dict(dashboard_v4.REMOTE_CACHE,{},clear=True),patch.object(dashboard_v4.shutil,"which",return_value="mock"),patch.object(dashboard_v4.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout="KernelWorkerStatus.RUNNING",stderr="")) as call:
            self.assertEqual(dashboard_v4.remote_status("owner/test")["status"],"RUNNING")
            self.assertEqual(dashboard_v4.remote_status("owner/test")["status"],"RUNNING")
            call.assert_called_once()
            self.assertEqual(call.call_args.args[0],["mock","kernels","status","owner/test"])

    def test_public_cache_cpu_submission(self):
        from types import SimpleNamespace
        source=(ROOT/"src/kaggle_v4_external.py").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def fake(argv,**kwargs):
                if "push" in argv:
                    job=Path(argv[-1])
                    m=json.loads((job/"kernel-metadata.json").read_text())
                    self.assertEqual(m["enable_gpu"],"false")
                    self.assertEqual(m["dataset_sources"],[])
                    self.assertEqual(m["kernel_sources"],[])
                    code=(job/"run.py").read_text()
                    self.assertNotIn(v4.DATASET,code)
                    compile(code,"cache.py","exec")
                    return SimpleNamespace(stdout="",stderr="")
                return SimpleNamespace(stdout="KernelWorkerStatus.COMPLETE",stderr="")
            with patch.object(v4,"ROOT",root),patch.object(v4.core,"OUTPUT",root/"runs"),patch.object(v4.subprocess,"run",fake):
                ref=v4.build_public_cache("mock",source,"v4test")
                self.assertEqual(v4.find_public_cache(),ref)
                self.assertEqual(v4.history(),[])
    def test_loop_with_mocked_remote(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"src").mkdir()
            for p in (ROOT/"src").glob("kaggle_v4*.py"): shutil.copy(p,root/"src"/p.name)
            calls=[]
            def fake(kaggle,source,tag,i,kind,config=None):
                calls.append((kind,config))
                return {"kind":kind,"ref":"owner/job-"+str(i)},{"joint_score":2.4,"robust_score":2.3,"best_validation_pesq_wb":2.4,"external_validation_pesq_wb":2.4,"model_family":(config or {}).get("architecture","residual")}
            with patch.object(v4,"ROOT",root),patch.object(v4.core,"OUTPUT",root/"runs"),patch.object(v4.shutil,"which",return_value="mock"),patch.object(v4.legacy,"require_quota"),patch.object(v4,"submit",fake),patch.object(sys,"argv",["ralph_v4","--loop","--execute","--max-rounds","3","--epochs","2"]):
                v4.main()
            self.assertEqual([k for k,c in calls],["v4_public","v4_external","v4_external","v4_external"])
            self.assertFalse((root/"runs"/"active.lock").exists())
    def test_public_audit_packaging(self):
        public=(ROOT/"src/kaggle_v4_public.py").read_text(encoding="utf-8")
        external=(ROOT/"src/kaggle_v4_external.py").read_text(encoding="utf-8")
        result=v4.audit_with_public(external,public)
        self.assertIn("def public_recipe",result)
        compile(result,"audit.py","exec")


    def test_smoke_isolated_from_research(self):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/"src").mkdir()
            for p in (ROOT/"src").glob("kaggle_v4*.py"): shutil.copy(p,root/"src"/p.name)
            calls=[]
            def fake(kaggle,source,tag,i,kind,config=None):
                calls.append((kind,config))
                folder=v4.core.OUTPUT/("run_"+tag)/"round_0001"/kind
                folder.mkdir(parents=True)
                (folder/"remote_state.json").write_text(json.dumps({"kind":kind,"status":"COMPLETE","ref":"owner/test"}))
                return {},{}
            with patch.object(v4,"ROOT",root),patch.object(v4.core,"OUTPUT",root/"runs"),patch.object(v4.shutil,"which",return_value="mock"),patch.object(v4.legacy,"require_quota"),patch.object(v4,"submit",fake),patch.object(sys,"argv",["ralph_v4","--smoke","--execute"]):
                v4.main()
                self.assertEqual(v4.history(),[])
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][0],"v4_smoke")
            self.assertEqual(calls[0][1]["epochs"],2)
            self.assertEqual(calls[0][1]["external_size"],16)
            self.assertFalse((root/"runs"/"active.lock").exists())

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
