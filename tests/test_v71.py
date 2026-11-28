"""V7.1 regression fixtures: real local verifiers, fake remote trainer, no GPU."""
import ast
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"src")]
import ralph_v71 as v
import v71_session as worker

class SessionTests(unittest.TestCase):
    def setUp(self):
        self.base=v.search.source();sha=v.search.digest(self.base)
        old=v.ctl.config("mp_dns","zero",sha,{},epochs=0)
        r={"best_validation_pesq_wb":3.25,"domain_scores":{"external_normal":3.5,"external_stress":2.5},
           "v6":{"checkpoint_sha256":"b"*64,"attempted_epochs":0,"checkpoint_epoch":0,"root_external_domains":[3.5,2.5],"lr":3e-5,"gain":0}}
        self.parent=({"ref":"owner/parent","status":"COMPLETE"},r,old,Path("dummy"))
        self.diag={"phase_gain":.1,"magnitude_gain":2.}
    def plan(self,n=3):
        with patch.object(v.search,"cost",lambda c,e:.2 if c["smoke"] else .5):
            return v.make_plan(self.parent,self.diag,{},self.base,None,4.5,n,False,False,3,200)
    def test_budget_and_distinct_recipes(self):
        p=self.plan()
        self.assertEqual(len(p["candidates"]),3)
        self.assertEqual(len({c["config"]["v56_key"] for c in p["candidates"]}),3)
        self.assertTrue(all(c["config"]["v6_parent"]==p["parent"] for c in p["candidates"]))
        self.assertLessEqual(p["estimate_hours"],p["hours"])
        self.assertIsNone(v.make_plan(self.parent,self.diag,{},self.base,None,.01,3,False,False,3,200))
    def test_parallel_sub_disjoint_and_resume_guard(self):
        import v72_parallel as parallel
        import zipfile
        p=self.plan(3)
        # Reserve continuation family, exactly as the live Colab handoff.
        p["candidates"]=[i for i in p["candidates"] if i["family"]=="continue"]
        self.assertTrue(p["candidates"])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"data"
            files={}
            for split in ("train","validation"):
                q=data/split/"samples.csv";q.parent.mkdir(parents=True);q.write_text(split)
                files["data/"+split+"/samples.csv"]=v.ctl.sha(q)
            z=root/"handoff.zip"
            with zipfile.ZipFile(z,"w") as f:
                f.writestr("handoff_manifest.json",json.dumps({"protocol":"v71-colab-handoff-v1","plan_sha":v.digest(p),"files":files}))
                f.writestr("v71_plan.json",json.dumps(p));f.writestr("v71_base.txt",self.base)
            sub=parallel.reservation(z,data,self.base)
            with patch.object(v.search,"cost",lambda c,e:.2 if c["smoke"] else .5):
                plan=v.make_plan(self.parent,self.diag,{},self.base,None,4.5,3,False,False,3,200,sub=sub)
            self.assertEqual(plan["parallel_sub"],sub["plan_sha"])
            families=[i["family"] for i in plan["candidates"]]
            self.assertEqual(len(families),3);self.assertEqual(len(set(families)),3)
            self.assertNotIn("continue",families)
            folder=root/"run_demo/round_0001/v4_external";folder.mkdir(parents=True)
            st={"ref":"owner/job","status":"SUBMIT_PENDING","v71_session":True,"plan_sha":v.digest(p)}
            (folder/"remote_state.json").write_text(json.dumps(st));(folder/"v71_plan.json").write_text(json.dumps(p))
            with patch.object(v.core,"OUTPUT",root):
                with self.assertRaisesRegex(RuntimeError,"MAIN Kaggle"):parallel.require_sub_pending(sub)
                st["plan_sha"]=v.digest(plan)
                (folder/"remote_state.json").write_text(json.dumps(st));(folder/"v71_plan.json").write_text(json.dumps(plan))
                parallel.require_sub_pending(sub)
    def test_source_preserves_eval(self):
        p=self.plan(1);cfg=p["candidates"][0]["config"]
        src=worker.stage_source(self.base,cfg,Path("/tmp/test"),p["dataset"])
        def funcs(s):return {n.name:ast.dump(n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
        a,b=funcs(self.base),funcs(src)
        for key in ("evaluate","score_waveforms","v6_eval","v6_loaders","v6_parent_path","v6_run"):
            self.assertEqual(a[key],b[key],key)
        work=next(n for n in ast.parse(src).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="WORK" for t in n.targets))
        self.assertIsInstance(work.value,ast.Call)
        self.assertEqual(work.value.func.id,"Path")
        self.assertEqual(ast.literal_eval(work.value.args[0]),str(Path("/tmp/test")))
    def fixture(self,base,cfg,work,dataset,timeout):
        work.mkdir(parents=True);parent=cfg["v6_parent"];sm=cfg["smoke"];n=cfg["epochs"]
        score=3.25 if sm else 3.3 if cfg["v7_recipe"]=="continue" else 3.4
        cp=work/"checkpoint.pt";cp.write_bytes(cfg["v56_key"].encode())
        r={"smoke":sm,"evaluation_protocol":"v4-smoke-fixture-v1" if sm else "v4.2-fixed-dev-v2",
           "validation_files":164,"external_validation_files":64,"best_validation_pesq_wb":score,"external_validation_pesq_wb":3.,
           "joint_score":(score+3.)/2,"robust_score":.5*(score+3.5+2.5)/3+.5*2.5,
           "domain_scores":{"challenge":score,"external_normal":3.5,"external_stress":2.5},"model_family":"mp_dns",
           "epochs_completed":n,"best_epoch":0 if sm else 1,"v56_requested_epochs":n,"epoch_seconds":[60.]*n,
           "zero_shot":{"challenge":3.25,"external_domains":[3.5,2.5]},
           "v6":{"protocol":"v6-max-pesq-v1","mode":"train","parent":parent,"checkpoint_sha256":v.ctl.sha(cp),
                 "attempted_epochs":n,"checkpoint_epoch":0 if sm else 1,"root_external_domains":[3.5,2.5],"start_pesq":3.25,"gain":score-3.25},
           "v7":{"protocol":v.search.VERSION,"recipe":cfg["v7_recipe"],"mutation":cfg["v7_mutation"],"parent":parent,"warmup_updates":0,"warmup_calibration":None},
           **{k:cfg[k] for k in ("v56_key","v56_action","v56_source_sha","sepformer_scale_policies")}}
        for name,vals in (("validation_per_file.csv",[score]*164),("external_validation.csv",[3.5]*32+[2.5]*32),
            ("zero_shot_validation.csv",[3.25]*164),("zero_shot_external.csv",[3.5]*32+[2.5]*32)):
            with (work/name).open("w",encoding="utf-8",newline="") as f:
                w=csv.writer(f);w.writerow(["index","pesq_wb"]);w.writerows(enumerate(vals))
        (work/"result.json").write_text(json.dumps(r),encoding="utf-8")
        return r
    def test_same_session_and_verified_import(self):
        p=self.plan(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            def run(*args):
                calls.append(args[1]["smoke"]);return self.fixture(*args)
            worker.run_session(self.base,p,root,runner=run,clock=lambda:0)
            self.assertEqual(calls,[True,False,True,False])
            self.assertEqual(len(list(root.rglob("best_model.pt"))),1)
            (root/"v71_plan.json").write_text(json.dumps(p));(root/"v71_base.txt").write_text(self.base,encoding="utf-8")
            state={"ref":"owner/session","status":"SUBMIT_PENDING","v71_session":True,"kind":"v71_session"}
            winner=v.import_session(root,state)
            self.assertEqual(winner[2]["best_validation_pesq_wb"],3.4)
            self.assertEqual(state["status"],"COMPLETE");self.assertTrue(state["v6"])
            self.assertEqual(v.ctl.verify(root,state)[0]["best_validation_pesq_wb"],3.4)
            self.assertEqual(len(v.verify_stages(root,state["ref"])[0]),4)
            (root/"v71_trials/trial_00/train/checkpoint.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"checkpoint corrupt"):v.verify_stages(root,state["ref"])
    def test_failed_stage_preserves_winner(self):
        p=self.plan(2);n=0
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def run(*args):
                nonlocal n;n+=1
                if n==3:
                    args[2].mkdir(parents=True);(args[2]/"best_model.pt").write_bytes(b"incomplete")
                    raise RuntimeError("fixture failure")
                return self.fixture(*args)
            worker.run_session(self.base,p,root,runner=run,clock=lambda:0)
            r=json.loads((root/"v71_report.json").read_text())
            self.assertEqual(r["reason"],"stage_error");self.assertEqual(r["winner"],0)
            self.assertEqual(len(list(root.rglob("best_model.pt"))),1)
    def test_resume_polls_same_ref_no_push(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);st={"ref":"owner/session","status":"SUBMIT_PENDING"}
            responses=[SimpleNamespace(returncode=0,stdout="KernelWorkerStatus.COMPLETE",stderr=""),SimpleNamespace(returncode=0,stdout="",stderr="")]
            with patch.object(v.ctl,"command",side_effect=responses) as cmd,patch.object(v,"import_session",return_value="winner"):
                self.assertEqual(v.poll("kaggle",root,st,1,clock=lambda:0,sleep=lambda _:None),"winner")
            self.assertEqual([c.args[1:3] for c in cmd.call_args_list],[("kernels","status"),("kernels","output")])
if __name__=="__main__":unittest.main()
