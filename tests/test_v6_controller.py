"""V6 controller/source tests (no network, no GPU, no private data)."""
import ast
import csv
import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import ralph_v6 as v
import ralph_v2 as core

def args(**kw):
    d=dict(families=list(v.FAMILIES),pilot_epochs=3,mix=2048,public_mix=512,max_mix=32768,
           max_epochs=200,chunk_epochs=10,min_pilots=3,korean_limit=256,enable_critic=True,enable_cascade=True)
    d.update(kw);return SimpleNamespace(**d)

class V6Controller(unittest.TestCase):
    def test_generated_source_preserves_evaluator_and_model_classes(self):
        base=(ROOT/"src/kaggle_v4_external.py").read_text(encoding="utf-8");code=v.source()
        compile(code,"v6.py","exec")
        a={n.name:ast.dump(n) for n in ast.parse(base).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        b={n.name:ast.dump(n) for n in ast.parse(code).body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        for k in ("SpeechPairs","evaluate","score_waveforms","CMGANTransfer","SepformerTransfer","ResidualPhaseUNet","public_assets"):
            self.assertEqual(a[k],b[k],k)
        for k in ("V6MPNet","V6CrossMix","V6Refiner","v6_run","v6_split"):
            self.assertIn(k,b)

    def test_dry_plan_and_budget(self):
        a=args()
        self.assertIsNone(v.choose({},"a"*64,{},None,.001,6,a))
        c=v.choose({},"a"*64,{},None,20,6,a)
        self.assertEqual((c["architecture"],c["v6_mode"]),("mp_dns","zero"))
        self.assertFalse(c["smoke"])
        self.assertNotEqual(c["v56_key"],v.config("mp_vb","zero","a"*64,{})["v56_key"])

    def test_cfg_has_safe_policy(self):
        c=v.config("sepformer_dns_transfer","train","a"*64,{"sepformer_dns_transfer":"rms_025"},epochs=5)
        self.assertEqual(c["sepformer_scale_policies"],{"sepformer_dns_transfer":"rms_025"})
        self.assertEqual(c["v6_crop"],16000)
        self.assertEqual(c["v6_transfer"],"progressive")
        d=v.config("complex_residual","train","a"*64,{},epochs=5,critic=True)
        self.assertEqual(d["v6_crop"],48000)
        self.assertNotEqual(c["v56_key"],d["v56_key"])

    def test_network_retry_no_push(self):
        # Simulated DNS loss, active job, then complete. Only status/output.
        responses=[SimpleNamespace(returncode=1,stdout="",stderr="DNS failure"),
                   SimpleNamespace(returncode=0,stdout="KernelWorkerStatus.RUNNING",stderr=""),
                   SimpleNamespace(returncode=0,stdout="KernelWorkerStatus.COMPLETE",stderr=""),
                   SimpleNamespace(returncode=0,stdout="",stderr="")]
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"run"/"round"/"v4_external";f.mkdir(parents=True)
            state={"ref":"owner/job","status":"SUBMIT_PENDING"}
            with patch.object(v,"command",side_effect=responses) as cli,patch.object(v,"verify",return_value=({"ok":1},{})):
                r=v.poll("kaggle",f,state,999,interval=0,clock=lambda:0,sleep=lambda t:None)
            self.assertEqual(r,{"ok":1})
            self.assertEqual(json.loads((f/"remote_state.json").read_text())["status"],"COMPLETE")
            self.assertEqual([c.args[1:3] for c in cli.call_args_list],[("kernels","status")]*3+[("kernels","output")])

    def test_timeout_keeps_unresolved(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"run"/"round"/"v4_external";f.mkdir(parents=True)
            st={"ref":"owner/job","status":"SUBMIT_PENDING"}
            with patch.object(v,"command",return_value=SimpleNamespace(returncode=1,stdout="",stderr="DNS")):
                with self.assertRaises(TimeoutError):
                    v.poll("kaggle",f,st,1,interval=0,clock=iter([0,2]).__next__,sleep=lambda t:None)
            self.assertEqual(st["status"],"SUBMIT_PENDING")

    def test_configured_smoke_epoch_and_same_ref_recovery(self):
        # Real on-disk verifier fixture; do not mock verify or loosen score/hash checks.
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/"run"/"round"/"v4_smoke";folder.mkdir(parents=True)
            cfg=v.config("mp_dns","train","a"*64,{},epochs=1,smoke=True,mix=16,public_mix=16)
            cfg["v7_recipe"]="continue"
            (folder/"source.py").write_text("EXPERIMENT_CONFIG = "+repr(cfg),encoding="utf-8")
            (folder/"best_model.pt").write_bytes(b"test checkpoint")
            state={"ref":"owner/existing","kind":"v4_smoke","status":"SUBMIT_PENDING","v6":True,**{k:cfg[k] for k in ("v56_key","v56_source_sha","v56_action","architecture","epochs","sepformer_scale_policies")}}
            core.json_write(folder/"remote_state.json",state)
            r={"smoke":True,"evaluation_protocol":"v4-smoke-fixture-v1","validation_files":164,"external_validation_files":64,
                "best_validation_pesq_wb":3.2,"external_validation_pesq_wb":2.5,"joint_score":2.85,
                "robust_score":.5*(3.2+3.+2.)/3+.5*2.,"domain_scores":{"challenge":3.2,"external_normal":3.,"external_stress":2.},
                "model_family":"mp_dns","epochs_completed":1,"best_epoch":1,"v56_requested_epochs":1,
                "zero_shot":{"challenge":3.1,"external_domains":[3.,2.]},
                "v6":{"protocol":"v6-max-pesq-v1","mode":"train","parent":None,"checkpoint_sha256":v.sha(folder/"best_model.pt"),
                    "attempted_epochs":1,"checkpoint_epoch":1,"root_external_domains":[3.,2.],"start_pesq":3.1,"gain":.1},
                **{k:cfg[k] for k in ("v56_key","v56_source_sha","v56_action","sepformer_scale_policies")}}
            for name,values in (("validation_per_file.csv",[3.2]*164),("external_validation.csv",[3.]*32+[2.]*32),
                ("zero_shot_validation.csv",[3.1]*164),("zero_shot_external.csv",[3.]*32+[2.]*32)):
                with (folder/name).open("w",newline="",encoding="utf-8") as out:
                    w=csv.writer(out);w.writerow(["index","pesq_wb"]);w.writerows(enumerate(values))
            core.json_write(folder/"result.json",r)
            # Legacy caller still rejects one epoch; only explicit V6 archived contract accepts it.
            with self.assertRaisesRegex(ValueError,"Incomplete legacy smoke"):
                v.v4.verify(folder,"v4_smoke")
            self.assertEqual(v.verify(folder,state)[0],r)
            for n in (0,2):
                core.json_write(folder/"result.json",{**r,"epochs_completed":n})
                with self.assertRaisesRegex(ValueError,"Incomplete V6/V7 configured smoke"):
                    v.verify(folder,state)
            core.json_write(folder/"result.json",r)
            with self.assertRaisesRegex(ValueError,"Invalid archived"):
                v.verify(folder,{**state,"epochs":2})
            with patch.object(v,"command",side_effect=[
                SimpleNamespace(returncode=0,stdout="KernelWorkerStatus.COMPLETE",stderr=""),
                SimpleNamespace(returncode=0,stdout="",stderr="")]) as cli:
                got=v.poll("kaggle",folder,state,999,interval=0,clock=lambda:0,sleep=lambda _:None)
            self.assertEqual(got,r)
            self.assertEqual(json.loads((folder/"remote_state.json").read_text())["status"],"COMPLETE")
            self.assertEqual([call.args[1:3] for call in cli.call_args_list],[("kernels","status"),("kernels","output")])
            (folder/"best_model.pt").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"checkpoint corrupt"):
                v.verify(folder,state)

if __name__=="__main__": unittest.main()
