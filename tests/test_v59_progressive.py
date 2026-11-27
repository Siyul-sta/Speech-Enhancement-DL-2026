"""V5.9 controller/source contracts; no GPU or model downloads."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import ralph_v59 as v
import ralph_v56 as old

class V59Tests(unittest.TestCase):
    def test_generated_source_and_unchanged_model_definitions(self):
        import ast
        base=(ROOT/"src/kaggle_v4_external.py").read_text(encoding="utf-8")
        source=v.build(base)
        compile(source,"trainer.py","exec")
        original={n.name:ast.dump(n) for n in ast.parse(base).body if isinstance(n,ast.ClassDef)}
        generated={n.name:ast.dump(n) for n in ast.parse(source).body if isinstance(n,ast.ClassDef)}
        self.assertEqual(original,generated)
        self.assertIn("V5.9 parent checkpoint hash mismatch",source)
        self.assertIn("transfer.apply(effective_epoch)",source)
        self.assertIn("ReduceLROnPlateau",source)
        with self.assertRaises(ValueError): v.build(base.replace('warm_ref = cfg.get("warm_ref")','warm_ref = None'))

    def test_catalog_and_policy(self):
        policies={"sepformer_dns_transfer":"rms_025","sepformer_wham_transfer":"rms_050"}
        stages=v.catalog([],policies)
        self.assertEqual(stages[0][0],"v53_stage:sepformer-dns-progressive")
        self.assertEqual(stages[0][1]["sepformer_scale_policies"],{"sepformer_dns_transfer":"rms_025"})
        self.assertTrue(all(v.eligible_stage(p) for _,p,_ in stages))
        self.assertFalse(any(p.get("v54_stage") or p.get("pesq_aware") for _,p,_ in stages))

    def test_chunk_budget_and_cap(self):
        self.assertEqual(v.next_chunk(12,12,20,200),13)
        self.assertEqual(v.next_chunk(25,12,20,200),13)
        self.assertEqual(v.next_chunk(38,12,20,200),12)
        self.assertEqual(v.next_chunk(50,5,20,200),20)
        self.assertEqual(v.next_chunk(195,12,20,200),0)
        self.assertEqual(v.next_chunk(100,5,4,200),0)
        for total in range(1,201):
            n=v.next_chunk(total,5,20,200)
            self.assertTrue(n==0 or 5<=n<=20 and total+n<=200)

    def test_continuation_keeps_recipe_and_lowers_lr(self):
        name,p,low=v.catalog([],{"sepformer_dns_transfer":"rms_025"})[0]
        cfg=v.make_cfg(p,name,low,None,"b"*64)
        parent={"ref":"owner/job","sha256":"a"*64,"attempted_epochs":12,"checkpoint_epoch":10,"score":2.6,"external_domains":[2.,2.],"root_external_domains":[2.,2.]}
        result={"v53_initial_hash":"c"*64,"v59":{"final_lrs":[1e-5]}}
        nxt=v.make_cfg(p,name,13,None,"b"*64,parent,cfg,result)
        self.assertEqual(nxt["warm_ref"],parent["ref"])
        self.assertEqual(nxt["v59_parent"],parent)
        self.assertEqual(nxt["v53_expected_initial_hash"],"c"*64)
        self.assertEqual(nxt["lr"],1e-5)
        self.assertEqual(nxt["sepformer_scale_policies"],cfg["sepformer_scale_policies"])
        self.assertNotEqual(nxt["v56_key"],cfg["v56_key"])
        self.assertEqual(nxt["epochs"],13)

    def test_smoke_before_pilot_and_budget(self):
        stages=v.catalog([],{});sha="b"*64
        args=([],{},set(),stages,"a"*64,sha,None,{})
        self.assertIsNone(v.choose(*args,.01,6))
        smoke=v.choose(*args,20,6)
        self.assertEqual(smoke["v56_action"],"smoke")
        ledger={smoke["v56_key"]:{"status":"COMPLETE","kind":"v4_smoke"}}
        cfg=v.choose([],ledger,set(),stages,"a"*64,sha,None,{},20,6)
        self.assertEqual(cfg["v56_action"],"v59_pilot")
        self.assertEqual(cfg["architecture"],"sepformer_dns_transfer")

    def test_parent_lineage_and_tamper(self):
        name,p,low=v.catalog([],{} )[2]
        cfg=v.make_cfg(p,name,low,None,"b"*64)
        with tempfile.TemporaryDirectory() as d:
            cp=Path(d)/"best_model.pt";cp.write_bytes(b"checkpoint")
            st={"ref":"owner/job","source_path":str(Path(d)/"source.py")}
            r={"epochs_completed":5,"best_epoch":4,"v56_action":"v59_pilot","warm_ref":None,
               "best_validation_pesq_wb":2.6,"domain_scores":{"external_normal":2.,"external_stress":1.9},
               "v59":{"protocol":"v59-weights-continuation-v1","parent":None,"attempted_epochs":5,"checkpoint_epoch":4,
                      "checkpoint_sha256":v.sha_file(cp),"root_external_domains":[2.,1.9],"start_pesq":None,"gain":None,
                      "optimizer_restored":False,"final_lrs":[cfg["lr"]]}}
            self.assertEqual(v.parent_record(st,r,cfg)[0]["checkpoint_epoch"],4)
            cp.write_bytes(b"tampered")
            with self.assertRaises(ValueError): v.verify_lineage(st,r,cfg)

    def test_choose_extension_and_stop_plateau(self):
        stages=v.catalog([],{});base="a"*64;sha="b"*64
        rows=[]
        for i,(name,p,low) in enumerate(stages[:3]):
            cfg=v.make_cfg(p,name,low,None,sha)
            st={"ref":f"owner/job{i}","kind":"v4_external"}
            r={"best_validation_pesq_wb":2.8-i*.15,"v56_action":"v59_pilot","v53_initial_hash":"c"*64}
            rows.append((name,st,r,cfg))
        def parent(st,r,cfg):
            n=12 if cfg.get("v53_stage") else 5
            return {"ref":st["ref"],"attempted_epochs":n,"checkpoint_epoch":n,"sha256":"d"*64,"score":r["best_validation_pesq_wb"],
                    "external_domains":[2.,2.],"root_external_domains":[2.,2.]},.2
        ledger={}
        for _,_,_,cfg in rows:
            s=v.smoke(cfg["architecture"],sha,{})
            ledger[s["v56_key"]]={"kind":"v4_smoke","status":"COMPLETE"}
        with patch.object(v,"compatible_records",return_value=rows),patch.object(v,"parent_record",side_effect=parent):
            out=v.choose([],ledger,set(),stages,base,sha,None,{},20,6)
            self.assertEqual(out["v56_action"],"v59_extend")
            self.assertEqual(out["v59_parent"]["ref"],"owner/job0")
        with patch.object(v,"compatible_records",return_value=rows),patch.object(v,"parent_record",side_effect=lambda s,r,c:(parent(s,r,c)[0],.001)):
            out=v.choose([],ledger,set(),stages[:3],base,sha,None,{},20,6)
            self.assertIsNone(out)

if __name__=="__main__": unittest.main()
