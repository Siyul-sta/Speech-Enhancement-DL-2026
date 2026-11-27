"""V5.5 CPU contracts; no GPU, private audio, or public model downloads."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"src")]
import v55_gating as g
import v55_ensemble as e
import submit_v5 as s

class Scale(nn.Module):
    def __init__(self,scale):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(float(scale)))
    def forward(self,x): return x*self.weight

class V55Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(55)

    def test_simplex_and_frequency_identity(self):
        x=torch.randn(2,4096)*.05
        p=torch.stack((x,x*.5),1)
        for mode in ("clip","frequency"):
            gate=g.EnsembleGate(2,mode)
            w=gate(x)
            self.assertEqual(w.shape,(2,2,1 if mode=="clip" else 4))
            torch.testing.assert_close(w.sum(1),torch.ones_like(w[:,0]))
            self.assertTrue((w>=0).all())
            y=g.mix_outputs(x,p,w)
            torch.testing.assert_close(y,x*.75,atol=1e-6,rtol=1e-5)
            self.assertTrue(torch.isfinite(gate(torch.zeros_like(x))).all())

    def test_fit_gate_and_frozen_experts(self):
        x=torch.randn(12,4096)*.03
        p=torch.stack((x,x*.2),1)
        for mode in ("clip","frequency"):
            gate=g.fit_gate(x,p,x,mode,epochs=5)
            with torch.no_grad(): self.assertGreater(float(gate(x)[:,0].mean()),.51)
            ensemble=g.GatedEnsemble([Scale(1),Scale(.2)],gate)
            ensemble(x).square().mean().backward()
            self.assertTrue(all(p.grad is None for p in ensemble.experts.parameters()))
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in gate.parameters()))

    def test_speaker_disjoint_and_fixed_weights(self):
        rows=[{"segment_id":f"speaker{i}_{j}"} for i in range(10) for j in range(20)]
        fit,select=e.split_train(rows,64,32)
        self.assertEqual((len(fit),len(select)),(64,32))
        self.assertFalse(set(fit)&set(select))
        self.assertFalse({e.speaker(rows[i]) for i in fit}&{e.speaker(rows[i]) for i in select})
        self.assertEqual((fit,select),e.split_train(rows,64,32))
        weights=list(e.fixed_weights(3))
        self.assertEqual(len(weights),13)
        self.assertEqual(len(set(weights)),13)
        self.assertTrue(all(abs(sum(w)-1)<1e-6 and min(w)>=0 for w in weights))
        with self.assertRaises(ValueError): e.split_train([{"segment_id":"unknown"}]*100,64,32)

    def test_actual_score_path_and_fallback(self):
        x=torch.ones(2,4096)*.123456
        def fake(sr,c,y,mode):
            self.assertEqual((sr,mode),(16000,"wb"))
            self.assertTrue(np.allclose(y*32768,np.round(y*32768)))
            return float(4.5-np.mean((c-y)**2))
        self.assertEqual(len(e.actual_scores(x,x,fake)),2)
        singles=[{"type":"fixed","weights":[1.,0.],"pesq":2.3},{"type":"fixed","weights":[0.,1.],"pesq":2.5}]
        self.assertEqual(e.choose_candidate(singles+[{"type":"clip","pesq":2.51}]),singles[1])
        candidate={"type":"frequency","pesq":2.6}
        self.assertEqual(e.choose_candidate(singles+[candidate]),candidate)

    def test_select_verified_distinct_families(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);records=[]
            for i,(family,score,blob) in enumerate([("temporal",2.9,b"a"),("temporal",2.8,b"b"),("residual",2.7,b"c"),("dual_axis",2.6,b"a"),("residual",2.5,b"d")]):
                p=root/str(i);p.mkdir()
                (p/"source.py").write_text("# fixture")
                (p/"result.json").write_text("{}")
                (p/"best_model.pt").write_bytes(blob)
                records.append(({"kind":"v4_external","status":"COMPLETE","source_path":str(p/"source.py")},{"evaluation_protocol":"v4.2-fixed-dev-v2","model_family":family,"best_validation_pesq_wb":score}))
            self.assertEqual([p.name for p in e.select_experts(records,3)],["0","2","1"])
            records[0][1]["smoke"]=True
            self.assertEqual(e.select_experts(records,2)[0].name,"1")

    def test_strict_gate_bundle_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            entries=[]
            source_code="""import torch
from torch import nn
EXPERIMENT_CONFIG={"architecture":"temporal"}
class TemporalEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(1.))
    def forward(self,x): return x*self.weight
"""
            for i,value in enumerate((1.,.5)):
                source=root/f"source{i}.py"
                source.write_text(source_code,encoding="utf-8")
                checkpoint=root/f"model{i}.pt"
                torch.save({"weight":torch.tensor(value)},checkpoint)
                entries.append({"weight":.5,"family":"temporal","source":source.name,"checkpoint":checkpoint.name,"source_sha256":s.sha256(source),"checkpoint_sha256":s.sha256(checkpoint)})
            gate=root/"gate.pt";torch.save(g.EnsembleGate(2,"frequency").state_dict(),gate)
            config=root/"bundle.json"
            cfg={"models":entries,"gating":{"mode":"frequency","experts":2,"checkpoint":"gate.pt","checkpoint_sha256":s.sha256(gate)}}
            config.write_text(json.dumps(cfg),encoding="utf-8")
            models=s.load_bundle(config,torch.device("cpu"))
            x=np.random.default_rng(55).normal(0,.03,48000).astype(np.float32)
            np.testing.assert_allclose(s.predict(models,x,torch.device("cpu")),x*.75,atol=1e-6,rtol=1e-5)
            cfg["gating"]["checkpoint_sha256"]="0"*64
            config.write_text(json.dumps(cfg),encoding="utf-8")
            with self.assertRaises(ValueError): s.load_bundle(config,torch.device("cpu"))

if __name__=="__main__": unittest.main()
