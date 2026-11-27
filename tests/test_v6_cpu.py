"""V6 tensor tests with synthetic fixtures (no actual challenge or GPU)."""
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import ralph_v6 as ctl
from v6_gating import V6Gate,V6Ensemble,v6_align,v6_combine,v6_fit_gate
from v6_preflight import load_module

class V6CPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);torch.manual_seed(42)
        cls.tmp=tempfile.TemporaryDirectory()
        p=Path(cls.tmp.name)/"runner.py";p.write_text(ctl.source(),encoding="utf-8")
        cls.mod=load_module(p)
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_noisy_only_native_cascade_identity_and_gradients(self):
        mod=self.mod
        x=torch.randn(1,4096)*.05
        model=mod.v6_make_model({"architecture":"complex_residual","v6_base_family":None}).eval()
        model2=mod.V6Refiner(model,3).eval()
        with torch.no_grad():
            y=model(x);z=model2(x)
        self.assertTrue(torch.allclose(y,z,atol=2e-5,rtol=2e-4))
        model2.train()
        loss=mod.v6_loss(model2(x),x*.8,{"id":"V3_RESIDUAL_PHASE"})
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model2.phase.parameters()))
        self.assertFalse(any(p.grad is not None for p in model2.coarse.parameters()))

    def test_snr_and_train_only_speaker_split(self):
        mod=self.mod;rng=np.random.default_rng(42)
        s=rng.normal(0,.02,48000).astype("float32");n=rng.normal(0,.02,48000).astype("float32")
        x,c=mod.v6_mix(s,n,12)
        actual=10*torch.log10(c.square().sum()/(x-c).square().sum())
        self.assertAlmostEqual(float(actual),12,places=4)
        class Train:
            split="train"
            manifest=pd.DataFrame({"segment_id":[f"speaker{i%36}_clip{i}" for i in range(1296)]})
            def __len__(self): return 1296
        train=Train();pools=mod.v6_split(train)
        self.assertEqual(len(set().union(*map(set,pools))),1296)
        self.assertEqual(sum(map(len,pools)),1296)
        names=[{i%36 for i in p} for p in pools]
        self.assertFalse(any(a&b for i,a in enumerate(names) for b in names[i+1:]))
        train.split="validation"
        with self.assertRaises(ValueError): mod.v6_split(train)

    def test_gating_finite_and_simplex(self):
        x=torch.randn(3,4096)*.05;p=torch.stack((x,x*.6),1)
        for mode in ("clip","frequency","timefreq"):
            g=V6Gate(2,mode);w=g(x)
            self.assertTrue(torch.allclose(w.sum(1),torch.ones_like(w.sum(1))))
            y=v6_combine(x,p,w)
            self.assertTrue(torch.allclose(y,x*.8,atol=2e-6))
            fit=v6_fit_gate(x,p,x*.65,mode,epochs=1)
            self.assertTrue(torch.isfinite(fit(x)).all())

    def test_alignment_and_single_identity(self):
        x=torch.randn(2,4096)*.05
        p=torch.stack((x,torch.roll(x,8,dims=-1)*.5),1)
        for mode in ("none","rms","lag_rms"):
            self.assertTrue(torch.isfinite(v6_align(x,p,mode)).all())
        z=v6_align(x,p,"lag_rms")
        corr=lambda a,b:float((a*b).sum()/(a.norm()*b.norm()))
        self.assertGreater(corr(x,z[:,1]),.99)
        e=V6Ensemble([torch.nn.Identity()],weights=[1.])
        self.assertTrue(torch.equal(e(x),x))

    def test_pinned_mp_asset_rejects_tampering(self):
        mod=self.mod
        self.assertEqual(mod.v6_git_blob(b"test"),"30d74d258442c7c65512eafab474568dd706c430")
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"models/model.py";p.parent.mkdir();p.write_text("tampered")
            with self.assertRaises(ValueError): mod.v6_mp_assets(Path(d),"dns",False)

    def test_crossmix_deterministic_and_train_only(self):
        import soundfile as sf
        mod=self.mod;rng=np.random.default_rng(44)
        class Train:
            split="train"
            manifest=pd.DataFrame({"noise_type":["a","b"]*32})
            clean=rng.normal(0,.025,(64,48000)).astype("float32")
            noisy=clean+rng.normal(0,.01,clean.shape).astype("float32")
            def __len__(self): return 64
        t=Train()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"speech.wav";sf.write(p,t.clean[0],16000)
            mix=mod.V6CrossMix(t,range(64),[p],[],1000000,seed=42,hard=True,korean=[p])
            self.assertEqual(len(mix),1000000)
            a,b=mix[0];c,e=mix[0]
            self.assertTrue(torch.equal(a,c) and torch.equal(b,e))
            mix.epoch=1;f,_=mix[0]
            self.assertFalse(torch.equal(a,f))
            mix.update_hard({"a":1.2,"b":3.2})
            self.assertGreater(mix.probs[0],mix.probs[1])
            t.split="validation"
            with self.assertRaises(ValueError): mod.V6CrossMix(t,range(64),[p],[],32)

    def test_native_offline_bundle_roundtrip(self):
        import json,re
        import v6_offline as offline_runner
        import v57_offline as old_offline
        mod=self.mod;cfg={"architecture":"complex_residual","v6_base_family":None}
        x=torch.randn(1,4096)*.03
        model=mod.v6_make_model(cfg).eval()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/"source.py";cp=root/"best_model.pt"
            source.write_text(re.sub(r"^EXPERIMENT_CONFIG = .+$",lambda _: "EXPERIMENT_CONFIG = "+repr(cfg),ctl.source(),count=1,flags=re.M),encoding="utf-8")
            torch.save(model.state_dict(),cp)
            spec={"kind":"v6","family":"complex_residual","source":"source.py","source_sha256":ctl.sha(source),"checkpoint":"best_model.pt","checkpoint_sha256":ctl.sha(cp)}
            (root/"v6_bundle.json").write_text(json.dumps({"protocol":"v6-offline-v1","experts":[spec],"alignment":"none","weights":[1.]}))
            (root/"package_manifest.json").write_text(json.dumps(old_offline.package_manifest(root)))
            restored=offline_runner.load_package(root)
            with torch.no_grad(): self.assertTrue(torch.allclose(model(x),restored(x),atol=2e-6))
            cp.write_bytes(b"tampered")
            with self.assertRaises(ValueError): offline_runner.load_package(root)

    def test_actual_pinned_mp_if_enabled(self):
        import os
        if os.environ.get("V6_TEST_MP")!="1": self.skipTest("Optional real upstream assets")
        mod=self.mod
        root=Path(self.tmp.name)/"assets"
        for family in ("mp_dns","mp_vb"):
            m=mod.V6MPNet(family,root,allow_download=True).eval()
            x=torch.randn(1,48000)*.03
            with torch.no_grad(): y=m(x)
            self.assertEqual(y.shape,x.shape);self.assertTrue(torch.isfinite(y).all())
            other=mod.V6MPNet(family,root,False,False).eval()
            other.load_state_dict(m.state_dict(),strict=True)
            with torch.no_grad(): z=other(x)
            self.assertTrue(torch.allclose(y,z,atol=2e-5,rtol=2e-4))
            m.train();m(x[:,:4096]).abs().mean().backward()
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

if __name__=="__main__": unittest.main()
