import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tools")]
import kaggle_v4_external as trainer
import infer_v4
import kaggle_v4_public as public
import v4_research_catalog as catalog
from unittest.mock import patch

class CpuModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
    def test_native_model_forward_backward(self):
        for cls in (trainer.ResidualPhaseUNet,trainer.TemporalEnhancer,trainer.AdaptiveMixture):
            with self.subTest(model=cls.__name__):
                model=cls()
                x=torch.randn(1,4096)*.05
                y=model(x)
                self.assertEqual(y.shape,x.shape)
                self.assertTrue(torch.isfinite(y).all())
                y.square().mean().backward()
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
    def test_composed_model(self):
        for n in (0,1,2):
            with patch.dict(trainer.EXPERIMENT_CONFIG,{"network_spec":catalog.default_network(n)}):
                model=trainer.ComposedEnhancer()
                x=torch.randn(1,1024)*.05
                y=model(x)
                self.assertEqual(y.shape,x.shape)
                y.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    def test_synthetic_validation_repeatable(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            c=path/"clean.wav"; n=path/"noise.wav"
            sf.write(c,np.sin(np.arange(48000)*.07).astype(np.float32)*.1,16000)
            sf.write(n,np.random.default_rng(1).normal(0,.02,48000).astype(np.float32),16000)
            data=trainer.SyntheticPairs([c],[n],2,42,False)
            x,y=data[0]; x2,y2=data[0]
            self.assertEqual(tuple(x.shape),(48000,))
            self.assertTrue(torch.equal(x,x2) and torch.equal(y,y2))
            self.assertTrue(torch.isfinite(x).all())
            self.assertLessEqual(float(x.abs().max()),1)
            self.assertEqual(trainer.quality_paths([c,c,n]),[c,n])
            stress=trainer.SyntheticPairs([c],[n],2,42,False,stress=True)
            sx,sy=stress[0]
            self.assertTrue(torch.equal(sx,stress[0][0]))
            self.assertFalse(torch.equal(sx,x))
    def test_inference_lengths(self):
        for length in (1,24001,48000,48001,100003):
            x=np.random.default_rng(42).normal(0,.1,length).astype(np.float32)
            np.testing.assert_allclose(infer_v4.enhance(torch.nn.Identity(),x,"cpu"),x,atol=1e-7)
        self.assertEqual(len(infer_v4.resample(np.zeros(44100,dtype=np.float32),44100,16000)),16000)

    def test_public_recipe_replay(self):
        def fake(name,device):
            return lambda x:x*(.2 if name=="metricgan" else .6)
        x=torch.ones(1,100)*.5
        with patch.object(public,"load_model",fake):
            torch.testing.assert_close(public.public_recipe("metricgan:0.75","cpu")(x),x*.4)
            torch.testing.assert_close(public.public_recipe("blend:metricgan:deepfilter:0.25","cpu")(x),x*.5)
            with self.assertRaises(ValueError): public.public_recipe("metricgan:2","cpu")
    def test_urgent_decode_and_long_audit(self):
        import io
        x=np.sin(np.arange(22050)*.1).astype(np.float32)*.1
        buf=io.BytesIO()
        sf.write(buf,x,22050,format="WAV",subtype="FLOAT")
        y=trainer.decode_audit_audio({"bytes":buf.getvalue()})
        self.assertEqual(len(y),16000)
        np.testing.assert_allclose(trainer.enhance_long(torch.nn.Identity(),y,"cpu"),y,atol=1e-7)
    def test_public_archive_fixture(self):
        import io,tarfile,zipfile
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            # Tiny deterministic fixture, same extension and directory schema as OpenSLR.
            def wav(freq,fmt):
                buf=io.BytesIO()
                sf.write(buf,np.sin(np.arange(16000)*freq).astype(np.float32)*.1,16000,format=fmt)
                return buf.getvalue()
            def tar(items):
                buf=io.BytesIO()
                with tarfile.open(fileobj=buf,mode="w:gz") as tf:
                    for name,data in items:
                        info=tarfile.TarInfo(name); info.size=len(data)
                        tf.addfile(info,io.BytesIO(data))
                return buf.getvalue()
            train=tar([(f"LibriSpeech/train-clean-100/1/1/1-1-{i}.flac",wav(.05+i*.001,"FLAC")) for i in range(40)])
            dev=tar([("LibriSpeech/dev-clean/2/2/2-2-1.flac",wav(.09,"FLAC"))])
            noise=tar([(f"musan/noise/g{i}/n.wav",wav(.12+i*.03,"WAV")) for i in range(10)])
            def fake(url,timeout=60):
                return io.BytesIO(train if "train-clean" in url else dev if "dev-clean" in url else noise)
            with patch.object(trainer,"WORK",root),patch.dict(trainer.EXPERIMENT_CONFIG,{"data_profile":"base","public_cache":None}),patch("urllib.request.urlopen",fake):
                tr,dv,tn,dn,ts,te=trainer.public_assets()
                self.assertTrue(tr and dv and tn and dn and te)
                self.assertFalse(set(tn)&set(dn) or set(tn)&set(te) or set(dn)&set(te))
                self.assertEqual(ts,[])
                self.assertFalse(list((root/"external").glob("*.tar.gz")))
                manifest=__import__("json").loads((root/"external_manifest.json").read_text())
                self.assertNotIn("test",manifest)


    def test_training_steps_actual_weights(self):
        import copy
        model=trainer.TemporalEnhancer().train()
        teacher=copy.deepcopy(model).eval().requires_grad_(False)
        distill=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        x=torch.randn(2,4096)*.05
        y=x*.6
        cfg={"id":"V3_RESIDUAL_PHASE","loss_mode":"composed","loss_weights":{"wave":.2,"spectral":.3,"sisdr":.5}}
        before=model.out.weight.detach().clone()
        old_teacher=teacher.out.weight.detach().clone()
        loss=trainer.supervised_step(model,optimizer,x,y,cfg,distill,teacher)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before,model.out.weight))
        self.assertFalse(torch.equal(old_teacher,teacher.out.weight))
        for mode in ("consistency","remix"):
            before=model.out.weight.detach().clone()
            self.assertTrue(np.isfinite(trainer.consistency_step(model,optimizer,teacher,x,mode)))
            self.assertFalse(torch.equal(before,model.out.weight))
        self.assertTrue(all(p.grad is None for p in distill.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_wer(self):
        self.assertEqual(trainer.word_error_count("one two three","one four three"),(1,3))
        self.assertEqual(trainer.word_error_count("one two","one"),(1,2))

if __name__=="__main__":
    unittest.main()
