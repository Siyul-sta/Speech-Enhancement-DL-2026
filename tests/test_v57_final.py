"""V5.7 CPU contracts. Public CMGAN test opt-in; SpeechBrain uses fixture architecture."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"src")]
import v57_offline as o
import v57_finalize as f
import v57_public_assets as a
import submit_v5 as s
import kaggle_v4_external as t

SOURCE="""import torch
from torch import nn
EXPERIMENT_CONFIG={"architecture":"temporal"}
class TemporalEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(.5))
    def forward(self,x): return x*self.weight
"""

class V57Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_paired_bootstrap_and_subgroups(self):
        r=o.paired_bootstrap([.2,.2,.2],["a","b","c"])
        np.testing.assert_allclose(r["ci95"],[.2,.2])
        self.assertEqual(r["method"],"speaker-cluster bootstrap")
        self.assertIsNone(o.paired_bootstrap([.1,.2],["a","a"])["ci95"])
        rows=[{"index":i,"pesq_wb":p,"noisy_pesq":2.,"gain":p-2.,"snr_db":3.,"noise":"wind" if i else "alarm","input_snr_group":"0–<5","speaker":"unknown"} for i,p in enumerate((1.5,2.5,3.))]
        summary=o.summarize(rows)
        self.assertEqual(summary["degraded_count"],1)
        self.assertEqual(summary["worst_indices"],[0,1,2])
        self.assertEqual(summary["by_noise"]["wind"]["count"],2)

    def test_package_and_actual_offline_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);pkg=root/"package";pkg.mkdir()
            source=pkg/"source.py";source.write_text(SOURCE,encoding="utf-8")
            checkpoint=pkg/"model.pt";torch.save({"weight":torch.tensor(.5)},checkpoint)
            f.write(pkg/"bundle.json",{"models":[{"weight":1.,"family":"temporal","source":"source.py","checkpoint":"model.pt","source_sha256":s.sha256(source),"checkpoint_sha256":s.sha256(checkpoint)}]})
            f.finalize_package(pkg)
            o.verify_package(pkg)
            inp=root/"input";inp.mkdir();out=root/"output"
            x=np.full(48000,.25,np.float32)
            sf.write(inp/"a.wav",x,16000,subtype="PCM_16")
            f.offline_command(pkg,"--probe")
            f.offline_command(pkg,"--input",inp,"--output",out,"--zip")
            y=s.read_wav(out/"a.wav")
            np.testing.assert_allclose(y,x*.5,atol=1/32768)
            self.assertTrue((root/"output.zip").is_file())
            self.assertEqual(len(o.verify_outputs(inp,out)),1)
            (out/"extra.txt").write_text("bad")
            with self.assertRaises(ValueError): o.verify_outputs(inp,out)
            source.write_text(SOURCE+"# tampered",encoding="utf-8")
            with self.assertRaises(ValueError): o.verify_package(pkg)

    def test_offline_network_guard(self):
        code="from v57_offline import block_network; import socket; block_network(); socket.socket()"
        p=subprocess.run([sys.executable,"-c",code],cwd=ROOT/"tools",capture_output=True,text=True)
        self.assertNotEqual(p.returncode,0)
        self.assertIn("Offline inference forbids",p.stderr)

    def test_candidate_policy(self):
        rows=[{"status":"REJECTED","rank":1},{"status":"COMPLETE","rank":2,"pcm16_pesq":2.4},{"status":"COMPLETE","rank":3,"pcm16_pesq":2.6}]
        self.assertEqual(f.choose_best(rows)["rank"],3)
        with self.assertRaises(RuntimeError): f.choose_best(rows[:1])

    def test_archived_speechbrain_config_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/"offline_hyperparams.yaml";p.write_text("modules: {}")
            f.write(root/"offline_hyperparams.json",{"family":"metricgan","sha256":a.file_hash(p),"speechbrain":"1.0.3","hyperpyyaml":"1.2.2"})
            meta=a.package_public("metricgan",root,"",root/"assets")
            self.assertEqual(meta["kind"],"speechbrain")
            with self.assertRaises(ValueError): a.package_public("sepformer_dns_transfer",root,"",root/"wrong")
            p.write_text("different")
            with self.assertRaises(ValueError): a.package_public("metricgan",root,"",root/"tampered")

    def test_overlap_guard(self):
        # Small mocked challenge manifests; exact waveform overlap must fail.
        rows=[{"segment_id":"speaker1_0","clean_path":"clean.wav","noisy_path":"noisy.wav"}]
        audit=[{"segment_id":"speaker2_0","clean_path":"clean.wav","noisy_path":"noisy.wav"}]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);challenge=root/"challenge";challenge.mkdir();other=root/"audit";other.mkdir()
            x=np.zeros(48000,np.float32);y=np.full_like(x,.1)
            for place in (challenge,other):
                sf.write(place/"clean.wav",x,16000,subtype="PCM_16");sf.write(place/"noisy.wav",y,16000,subtype="PCM_16")
            with patch.object(o,"read_manifest",side_effect=[audit,rows,rows]):
                with self.assertRaisesRegex(ValueError,"waveform overlap"): f.audit_disjoint(challenge,other,other/"manifest.csv")

    @unittest.skipUnless(os.environ.get("V5_TEST_PUBLIC_CMGAN")=="1","opt-in pinned real CMGAN")
    def test_actual_cmgan_offline_reconstruction(self):
        with tempfile.TemporaryDirectory() as d,patch.object(t,"WORK",Path(d)):
            root=Path(d);original=t.CMGANTransfer().eval()
            x=(.1*torch.sin(torch.arange(4096)*.0864))[None]
            with torch.no_grad(): reference=original(x)
            meta=a.package_public("cmgan_transfer",root,(ROOT/"src/kaggle_v4_external.py").read_text(encoding="utf-8"),root/"assets",cmgan_root=root/"cmgan_assets")
            with patch.object(t,"cmgan_assets",side_effect=AssertionError("Offline constructor must not download")):
                with patch.object(t,"CMGANTransfer",t.CMGANTransfer):
                    restored=a.construct(t,"cmgan_transfer",root/"assets",meta).eval()
                    restored.load_state_dict(original.state_dict(),strict=True)
                    with torch.no_grad(): torch.testing.assert_close(restored(x),reference,atol=1e-5,rtol=1e-4)
            (root/"assets"/"LICENSE").write_text("tampered")
            with self.assertRaises(ValueError): a.make_cmgan_module(root/"assets",meta)

if __name__=="__main__": unittest.main()
