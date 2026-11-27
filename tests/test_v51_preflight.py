"""V5.1 diagnostics and smoke contracts; fixtures, not real PESQ evidence."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"tools")]
import v51_data_audit as audit
import v51_recover as recovery
import ralph_v4 as v4
import kaggle_v4_external as trainer

class V51Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_dataset_overlap_and_unknown_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); root=base/"dataset"; root.mkdir()
            rng=np.random.default_rng(42)
            c=rng.normal(0,.1,48000).astype(np.float32)
            n=rng.normal(0,.03,48000).astype(np.float32)
            for split in ("train","validation"):
                p=root/split;p.mkdir()
                sf.write(p/"clean.wav",c,16000,subtype="FLOAT")
                sf.write(p/"noisy.wav",c+n,16000,subtype="FLOAT")
                with (p/"samples.csv").open("w",newline="",encoding="utf-8") as f:
                    w=csv.writer(f);w.writerow(["segment_id","clean_path","noisy_path"])
                    w.writerow(["speaker001_"+split,split+"/clean.wav",split+"/noisy.wav"])
            with patch.dict(audit.EXPECTED,{"train":1,"validation":1}):
                r=audit.audit(root,base/"report")
            self.assertEqual(r["cross_split_exact_overlap"],{"clean":1,"noisy":1,"segment_id":0,"known_speaker_groups":1})
            self.assertEqual(r["splits"]["validation"]["by_noise_label"]["unknown"]["count"],1)
            self.assertAlmostEqual(audit.describe_pair(c+n,c)["input_snr_db"],10.46,delta=.3)

    def test_metric_gradient_probe(self):
        m=trainer.TemporalEnhancer()
        critic=trainer.PESQCritic()
        opt=torch.optim.AdamW(m.parameters(),lr=.001)
        co=torch.optim.AdamW(critic.parameters(),lr=.001)
        x=torch.randn(2,4096)*.05; c=x*.7
        def metric(sr,c,y,mode): return float(4.5-np.mean((c-y)**2))
        probe={}
        trainer.supervised_step(m,opt,x,c,{"id":"V3_RESIDUAL_PHASE"},critic=critic,critic_optimizer=co,pesq_fn=metric,metric_weight=.05,update_critic=True,metric_probe=probe)
        self.assertGreater(probe["gradient_norm"],0)
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,c),batch_size=2)
        r=trainer.critic_calibration(m,critic,loader,"cpu",metric)
        self.assertEqual(r["count"],2);self.assertTrue(np.isfinite(r["mae"]))

    def test_metric_smoke_validation_rejects_missing_gradient(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name,count in (("validation_per_file.csv",164),("external_validation.csv",64)):
                with (p/name).open("w",newline="") as f:
                    w=csv.writer(f);w.writerow(["index","pesq_wb"]);w.writerows((i,2.) for i in range(count))
            (p/"best_model.pt").write_bytes(b"fixture")
            r={"validation_files":164,"best_validation_pesq_wb":2.,"external_validation_files":64,"external_validation_pesq_wb":2.,"joint_score":2.,"evaluation_protocol":"v4-smoke-fixture-v1","domain_scores":{"challenge":2.,"external_normal":2.,"external_stress":2.},"robust_score":2.,"smoke":True,"epochs_completed":3,"smoke_metric_protocol":"v5.1-three-epoch","pesq_aware":True,"critic_updates":3,"metric_generator_steps":8,"metric_gradient_norm":.01,"critic_calibration":{"count":64,"mae":.2,"rmse":.3,"pearson":.5}}
            def write(): (p/"result.json").write_text(json.dumps(r))
            write();self.assertEqual(v4.verify(p,"v4_smoke")["metric_generator_steps"],8)
            for norm in (0.,None,float("nan")):
                r["metric_gradient_norm"]=norm;write()
                with self.assertRaises(ValueError): v4.verify(p,"v4_smoke")

    def test_recovery_never_downloads_running_job(self):
        with tempfile.TemporaryDirectory() as d,patch.object(recovery.reconcile,"status",return_value=("RUNNING","")),patch.object(recovery.subprocess,"run") as proc:
            with self.assertRaises(RuntimeError): recovery.recover("owner/job",Path(d)/"out",kaggle="mock")
            proc.assert_not_called()

if __name__=="__main__": unittest.main()
