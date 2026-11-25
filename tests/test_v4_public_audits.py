"""Public-data and metric smoke tests. No challenge data or GPU."""
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from pesq import pesq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import kaggle_v4_external as trainer

class PublicAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
    def test_urgent_paired_and_blind_one_sample(self):
        with tempfile.TemporaryDirectory() as d,patch.object(trainer,"WORK",Path(d)):
            paired=trainer.urgent_audit(torch.nn.Identity(),"cpu",pesq,limit=1)
            self.assertEqual(paired["count"],1)
            self.assertTrue(math.isfinite(paired["scores"]["pesq_wb"]))
            blind=trainer.urgent_audit(torch.nn.Identity(),"cpu",pesq,limit=1,blind=True)
            self.assertEqual(blind["count"],1)
            self.assertTrue(math.isfinite(blind["scores"]["ovrl"]))
    def test_fleurs_train_and_test_one_sample(self):
        with tempfile.TemporaryDirectory() as d,patch.object(trainer,"WORK",Path(d)):
            tr=trainer.fleurs_speech("train",1)
            te=trainer.fleurs_speech("test",1)
            self.assertNotEqual(tr[0],te[0])
            for p in tr+te:
                y,sr=sf.read(p)
                self.assertEqual(sr,16000)
                self.assertGreater(len(y),15999)

    def test_real_pesq_parallel_equals_serial(self):
        from contextlib import closing
        with closing(trainer.urgent_rows()) as rows:
            row=next(rows)
        c=trainer.decode_audit_audio(row["clean_audio"])
        x=trainer.decode_audit_audio(row["noisy_audio"])
        n=min(len(c),len(x),48000)
        c=np.stack([c[:n]]*4); x=np.stack([x[:n],x[:n]*.9,x[:n]*.8,x[:n]*.7])
        try:
            serial=trainer.score_waveforms(c,x,pesq,workers=0)
            parallel=trainer.score_waveforms(c,x,pesq,workers=2)
            self.assertIsNotNone(trainer._PESQ_POOL)
            np.testing.assert_array_equal(serial,parallel)
        finally:
            trainer.close_pesq_pool()

    def test_actual_whisper_wer_pipeline(self):
        with tempfile.TemporaryDirectory() as d,patch.object(trainer,"WORK",Path(d)):
            root=Path(d); clean=root/"1-1-1.wav"; noise=root/"noise.wav"
            # Public read speech as source; test checks real ASR output plumbing only,
            # not a meaningful WER estimate for this synthetic transcript.
            from contextlib import closing
            with closing(trainer.urgent_rows()) as rows:
                row=next(rows)
            x=trainer.decode_audit_audio(row["clean_audio"])
            sf.write(clean,x,16000)
            sf.write(noise,np.random.default_rng(42).normal(0,.001,48000),16000)
            (root/"1-1.trans.txt").write_text("1-1-1 "+str(row["transcript"])+"\n",encoding="utf-8")
            result=trainer.transcription_audit(torch.nn.Identity(),[clean],[noise],"cpu")
            self.assertGreater(result["words"],0)
            self.assertTrue(math.isfinite(result["wer"]))

if __name__=="__main__":
    unittest.main()
