"""V5.7.2 scale contracts (no pretrained downloads)."""
import sys
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"src")]
import kaggle_v4_external as t
import v572_scale_audit as a
import ralph_v56 as v

class V572Tests(unittest.TestCase):
    def test_bounds_and_finite_gradients(self):
        x=torch.tensor([[.2,-.1,.3,-.2],[0.,0.,0.,0.]])
        for mode in t.SEPFORMER_SCALE_MODES:
            y=torch.tensor([[117.,-4.,2.,-.6],[0.,0.,0.,0.]],requires_grad=True)
            z=t.sepformer_scale_output(x,y,mode)
            self.assertEqual(z.shape,x.shape)
            self.assertTrue(torch.isfinite(z).all())
            if mode!="raw": self.assertLessEqual(float(z.detach().abs().max()),.98001)
            z.square().mean().backward()
            self.assertTrue(torch.isfinite(y.grad).all(),mode)

    def test_peak_preserves_waveform_ratios_and_small_signal(self):
        x=torch.ones(1,4)*.1
        y=torch.tensor([[100.,-50.,25.,-10.]])
        z=t.sepformer_scale_output(x,y,"peak")
        torch.testing.assert_close(z/y,torch.full_like(y,.0098))
        small=y/1000
        torch.testing.assert_close(t.sepformer_scale_output(x,small,"peak"),small)

    def test_speaker_disjoint_and_deterministic(self):
        rows=[{"speaker_id":f"s{i}","segment_id":f"{i}_{j}"} for i in range(12) for j in range(5)]
        first,second=a.split(rows,12)
        self.assertEqual((first,second),a.split(rows,12))
        self.assertEqual(len(first),12);self.assertEqual(len(second),12)
        self.assertFalse({a.speaker(rows[i]) for i in first}&{a.speaker(rows[i]) for i in second})

    def test_holdout_veto_not_reselection(self):
        sel={m:{"pesq":2., "mean_clipping":0., "max_peak":.9} for m in a.MODES}
        hold={m:dict(r) for m,r in sel.items()}
        sel["rms_050"]["pesq"]=3.
        hold["rms_050"]["pesq"]=1.8
        hold["peak"]["pesq"]=4.
        self.assertEqual(a.choose(sel,hold)["mode"],"rms_050")
        self.assertEqual(a.choose(sel,hold)["status"],"FAIL")
        hold["rms_050"]["pesq"]=2.1
        self.assertEqual(a.choose(sel,hold)["status"],"PASS")

    def test_smoke_key_and_plan_policy(self):
        family="sepformer_dns_transfer"
        left=v.smoke_cfg(family,"a"*64,{family:"peak"})
        right=v.smoke_cfg(family,"a"*64,{family:"rms_050"})
        self.assertNotEqual(left["v56_key"],right["v56_key"])
        self.assertEqual(left["sepformer_scale_policies"],{family:"peak"})
        self.assertEqual(t.sepformer_scale_policy(family,{"sepformer_scale_policies":{family:"peak"}}),"peak")
        with self.assertRaises(ValueError): t.sepformer_scale_policy(family,{"sepformer_scale_policies":{family:"invalid"}})

if __name__=="__main__": unittest.main()
