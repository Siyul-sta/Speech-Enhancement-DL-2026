"""Offline oracle diagnostics, never a deployable enhancer.
Run from repository root:
  py tools/oracle_mask_audit.py --root PATH --split validation --out runs/oracle_validation
  py tools/oracle_mask_audit.py --self-test
Requires torch, numpy, soundfile, pesq. Writes metrics only, no audio.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR, NFFT, HOP, LENGTH = 16000, 512, 128, 48000

def project(y, s, max_gain, max_phase):
    """Closest complex coefficient in a bounded radial sector (Euclidean, not PESQ)."""
    radius = y.abs()
    phase = torch.angle(y)
    delta = torch.angle(s * y.conj()).clamp(-max_phase, max_phase)
    unit = torch.polar(torch.ones_like(phase), phase + delta)
    gain_radius = (s * unit.conj()).real.clamp_min(0)
    return unit * torch.minimum(gain_radius, max_gain * radius)

def stft(x, w):
    return torch.stft(x, NFFT, hop_length=HOP, window=w, return_complex=True)

def istft(z, w):
    return torch.istft(z, NFFT, hop_length=HOP, window=w, length=LENGTH)

def safe_read(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path outside dataset root")
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR or x.shape != (LENGTH, 1) or not np.isfinite(x).all():
        raise ValueError("Invalid 16 kHz mono 3-second input: " + str(path))
    return torch.from_numpy(x[:, 0].copy())

def self_test():
    y = torch.tensor([1+0j, 1+0j, 0+0j, 1+0j])
    s = torch.tensor([2+0j, 0+2j, 1+0j, -2+0j])
    a = project(y, s, 1.0, 0.5)
    b = project(y, s, 2.0, math.pi)
    assert torch.all(a.abs() <= y.abs() + 1e-6)
    assert torch.all(torch.angle(a[:2]).abs() <= 0.500001)
    assert torch.allclose(b, torch.tensor([2+0j, 0+2j, 0+0j, -2+0j]), atol=1e-6)
    assert torch.all((b-s).abs() <= (a-s).abs()+1e-6)
    w = torch.hann_window(NFFT)
    x = torch.randn(LENGTH, generator=torch.Generator().manual_seed(42))*0.05
    assert torch.allclose(istft(stft(x,w),w), x, atol=1e-6)
    print("Oracle projection and STFT round-trip self-test passed")

def run(args):
    from pesq import pesq
    root = Path(args.root).resolve()
    with (root / args.split / "samples.csv").open(encoding="utf-8-sig", newline="") as f:
        manifest = list(csv.DictReader(f))
    expected = {"train":1296, "validation":164}[args.split]
    if len(manifest) != expected:
        raise ValueError(f"Expected {expected} pairs, got {len(manifest)}")
    if args.limit is not None and not 1 <= args.limit <= expected:
        raise ValueError("Invalid limit")
    manifest = manifest[:args.limit] if args.limit is not None else manifest
    w = torch.hann_window(NFFT)
    results = []
    def score(c, x):
        value = float(pesq(SR, c.numpy().clip(-1,1), x.numpy().clip(-1,1), "wb"))
        if not math.isfinite(value):
            raise ValueError("Non-finite PESQ")
        return value
    with torch.no_grad():
        for i, row in enumerate(manifest):
            x = safe_read(root, row["noisy_path"])
            c = safe_read(root, row["clean_path"])
            y, s = stft(x,w), stft(c,w)
            r = y.abs()
            energy = s.abs().square()
            weight = energy.sum().clamp_min(1e-12)
            phase_diff = torch.angle(s*y.conj()).abs()
            data = {"index":i, "segment_id":row["segment_id"],
                    "input_snr_db":float(10*torch.log10((c.square().sum()+1e-12)/((x-c).square().sum()+1e-12))),
                    "target_energy_gain_gt1_fraction":float(energy[r<s.abs()].sum()/weight),
                    "target_energy_phase_gt0p5_fraction":float(energy[phase_diff>0.5].sum()/weight),
                    "noisy":score(c,x)}
            for name,g,p in (("mag_only",1,0),("bounded_phase",1,0.5),("gain2_phase",2,0.5),("gain2_fullphase",2,math.pi)):
                data[name] = score(c,istft(project(y,s,g,p),w))
            data["clean_stft_identity"] = score(c,istft(s,w))
            results.append(data)
            print(f"{i+1}/{len(manifest)} bounded={data['bounded_phase']:.4f} relaxed={data['gain2_fullphase']:.4f}",flush=True)
    keys = [k for k in results[0] if k not in ("index","segment_id")]
    means = {k:float(np.mean([r[k] for r in results])) for k in keys}
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    with (out/"oracle_per_file.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    summary={"split":args.split,"files":len(results),"complete":len(results)==expected,
             "means":means,"diagnostic_only":True,
             "note":"Target-informed projections minimize coefficient error, not PESQ; scores are not strict PESQ ceilings or model results."}
    (out/"oracle_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(summary,indent=2,allow_nan=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root")
    p.add_argument("--split",choices=("train","validation"),default="train")
    p.add_argument("--out",default="runs/oracle_train")
    p.add_argument("--limit",type=int)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.root: p.error("--root is required")
        run(args)
