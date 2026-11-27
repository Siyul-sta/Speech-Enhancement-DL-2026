"""Strict offline challenge inference and fixed global checkpoint ensembles.
Only trusted archived sources and local weights. Native model families only;
public-model wrappers are rejected until their assets are independently packaged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

FAMILIES={"residual":"ResidualPhaseUNet","residual_relaxed":"RelaxedPhaseUNet","complex_residual":"ComplexResidualUNet","cascade_native_two":"CascadeNativeTwo","cascade_native_three":"CascadeNativeThree","cascade_native":"CascadeEnhancer",
          "dual_axis":"DualAxisEnhancer","temporal":"TemporalEnhancer","composed":"ComposedEnhancer"}
SR,LENGTH=16000,48000

def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def safe_path(root, name):
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError("Missing or nonlocal bundle asset: "+str(name))
    return p

def load_bundle(config,device):
    cfg=json.loads(config.read_text(encoding="utf-8"))
    specs=cfg["models"]
    if not isinstance(specs,list) or not 1<=len(specs)<=5:
        raise ValueError("Expected 1–5 fixed models")
    models=[]
    for i,s in enumerate(specs):
        w=float(s["weight"])
        if not math.isfinite(w) or w<=0: raise ValueError("Invalid fixed weight")
        source=safe_path(config.parent,s["source"])
        checkpoint=safe_path(config.parent,s["checkpoint"])
        for key,p in (("source_sha256",source),("checkpoint_sha256",checkpoint)):
            if sha256(p)!=s[key]: raise ValueError("Asset hash mismatch: "+str(p))
        spec=importlib.util.spec_from_file_location("trusted_bundle_"+str(i),source)
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        family=mod.EXPERIMENT_CONFIG.get("architecture","residual")
        if family not in FAMILIES: raise ValueError("Offline adapter not admitted: "+family)
        model=getattr(mod,FAMILIES[family])().to(device).eval()
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True)
        models.append((w,model))
    if not math.isclose(sum(w for w,m in models),1.,abs_tol=1e-6):
        raise ValueError("Fixed weights must sum to one")
    if cfg.get("gating") is not None:
        from v55_gating import EnsembleGate,GatedEnsemble
        g=cfg["gating"]
        if not isinstance(g,dict) or g.get("mode") not in ("clip","frequency") or g.get("experts")!=len(models):
            raise ValueError("Invalid gating config")
        checkpoint=safe_path(config.parent,g["checkpoint"])
        if sha256(checkpoint)!=g["checkpoint_sha256"]: raise ValueError("Gate checkpoint hash mismatch")
        gate=EnsembleGate(len(models),g["mode"]).to(device)
        gate.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True)
        if not all(torch.isfinite(p).all() for p in gate.parameters()): raise ValueError("Nonfinite gate weights")
        return [(1.,GatedEnsemble([m for _,m in models],gate).to(device).eval())]
    return models

def read_wav(path):
    x,sr=sf.read(path,dtype="float32",always_2d=True)
    if sr!=SR or x.shape!=(LENGTH,1) or not np.isfinite(x).all():
        raise ValueError("Expected 16kHz mono 48000 samples: "+str(path))
    return x[:,0]

@torch.inference_mode()
def predict(models,x,device):
    z=torch.from_numpy(np.ascontiguousarray(x))[None].to(device)
    out=torch.zeros_like(z)
    for w,model in models:
        y=model(z)
        if y.shape!=z.shape or not torch.isfinite(y).all():
            raise RuntimeError("Invalid model output")
        out+=w*y
    return out.cpu().numpy()[0].clip(-1,1)

def run_directory(models,inp,out,device):
    if inp.resolve()==out.resolve(): raise ValueError("Input and output must differ")
    files=sorted(p for p in inp.iterdir() if p.is_file() and p.suffix.lower()==".wav")
    if not files: raise ValueError("No WAV input")
    out.mkdir(parents=True,exist_ok=True)
    if any(p.is_file() and p.suffix.lower()==".wav" for p in out.rglob("*")):
        raise ValueError("Use an empty output directory (no stale WAV)")
    for p in files:
        y=predict(models,read_wav(p),device)
        target=out/p.name
        sf.write(target,y,SR,subtype="PCM_16")
        read_wav(target)
    expected={p.name for p in files}
    actual={p.name for p in out.iterdir() if p.is_file() and p.suffix.lower()==".wav"}
    if actual!=expected: raise RuntimeError("Output filenames mismatch")
    print("Verified",len(files),"output WAV files",flush=True)

def evaluate(models,root,device,out):
    from pesq import pesq
    root=root.resolve()
    with (root/"validation"/"samples.csv").open(encoding="utf-8-sig",newline="") as f:
        manifest=list(csv.DictReader(f))
    if len(manifest)!=164: raise ValueError("Expected 164 validation files")
    rows=[]
    for i,r in enumerate(manifest):
        x=read_wav(safe_path(root,r["noisy_path"]))
        c=read_wav(safe_path(root,r["clean_path"]))
        y=predict(models,x,device)
        # Score the quantized PCM-16 output, identical to submission representation.
        buffer=io.BytesIO()
        sf.write(buffer,y,SR,format="WAV",subtype="PCM_16")
        buffer.seek(0)
        y,sr=sf.read(buffer,dtype="float32")
        if sr!=SR or y.shape!=(LENGTH,): raise ValueError("Invalid PCM round-trip")
        p=float(pesq(SR,c,y,"wb"))
        snr=float(10*np.log10((np.sum(c.astype(np.float64)**2)+1e-12)/(np.sum((c.astype(np.float64)-y)**2)+1e-12)))
        if not np.isfinite([p,snr]).all(): raise ValueError("Invalid metric")
        rows.append((i,p,snr))
    out.mkdir(parents=True,exist_ok=True)
    with (out/"submission_validation.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f);writer.writerow(["index","pesq_wb","snr_db"]);writer.writerows(rows)
    summary={"validation_files":164,"pesq_wb":float(np.mean([r[1] for r in rows])),"snr_db":float(np.mean([r[2] for r in rows]))}
    (out/"submission_validation.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print(summary)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--input",type=Path,default=Path("input"))
    p.add_argument("--output",type=Path,default=Path("output"))
    p.add_argument("--device",default="cpu")
    p.add_argument("--validation-root",type=Path)
    args=p.parse_args()
    models=load_bundle(args.config.resolve(),torch.device(args.device))
    if args.validation_root: evaluate(models,args.validation_root,torch.device(args.device),args.output)
    else: run_directory(models,args.input,args.output,torch.device(args.device))

if __name__=="__main__": main()
