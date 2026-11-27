"""V6 CPU acceptance test: real pinned weights, noisy-only 48k, gradients and offline reconstruction."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import ralph_v6 as ctl
import ralph_v2 as core

REPORT=ctl.ROOT/"runs"/"v6_preflight"/"report.json"
ASSETS=ctl.ROOT/"runs"/"v6_preflight"/"mp_assets"
def sha_source(code): return hashlib.sha256(code.encode()).hexdigest()

def environment():
    import torch
    import numpy
    import soundfile
    return {"python":sys.version,"torch":str(torch.__version__),"numpy":numpy.__version__,"soundfile":soundfile.__version__}

def dataset_identity(root):
    return {split:ctl.sha(root/split/"samples.csv") for split in ("train","validation")}

def load_module(path):
    spec=importlib.util.spec_from_file_location("v6_local_probe",path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

def run(root,families,report=REPORT,assets=ASSETS):
    import torch
    import numpy as np
    import soundfile as sf
    from submit_v5 import safe_path
    import csv
    torch.set_num_threads(1)
    code=ctl.source()
    report.parent.mkdir(parents=True,exist_ok=True)
    runner=report.parent/"generated_v6.py";runner.write_text(code,encoding="utf-8")
    mod=load_module(runner)
    mod.WORK=report.parent/"local_assets"
    mod.WORK.mkdir(parents=True,exist_ok=True)
    with (root/"train/samples.csv").open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    if len(rows)!=1296: raise ValueError("Original train required")
    x,sr=sf.read(safe_path(root,rows[0]["noisy_path"]),dtype="float32",always_2d=True)
    if sr!=16000 or x.shape!=(48000,1): raise ValueError("Train clip contract")
    x=torch.from_numpy(x[:,0].copy())[None]
    from v572_scale_audit import require_report as scales
    from v571_preflight import require_report as legacy_report
    policies=scales(data_root=root)
    legacy=legacy_report()
    if legacy["sepformer_scale_policies"]!=policies: raise ValueError("Existing preflight policy mismatch")
    results={}
    for family in families:
        cfg=ctl.config(family,"train",sha_source(code),policies,epochs=2)
        mod.EXPERIMENT_CONFIG=cfg
        # Already accepted public architectures retain exact V5.7.2 model implementations.
        # New MP and native families execute full 48k and real gradients here.
        if family in mod.V6_PUBLIC:
            if family not in legacy["families"]: raise ValueError("Missing existing public family acceptance")
            results[family]={"status":"PASS","mode":"inherited_v571_offline_cpu","policy":cfg["sepformer_scale_policies"]}
            continue
        model=mod.v6_make_model(cfg,mp_root=assets).eval()
        with torch.no_grad():
            y=model(x)
        if y.shape!=x.shape or not torch.isfinite(y).all(): raise ValueError("Full 48000 output contract: "+family)
        # Load into a clean model without original weights; strict state equality and output.
        other=mod.v6_make_model(cfg,mp_root=assets,pretrained=False).eval()
        other.load_state_dict(model.state_dict(),strict=True)
        with torch.no_grad(): z=other(x)
        if not torch.allclose(y,z,rtol=2e-4,atol=2e-5): raise ValueError("Offline output mismatch: "+family)
        model.train()
        crop=x[:,:4096].clone()
        pred=model(crop)
        loss=mod.v6_loss(pred,crop,cfg)
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or any(not torch.isfinite(g).all() for g in grads) or not any(float(g.abs().max())>0 for g in grads): raise ValueError("Gradient contract: "+family)
        # The cascade must reproduce coarse output at initialization.
        refiner=mod.V6Refiner(other,2).eval()
        with torch.no_grad(): q=refiner(crop);b=other(crop)
        if not torch.allclose(q,b,rtol=2e-4,atol=2e-5): raise ValueError("Cascade identity contract")
        refiner.train();refiner(crop).abs().mean().backward()
        if any(p.grad is not None for p in refiner.coarse.parameters()) or not any(p.grad is not None and torch.isfinite(p.grad).all() for p in refiner.wave.parameters()): raise ValueError("Cascade freeze contract")
        results[family]={"status":"PASS","mode":"actual_cpu","samples":48000,"peak":float(y.abs().max()),"max_error":float((y-z).abs().max()),"loss":float(loss.detach())}
        print("V6 PREFLIGHT",family,json.dumps(results[family]),flush=True)
        del model,other,refiner
    payload={"protocol":"v6-preflight-v1","status":"PASS","source_sha256":sha_source(code),"dataset":dataset_identity(root),
             "environment":environment(),"families":results,"assets":str(assets.resolve())}
    core.json_write(report,payload)
    print("V6 PREFLIGHT PASS",len(results),flush=True)

def require(code_sha,root,families,report=REPORT):
    r=json.loads(Path(report).read_text(encoding="utf-8"))
    if r.get("protocol")!="v6-preflight-v1" or r.get("status")!="PASS" or r.get("source_sha256")!=code_sha or r.get("dataset")!=dataset_identity(root) or r.get("environment")!=environment():
        raise ValueError("V6 preflight missing/stale; rerun CPU acceptance")
    if any(r.get("families",{}).get(f,{}).get("status")!="PASS" for f in families): raise ValueError("Required family preflight missing")
    # Recheck pinned local blobs for each selected MP family.
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location("mp_assets_check",ctl.ROOT/"src/v6_mpsenet.py")
    mod=module_from_spec(spec);spec.loader.exec_module(mod)
    for f in families:
        if f in mod.V6_MP_FAMILIES: mod.v6_mp_assets(Path(r["assets"]),mod.V6_MP_FAMILIES[f],False)
    return r

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--families",nargs="+",choices=ctl.FAMILIES,default=list(ctl.FAMILIES))
    p.add_argument("--allow-public-download",action="store_true")
    a=p.parse_args()
    if not a.allow_public_download: p.error("Explicit public download permission required")
    run(a.data_root.resolve(),a.families)
if __name__=="__main__": main()
