"""V6.8 offline package runner. Only trusted hash-verified source and weights.
Uses unchanged V5.7 PCM16 evaluator and submission ZIP checks.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import torch
import submit_v5 as old
import v57_offline as offline
import v57_public_assets as public
from v6_gating import V6Ensemble,V6Gate

def checked(root,rel,digest):
    path=old.safe_path(root,rel)
    if old.sha256(path)!=digest: raise ValueError("Asset hash mismatch "+str(rel))
    return path

def load_expert(root,meta,index):
    if meta["kind"]=="legacy":
        path=checked(root,meta["bundle"],meta["sha256"])
        offline.verify_package(path.parent)
        models=old.load_bundle(path,torch.device("cpu"))
        from torch import nn
        class Legacy(nn.Module):
            def __init__(self,models):
                super().__init__();self.models=nn.ModuleList([m for _,m in models]);self.weights=[w for w,_ in models]
            def forward(self,x):
                return sum(w*m(x) for w,m in zip(self.weights,self.models))
        return Legacy(models).eval()
    if meta["kind"]!="v6": raise ValueError("Unknown expert kind")
    source=checked(root,meta["source"],meta["source_sha256"])
    cp=checked(root,meta["checkpoint"],meta["checkpoint_sha256"])
    spec=importlib.util.spec_from_file_location("v6_expert_"+str(index),source)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    cfg=mod.EXPERIMENT_CONFIG
    if cfg["architecture"]!=meta["family"]: raise ValueError("Expert family mismatch")
    base=cfg.get("v6_base_family") or cfg["architecture"]
    asset_root=None;coarse=None
    if base in mod.V6_MP_FAMILIES:
        asset_root=(root/meta["asset_root"]).resolve()
        if not asset_root.is_relative_to(root.resolve()): raise ValueError("Asset path escapes")
        mod.v6_mp_assets(asset_root,mod.V6_MP_FAMILIES[base],False,include_weights=False)
    elif base in mod.V6_PUBLIC:
        asset_root=(root/meta["asset_root"]).resolve()
        if not asset_root.is_relative_to(root.resolve()): raise ValueError("Asset path escapes")
        original=cfg["architecture"];cfg["architecture"]=base
        try: coarse=public.construct(mod,base,asset_root,meta["public_assets"])
        finally: cfg["architecture"]=original
    model=mod.v6_make_model(cfg,mp_root=asset_root,pretrained=False,coarse=coarse).eval()
    model.load_state_dict(torch.load(cp,map_location="cpu",weights_only=True),strict=True)
    return model

def load_package(root):
    root=root.resolve();offline.verify_package(root)
    cfg=json.loads((root/"v6_bundle.json").read_text(encoding="utf-8"))
    if cfg["protocol"]!="v6-offline-v1" or not 1<=len(cfg["experts"])<=5: raise ValueError("Invalid bundle")
    models=[load_expert(root,m,i) for i,m in enumerate(cfg["experts"])]
    if cfg.get("gate"):
        g=cfg["gate"];gate=V6Gate(len(models),g["mode"])
        gate.load_state_dict(torch.load(checked(root,g["path"],g["sha256"]),map_location="cpu",weights_only=True),strict=True)
        return V6Ensemble(models,cfg["alignment"],gate=gate).eval()
    return V6Ensemble(models,cfg["alignment"],weights=cfg["weights"]).eval()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package",type=Path,default=Path(__file__).resolve().parent)
    p.add_argument("--probe",action="store_true");p.add_argument("--data-root",type=Path);p.add_argument("--manifest",type=Path)
    p.add_argument("--input",type=Path);p.add_argument("--output",type=Path);p.add_argument("--count",type=int);p.add_argument("--zip",action="store_true")
    args=p.parse_args()
    root=args.package.resolve();torch.set_num_threads(2);offline.block_network()
    model=load_package(root);models=[(1.,model)];device=torch.device("cpu")
    if args.probe:
        x=np.sin(np.arange(48000,dtype=np.float32)*.0864)*.03
        old.predict(models,x,device)
        print("V6 OFFLINE PROBE PASS",flush=True);return
    if args.output is None: p.error("--output required")
    out=args.output.resolve()
    if out.exists() or out.is_relative_to(root): p.error("New output outside package required")
    if args.data_root:
        data=args.data_root.resolve()
        if out.is_relative_to(data): p.error("Output cannot change dataset")
        offline.evaluate(models,data,args.manifest or data/"validation/samples.csv",out,device,args.count)
    else:
        if not args.input: p.error("--input required")
        old.run_directory(models,args.input.resolve(),out,device)
        files=offline.verify_outputs(args.input.resolve(),out)
        (out.parent/(out.name+".sha256.json")).write_text(json.dumps(files,indent=2),encoding="utf-8")
        if args.zip:
            dest=out.parent/(out.name+".zip")
            with zipfile.ZipFile(dest,"x",compression=zipfile.ZIP_DEFLATED) as z:
                for n in sorted(files): z.write(out/n,arcname=n)
            with zipfile.ZipFile(dest) as z:
                if z.testzip() is not None or set(z.namelist())!=set(files) or any(hashlib.sha256(z.read(n)).hexdigest()!=h for n,h in files.items()): raise RuntimeError("ZIP verification failed")
        print("V6 VERIFIED SUBMISSION",len(files),flush=True)

if __name__=="__main__": main()
