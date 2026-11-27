"""V5.7.2 CPU-only SepFormer scaling ablation; select on train speakers only."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
import v571_preflight as pre
import ralph_v2 as core
import submit_v5 as s
from v51_data_audit import speaker,category
from v55_ensemble import pcm16
import kaggle_v4_external as trainer

REPORT=pre.ROOT/"runs/v572_scale_audit/report.json"
FAMILIES=("sepformer_dns_transfer","sepformer_wham_transfer")
MODES=trainer.SEPFORMER_SCALE_MODES
GUARD=.05

def split(rows,n=12):
    groups={}
    for i,r in enumerate(rows):
        g=speaker(r)
        if g!="unknown": groups.setdefault(g,[]).append(i)
    if len(groups)<4: raise ValueError("Need at least four known training speakers")
    rng=np.random.default_rng(572)
    names=sorted(groups);rng.shuffle(names)
    # Balance speaker counts and distribute samples across as many groups as possible.
    def sample(names):
        pools=[list(rng.permutation(groups[g])) for g in names]
        result=[]
        while len(result)<n and any(pools):
            for pool in pools:
                if pool and len(result)<n: result.append(int(pool.pop()))
        if len(result)!=n: raise ValueError("Insufficient train pairs")
        return sorted(result)
    cut=len(names)//2
    return sample(names[:cut]),sample(names[cut:])

def summarize(values):
    return {mode:{"pesq":float(np.mean([x["pesq"] for x in rows])),
                  "mean_clipping":float(np.mean([x["clipping"] for x in rows])),
                  "max_peak":max(x["peak"] for x in rows),
                  "mean_rms_ratio":float(np.mean([x["rms_ratio"] for x in rows])),
                  "count":len(rows)} for mode,rows in values.items()}

def choose(selection,holdout):
    # Select strictly on fit group; holdout may veto but never choose another mode.
    safe=[m for m in MODES if m!="raw" and selection[m]["mean_clipping"]==0. and selection[m]["max_peak"]<=.98001]
    if not safe: raise ValueError("No bounded mode on selection split")
    mode=max(safe,key=lambda m:selection[m]["pesq"])
    a,b=selection[mode],holdout[mode]
    approved=(b["mean_clipping"]==0. and b["max_peak"]<=.98001 and
              a["pesq"]>=selection["raw"]["pesq"]-GUARD and b["pesq"]>=holdout["raw"]["pesq"]-GUARD)
    return {"mode":mode,"status":"PASS" if approved else "FAIL",
            "selection_gain":a["pesq"]-selection["raw"]["pesq"],
            "holdout_gain":b["pesq"]-holdout["raw"]["pesq"]}

def require_report(path=REPORT,data_root=None):
    if not path.is_file(): raise RuntimeError("Run tools/v572_scale_audit.py before GPU execution")
    r=json.loads(path.read_text(encoding="utf-8"))
    if r.get("protocol")!="v572-scale-train-v1" or r.get("identity")!=pre.identity() or r.get("status")!="PASS" or set(r.get("families",{}))!=set(FAMILIES):
        raise RuntimeError("Missing, failed or stale amplitude audit")
    if data_root is None: raise RuntimeError("Amplitude audit requires --data-root to verify the train manifest")
    if s.sha256(data_root.resolve()/"train/samples.csv")!=r.get("train_manifest_sha256"): raise RuntimeError("Amplitude audit train manifest differs")
    policies={}
    for family,entry in r["families"].items():
        check=choose(entry["selection"],entry["holdout"])
        if check!=entry["selected"] or check["status"]!="PASS": raise RuntimeError("Invalid amplitude policy evidence")
        policies[family]=check["mode"]
    return policies

@torch.no_grad()
def inspect_model(family,root,rows,ids,cache):
    from pesq import pesq
    trainer.WORK=cache
    trainer.EXPERIMENT_CONFIG={"architecture":family,"seed":572}
    model=trainer.SepformerTransfer().eval()
    result={}
    for split_name,indices in ids.items():
        values={m:[] for m in MODES}
        for i in indices:
            row=rows[i]
            x=s.read_wav(s.safe_path(root,row["noisy_path"]))
            c=s.read_wav(s.safe_path(root,row["clean_path"]))
            z=torch.from_numpy(x)[None]
            raw=model.raw_forward(z)
            if raw.shape!=z.shape or not torch.isfinite(raw).all(): raise RuntimeError("Invalid raw SepFormer output")
            for mode in MODES:
                y=trainer.sepformer_scale_output(z,raw,mode).numpy()[0]
                p=float(pesq(16000,c,pcm16(y),"wb"))
                rms=float(np.sqrt(np.mean(y.astype(np.float64)**2))/max(1e-8,np.sqrt(np.mean(x.astype(np.float64)**2))))
                if not np.isfinite([p,rms]).all(): raise RuntimeError("Nonfinite scale score")
                values[mode].append({"index":i,"pesq":p,"clipping":float(np.mean(np.abs(y)>1)),"peak":float(np.max(np.abs(y))),"rms_ratio":rms})
            print("V572",family,split_name,i,"raw peak",float(raw.abs().max()),flush=True)
        result[split_name]=summarize(values)
        result[split_name+"_rows"]=values
    result["selected"]=choose(result["selection"],result["holdout"])
    return result

def run(args):
    import v55_ensemble as ens
    pre.cpu_numeric_settings()
    root=args.data_root.resolve();manifest,rows=ens.load_train(root)
    a,b=split(rows,args.samples_per_split)
    out=args.out.resolve()
    if out.is_relative_to(root): raise ValueError("Audit output must be outside dataset")
    out.mkdir(parents=True,exist_ok=True)
    cache=out/"assets_cache";cache.mkdir(exist_ok=True)
    report={"protocol":"v572-scale-train-v1","identity":pre.identity(),"status":"RUNNING",
            "train_manifest_sha256":s.sha256(manifest),"selection_indices":a,"holdout_indices":b,
            "selection_speakers":sorted({speaker(rows[i]) for i in a}),"holdout_speakers":sorted({speaker(rows[i]) for i in b}),
            "families":{},"guard_pesq":GUARD,"limitations":"Small train-only speaker-disjoint engineering study, not independent test; no GPU training."}
    core.json_write(out/"report.json",report)
    for family in FAMILIES:
        report["families"][family]=inspect_model(family,root,rows,{"selection":a,"holdout":b},cache)
        core.json_write(out/"report.json",report)
        print("V572 SELECTED",family,report["families"][family]["selected"],flush=True)
    report["status"]="PASS" if all(v["selected"]["status"]=="PASS" for v in report["families"].values()) else "FAIL"
    core.json_write(out/"report.json",report)
    print("V572 SCALE AUDIT",report["status"],out/"report.json",flush=True)
    if report["status"]!="PASS": raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--out",type=Path,default=REPORT.parent)
    p.add_argument("--samples-per-split",type=int,default=12,choices=range(8,65))
    args=p.parse_args()
    core.no_uncertain_prior_jobs()
    lock=core.OUTPUT/"active.lock";lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("x",encoding="utf-8") as f: f.write(str(os.getpid()))
    try: run(args)
    finally: lock.unlink(missing_ok=True)

if __name__=="__main__": main()
