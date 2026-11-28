"""Read-only V6 research while the GPU controller is active. Private local data only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import ralph_v6 as ctl
import ralph_v2 as core
import ralph_v4 as v4
import v6_finalize as fin
import v6_offline as off
import v57_offline as legacy
import submit_v5 as wav
from v6_gating import V6Gate,v6_align,v6_combine

ROOT=ctl.ROOT/"runs"/"v6_refine"
def stamp(): return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
def write(path,obj): core.json_write(path,obj)

def snapshot():
    """Only immutable COMPLETE entries; never touch active lock or remote state."""
    entries=[];rejected=[]
    for p in sorted(core.OUTPUT.glob("run_v4*/round_*/*/remote_state.json")):
        try:
            st=json.loads(p.read_text(encoding="utf-8"))
            if not st.get("v6") or st.get("status")!="COMPLETE" or st.get("kind")=="v4_smoke": continue
            r,cfg=ctl.verify(p.parent,st)
            if r.get("smoke") or r.get("evaluation_protocol")!="v4.2-fixed-dev-v2": continue
            if r.get("transductive_noisy") or r.get("self_supervised"): continue
            entries.append((st,r,cfg,p.parent))
        except (OSError,ValueError,KeyError,TypeError) as exc: rejected.append({"path":str(p.parent),"reason":repr(exc)})
    entries.sort(key=lambda a:(-a[1]["best_validation_pesq_wb"],a[0]["ref"]))
    return entries,rejected

def shortlist(entries,top):
    chosen=[];seen=set();hashes=set()
    for distinct in (True,False):
        for e in entries:
            st,r,cfg,folder=e
            family=(cfg.get("v6_base_family") or cfg["architecture"])
            h=r["v6"]["checkpoint_sha256"]
            if h in hashes or (distinct and family in seen): continue
            chosen.append(e);seen.add(family);hashes.add(h)
            if len(chosen)>=top:return chosen
    return chosen

def metadata(row,x,c):
    from v51_data_audit import category
    s=float(10*np.log10((np.sum(c.astype("float64")**2)+1e-12)/(np.sum((x.astype("float64")-c)**2)+1e-12)))
    band=next((label for edge,label in ((0,"below 0"),(5,"0–<5"),(10,"5–<10"),(15,"10–<15"),(20,"15–<20")) if round(s,1)<edge),"20+")
    return {"noise":category(row,("noise_type","noise_category","noise_label","noise_class")) or "unknown","snr":s,"snr_group":band}

def load_pairs(data,rows,ids):
    x=[];c=[];meta=[]
    for i in ids:
        row=rows[i]
        a=wav.read_wav(wav.safe_path(data,row["noisy_path"]))
        b=wav.read_wav(wav.safe_path(data,row["clean_path"]))
        if a.shape!=(48000,) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError("Invalid paired clip")
        x.append(a);c.append(b);meta.append({"index":i,**metadata(row,a,b)})
    return torch.from_numpy(np.stack(x)),torch.from_numpy(np.stack(c)),meta

def scores(c,p,metric=None):
    if metric is None:
        from pesq import pesq
        metric=lambda a,b:pesq(16000,a,b,"wb")
    out=np.asarray([metric(a,fin.pcm(b)) for a,b in zip(c.detach().cpu().numpy(),p.detach().cpu().numpy())],dtype=float)
    if len(out)!=len(c) or not np.isfinite(out).all():raise ValueError("Nonfinite PESQ")
    return out

def group_summary(matrix,meta,names):
    matrix=np.asarray(matrix,dtype=float)
    if matrix.shape!=(len(meta),len(names)) or not np.isfinite(matrix).all():raise ValueError("Invalid score matrix")
    report={}
    for key in ("noise","snr_group"):
        groups={}
        for k in sorted({m[key] for m in meta}):
            idx=[i for i,m in enumerate(meta) if m[key]==k]
            a=matrix[idx];means=a.mean(0)
            groups[k]={"count":len(idx),"mean":dict(zip(names,means.tolist())),"best":names[int(means.argmax())],
                       "best_mean":float(means.max()),"oracle_per_clip":float(a.max(1).mean())}
        report[key]=groups
    return report

def matrix_summary(matrix,names):
    a=np.asarray(matrix,dtype=float)
    if a.ndim!=2 or a.shape[1]!=len(names) or not np.isfinite(a).all():raise ValueError("Invalid score matrix")
    means=a.mean(0);best=int(means.argmax())
    # Pairwise failure overlap (bottom quartile of each model) and complementary wins.
    fail=a<=np.quantile(a,.25,axis=0,keepdims=True)
    overlap={}
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            union=(fail[:,i]|fail[:,j]).sum()
            overlap[names[i]+" | "+names[j]]={"failure_jaccard":float((fail[:,i]&fail[:,j]).sum()/max(1,union)),
                "j_beats_i_by_0.1":int((a[:,j]-a[:,i]>.1).sum()),"i_beats_j_by_0.1":int((a[:,i]-a[:,j]>.1).sum())}
    return {"means":dict(zip(names,means.tolist())),"best":names[best],"best_mean":float(means[best]),
        "oracle_per_clip":float(a.max(1).mean()),"oracle_gap":float(a.max(1).mean()-means[best]),
        "below_3":dict(zip(names,(a<3).sum(0).tolist())),"below_4":dict(zip(names,(a<4).sum(0).tolist())),
        "pairwise":overlap,"worst_indices":np.argsort(a[:,best])[:10].tolist()}

def csv_scores(path,meta,matrix,names):
    with path.open("w",encoding="utf-8",newline="") as f:
        fields=["index","noise","snr","snr_group",*names]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row,vals in zip(meta,matrix):w.writerow({**row,**dict(zip(names,map(float,vals)))})

def cached_predictions(cache,model,x,identity):
    # Key includes actual input tensor, archived model checkpoint/source and inference environment.
    # Private enhanced audio only on local disk, no upload.
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()+x.numpy().tobytes()).hexdigest()
    cache.mkdir(parents=True,exist_ok=True);p=cache/(key+".npy");m=cache/(key+".json")
    if p.is_file() and m.is_file():
        d=json.loads(m.read_text(encoding="utf-8"))
        if d.get("sha256")==ctl.sha(p):
            a=np.load(p,allow_pickle=False)
            if a.shape==tuple(x.shape) and np.isfinite(a).all():return torch.from_numpy(a.copy())
    ys=[]
    with torch.no_grad():
        for i,a in enumerate(x):
            y=model(a[None]).detach().cpu().float()
            if y.shape!=(1,48000) or not torch.isfinite(y).all():raise ValueError("Invalid expert prediction")
            ys.append(y[0])
            if (i+1)%32==0: print("V6 REFINE CPU",identity["ref"],i+1,"/",len(x),flush=True)
    a=torch.stack(ys).numpy();tmp=p.with_suffix(".partial")
    with tmp.open("wb") as f:np.save(f,a,allow_pickle=False)
    tmp.replace(p);write(m,{"sha256":ctl.sha(p),"identity":identity})
    return torch.from_numpy(a)

def oracle_diagnostic(clean,pred):
    # Diagnostic only on reserved train speakers. True magnitude/phase NEVER used in inference.
    w=torch.hann_window(512);c=torch.stft(clean,512,128,window=w,return_complex=True)
    p=torch.stft(pred,512,128,window=w,return_complex=True)
    mag=torch.istft(torch.polar(c.abs(),p.angle()),512,128,window=w,length=clean.shape[-1])
    phase=torch.istft(torch.polar(p.abs(),c.angle()),512,128,window=w,length=clean.shape[-1])
    a=scores(clean,pred);b=scores(clean,mag);d=scores(clean,phase)
    return {"base":float(a.mean()),"oracle_true_magnitude":float(b.mean()),"oracle_true_phase":float(d.mean()),
        "magnitude_gain":float((b-a).mean()),"phase_gain":float((d-a).mean()),"count":len(a),
        "caveat":"Target-informed counterfactual diagnostic, not achievable scores or rigorous bounds."}

def hypothesis(diag,groups,best_name):
    # Choose one controlled hypothesis; no claim of causal effect from descriptive groups.
    low=[v for k,v in groups["snr_group"].items() if k in ("below 0","0–<5")]
    other=[v for k,v in groups["snr_group"].items() if k not in ("below 0","0–<5")]
    def mean(v):
        n=sum(x["count"] for x in v)
        return sum(x["count"]*x["mean"][best_name] for x in v)/n if n else None
    a,b=mean(low),mean(other)
    if a is not None and b is not None and b-a>=.4 and diag["magnitude_gain"]>diag["phase_gain"]+.1:
        return {"name":"hard_noise","reason":"Reserved train low-SNR gap >=0.4 and magnitude oracle gain exceeds phase by >0.1.","low_snr_mean":a,"other_snr_mean":b}
    return {"name":"full_length","reason":"Test 3-second context instead of 1-second training crop, preserving existing loss/data recipe.","low_snr_mean":a,"other_snr_mean":b}

def predict_recipe(x,p,best,state):
    pp=v6_align(x,p,best["alignment"])
    if state is None:
        w=torch.tensor(best["weights"],dtype=x.dtype)[None,:,None,None].expand(len(x),-1,-1,-1)
        return v6_combine(x,pp,w)
    gate=V6Gate(p.shape[1],best["mode"]);gate.load_state_dict(state,strict=True);gate.eval()
    with torch.no_grad():return torch.cat([v6_combine(a,b,gate(a)) for a,b in zip(x.split(4),pp.split(4))])

def analyze(data,out,top=3,threads=2,gate_epochs=8,allow_download=False,cache=ROOT/"predictions"):
    if (core.OUTPUT/"v57_audit_seal.json").exists():raise RuntimeError("Independent audit sealed")
    data=data.resolve();out=out.resolve()
    if out.exists() or out.is_relative_to(data) or out.is_relative_to(core.OUTPUT.resolve()):raise ValueError("Use a fresh analysis directory outside dataset/history")
    torch.set_num_threads(threads)
    entries,rejected=snapshot()
    if not entries:raise ValueError("No verified complete V6 results")
    out.mkdir(parents=True)
    val_manifest=data/"validation/samples.csv";train_manifest=data/"train/samples.csv"
    val=legacy.read_manifest(val_manifest,164);train=legacy.read_manifest(train_manifest,1296)
    vx,vc,vm=load_pairs(data,val,range(164))
    # Quick diagnostic across all completed results, without model inference.
    names=[e[0]["ref"].rsplit("/",1)[-1] for e in entries]
    raw=np.stack([v4.verify_rows(e[3]/"validation_per_file.csv",164,e[1]["best_validation_pesq_wb"]) for e in entries],axis=1)
    quick={"protocol":"v6-raw-float-validation-diagnostics-v1","count":164,"models":[{"ref":e[0]["ref"],"family":e[1]["model_family"],"score":e[1]["best_validation_pesq_wb"],"checkpoint":e[1]["v6"]["checkpoint_sha256"]} for e in entries],
           "scores":matrix_summary(raw,names),"groups":group_summary(raw,vm,names),"rejected":rejected}
    write(out/"quick_report.json",quick);csv_scores(out/"raw_scores.csv",vm,raw,names)
    print("V6 REFINE QUICK",json.dumps(quick["scores"]),flush=True)
    fit_ids,select_ids=fin.train_splits(data)
    fx,fc,fm=load_pairs(data,train,fit_ids);sx,sc,sm=load_pairs(data,train,select_ids)
    from v6_preflight import ASSETS
    cmgan=None
    try:
        from v571_preflight import require_report
        cmgan=Path(require_report()["cmgan_root"])
    except (OSError,ValueError,KeyError,RuntimeError):pass
    package=out/"package";package.mkdir()
    chosen=shortlist(entries,top);accepted=[];specs=[];preds={"fit":[],"select":[],"val":[]}
    for i,e in enumerate(chosen):
        st,r,cfg,folder=e
        try:
            spec=fin.package_expert(package,i,(r["best_validation_pesq_wb"],"v6",r["model_family"],folder),ASSETS,cmgan,ctl.sha(train_manifest),allow_download)
            model=off.load_expert(package,spec,i)
            identity={"ref":st["ref"],"checkpoint":r["v6"]["checkpoint_sha256"],"source":spec["source_sha256"],"torch":str(torch.__version__),"threads":threads}
            got={k:cached_predictions(cache,model,x,{**identity,"split":k}) for k,x in (("fit",fx),("select",sx),("val",vx))}
            for k in preds:preds[k].append(got[k])
            accepted.append(e);specs.append(spec);del model
        except Exception as exc:
            rejected.append({"ref":st["ref"],"reason":repr(exc)})
            shutil.rmtree(package/f"expert_{i:02d}",ignore_errors=True)
            print("V6 REFINE REJECTED",st["ref"],repr(exc),flush=True)
    if not accepted:raise RuntimeError("No usable CPU experts")
    names=[e[0]["ref"].rsplit("/",1)[-1] for e in accepted]
    fp,sp,vp=[torch.stack(preds[k],dim=1) for k in ("fit","select","val")]
    # Fit on one Train speaker pool; choose recipe only on another Train pool.
    best,candidates,state=fin.search_gate((fx,fp,fc),(sx,sp,sc),gate_epochs)
    val_scores=np.stack([scores(vc,vp[:,i]) for i in range(len(names))],axis=1)
    train_scores=np.stack([scores(sc,sp[:,i]) for i in range(len(names))],axis=1)
    blend=predict_recipe(vx,vp,best,state)
    blend_scores=scores(vc,blend)
    best_i=int(val_scores.mean(0).argmax());best_name=names[best_i]
    matrix=np.column_stack((val_scores,blend_scores))
    csv_scores(out/"pcm16_scores.csv",vm,matrix,[*names,"train_selected_recipe"])
    csv_scores(out/"train_select_scores.csv",sm,train_scores,names)
    groups=group_summary(train_scores,sm,names)
    diag=oracle_diagnostic(sc,sp[:,best_i])
    hp=hypothesis(diag,groups,best_name)
    # Validation only chooses between frozen train-selected recipe and individual experts.
    if float(blend_scores.mean())>float(val_scores[:,best_i].mean())+.001:
        config={"protocol":"v6-offline-v1","experts":specs,"alignment":best["alignment"]}
        if state is not None:
            torch.save(state,package/"gate.pt")
            config["gate"]={"mode":best["mode"],"path":"gate.pt","sha256":ctl.sha(package/"gate.pt")}
        else:config["weights"]=best["weights"]
        winner="train_selected_recipe";winner_score=float(blend_scores.mean())
    else:
        config={"protocol":"v6-offline-v1","experts":[specs[best_i]],"alignment":"none","weights":[1.]}
        winner=best_name;winner_score=float(val_scores[:,best_i].mean())
    fin.seal_package(package,config)
    # Verify actual offline reconstructed package, independently of cached waveform paths.
    fin.invoke(package,"--probe")
    actual=out/"winner_replay"
    fin.invoke(package,"--data-root",data,"--manifest",val_manifest,"--count",164,"--output",actual)
    replay=json.loads((actual/"summary.json").read_text(encoding="utf-8"))
    if replay["count"]!=164 or replay["manifest_sha256"]!=ctl.sha(val_manifest) or abs(float(replay["pesq_wb"])-winner_score)>.005:
        raise ValueError("Offline winner score failed cached PCM16 reproduction")
    # Keep training parent the highest verified single checkpoint, not an oracle/gate bundle.
    champion=entries[0]
    report={"protocol":"v6-refine-v1","snapshot_champion":ctl.parent_info(champion[0],champion[1]),"train_manifest_sha256":ctl.sha(train_manifest),
        "validation_manifest_sha256":ctl.sha(val_manifest),"count":164,"models":names,"raw":quick["scores"],
        "pcm16":matrix_summary(val_scores,names),"validation_groups":group_summary(val_scores,vm,names),
        "train_select_groups":groups,"diagnostic_model":best_name,"train_oracle_diagnostic":diag,
        "train_fit_count":len(fit_ids),"train_select_count":len(select_ids),"gate_candidates":candidates,"gate_recipe":best,
        "ensemble_pesq":float(blend_scores.mean()),"winner":winner,"winner_pesq":float(replay["pesq_wb"]),
        "winner_package_manifest":ctl.sha(package/"package_manifest.json"),"hypothesis":hp,"rejected":rejected,
        "caveat":"Development validation/model selection, not private test performance. Train pools may not be unseen by legacy checkpoints."}
    write(out/"report.json",report)
    print("V6 REFINE RESULT",json.dumps({k:report[k] for k in ("winner","winner_pesq","ensemble_pesq","hypothesis")}),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True);p.add_argument("--out",type=Path)
    p.add_argument("--top",type=int,choices=range(1,6),default=3);p.add_argument("--threads",type=int,choices=range(1,5),default=2)
    p.add_argument("--gate-epochs",type=int,choices=range(1,31),default=8);p.add_argument("--allow-public-download",action="store_true")
    a=p.parse_args();analyze(a.data_root,a.out or ROOT/("analysis_"+stamp()),a.top,a.threads,a.gate_epochs,a.allow_public_download)
if __name__=="__main__":main()
