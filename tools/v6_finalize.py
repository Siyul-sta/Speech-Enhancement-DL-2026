"""V6 model selection, train-only gate calibration and frozen PCM16 submission package."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import ralph_v6 as ctl
import ralph_v2 as core
import ralph_v4 as v4
import v57_finalize as old_final
import v57_offline as offline
import v57_public_assets as public
import submit_v5 as old
import v6_offline as runner
from v6_gating import v6_align,v6_combine,v6_fit_gate

TOOLS=Path(__file__).resolve().parent

def write(p,obj): core.json_write(p,obj)

def shortlist(limit):
    pool=[]
    for st,r,cfg,folder in ctl.ledger().values():
        if not r or r["smoke"]: continue
        pool.append((float(r["best_validation_pesq_wb"]),"v6",r["model_family"],folder))
    for score,kind,folder in old_final.candidate_list(v4.history(),list((core.OUTPUT/"v56_ensembles").glob("ensemble_*")),max(12,limit)):
        r=json.loads((folder/("result.json" if kind!="ensemble" else "ensemble_report.json")).read_text(encoding="utf-8"))
        if r.get("v6"): continue
        pool.append((score,kind,r.get("model_family",kind),folder))
    pool.sort(key=lambda x:(-x[0],str(x[3])))
    used=set();families=set();selected=[]
    for distinct in (True,False):
        for score,kind,family,folder in pool:
            if distinct and family in families: continue
            asset=folder/("best_model.pt" if kind in ("v6","single") else "bundle.json" if kind=="ensemble" else "result.json")
            key=ctl.sha(asset)
            if key in used: continue
            selected.append((score,kind,family,folder));used.add(key);families.add(family)
            if len(selected)>=limit: return selected
    return selected

def package_expert(root,index,row,mp_root,cmgan_root,train_sha,allow_download):
    score,kind,family,folder=row
    dest=root/f"expert_{index:02d}";dest.mkdir(parents=True)
    if kind!="v6":
        pack=dest/"legacy"
        if kind=="single": old_final.package_single(folder,pack,allow_download,cmgan_root)
        elif kind=="recipe": old_final.package_recipe(folder,pack,allow_download,cmgan_root)
        elif kind=="ensemble": old_final.package_ensemble(folder,pack,train_sha)
        else: raise ValueError("Unknown legacy candidate")
        b=pack/"bundle.json"
        return {"kind":"legacy","bundle":b.relative_to(root).as_posix(),"sha256":ctl.sha(b)}
    state=json.loads((folder/"remote_state.json").read_text(encoding="utf-8"))
    r,cfg=ctl.verify(folder,state)
    if state["status"]!="COMPLETE" or r["smoke"]: raise ValueError("Unverified V6 candidate")
    for n in ("source.py","best_model.pt"): shutil.copy2(folder/n,dest/n)
    spec={"kind":"v6","family":family,"source":(dest/"source.py").relative_to(root).as_posix(),"source_sha256":ctl.sha(dest/"source.py"),
          "checkpoint":(dest/"best_model.pt").relative_to(root).as_posix(),"checkpoint_sha256":ctl.sha(dest/"best_model.pt")}
    base=cfg.get("v6_base_family") or family
    if base.startswith("mp_"):
        import importlib.util
        sp=importlib.util.spec_from_file_location("v6_mp_package",ctl.ROOT/"src/v6_mpsenet.py")
        mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
        root_src=mod.v6_mp_assets(mp_root,mod.V6_MP_FAMILIES[base],allow_download,include_weights=False)
        assets=dest/"assets"
        for n in ("models/model.py","models/transformer.py","utils.py","LICENSE","best_ckpt/config.json"):
            p=assets/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root_src/n,p)
        spec["asset_root"]=assets.relative_to(root).as_posix()
    elif base in ("semamba","semambapp","rose_cd"):
        # Use the exact archived source and pinned asset metadata, not current model code.
        import importlib.util
        sp=importlib.util.spec_from_file_location("v6_advanced_package_"+str(index),dest/"source.py")
        mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
        src=Path(mp_root)/base;assets=dest/"assets"
        if base=="semamba":
            mod.v6_sem_assets(src,allow_download,include_weights=False)
            names=[n for n in mod.V6_SEM_FILES if not n.endswith(".pth")]
        elif base=="semambapp":
            mod.v6_pp_assets(src,allow_download,include_weights=False)
            names=list(mod.V6_PP_FILES)
        else:
            mod.v6_rose_assets(src,allow_download,include_weights=False)
            names=list(mod.V6_ROSE_FILES)
        for n in names:
            target=assets/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/n,target)
        if base in ("semambapp","rose_cd"):
            lock=json.loads((folder/"v6_hf_lock.json").read_text(encoding="utf-8"))
            if lock!=cfg.get("v6_hf_lock"): raise ValueError("Advanced HF lock mismatch")
            write(assets/"hf_lock.json",lock)
        if base=="rose_cd":
            shutil.copy2(folder/"rose_config.json",assets/"rose_config.json")
        spec["asset_root"]=assets.relative_to(root).as_posix()
        if base=="semambapp":
            (assets/"UPSTREAM_NOTICE.txt").write_text("Personal local use: upstream source Eddieyjlee/SEMambapp pinned by blob. Check source redistribution rights and GPL-2.0 checkpoint terms before sharing. Do not upload this package publicly.",encoding="utf-8")
    elif base in public.PUBLIC:
        assets=dest/"assets"
        spec["public_assets"]=public.package_public(base,folder,(dest/"source.py").read_text(encoding="utf-8"),assets,allow_download,cmgan_root)
        spec["asset_root"]=assets.relative_to(root).as_posix()
    return spec

def pcm(y):
    import io
    buf=io.BytesIO();sf.write(buf,np.clip(y,-1,1),16000,format="WAV",subtype="PCM_16")
    buf.seek(0);return sf.read(buf,dtype="float32")[0]

def score(c,y):
    from pesq import pesq
    vals=[float(pesq(16000,a,pcm(b),"wb")) for a,b in zip(c.detach().cpu().numpy(),y.detach().cpu().numpy())]
    if not vals or not np.isfinite(vals).all(): raise ValueError("Invalid selection PESQ")
    return float(np.mean(vals))

@torch.no_grad()
def collect(root,rows,ids,models):
    x=[];p=[];c=[]
    for i in ids:
        row=rows[i]
        a=torch.from_numpy(old.read_wav(old.safe_path(root,row["noisy_path"])))[None]
        s=torch.from_numpy(old.read_wav(old.safe_path(root,row["clean_path"])))
        pred=torch.stack([m(a)[0].float() for m in models])
        if pred.shape!=(len(models),48000) or not torch.isfinite(pred).all(): raise ValueError("Expert output contract")
        x.append(a[0]);p.append(pred);c.append(s)
    return torch.stack(x),torch.stack(p),torch.stack(c)

def train_splits(data):
    # Use exact V6 speaker-disjoint pools; old experts may have seen them.
    from v6_preflight import load_module
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/"source.py";p.write_text(ctl.source(),encoding="utf-8");m=load_module(p)
        train=m.SpeechPairs(data,"train")
        _,cal,fit,select=m.v6_split(train)
    rng=np.random.default_rng(6026)
    return sorted(rng.choice(fit,min(64,len(fit)),replace=False).tolist()),sorted(rng.choice(select,32,replace=False).tolist())

def search_gate(fit,select,epochs):
    x,p,c=fit;a,q,b=select;m=p.shape[1]
    candidates=[]
    # Noisy-only transforms; use only train-select real PESQ to choose.
    for i in range(m):
        w=[float(k==i) for k in range(m)]
        candidates.append({"alignment":"none","weights":w,"score":score(b,q[:,i]),"type":"single"})
    if m<2: return candidates[0],candidates,None
    for align in ("none","rms","lag_rms"):
        pp=v6_align(x,p,align);qq=v6_align(a,q,align)
        weights=[tuple(1/m for _ in range(m))]
        for i,j in itertools.combinations(range(m),2):
            for alpha in (.25,.5,.75):
                weights.append(tuple(alpha if k==i else 1-alpha if k==j else 0. for k in range(m)))
        for w in sorted(set(weights)):
            y=v6_combine(a,qq,torch.tensor(w)[None,:,None,None].expand(len(a),-1,-1,-1))
            candidates.append({"alignment":align,"weights":list(w),"score":score(b,y),"type":"fixed"})
        for mode in ("clip","frequency","timefreq"):
            gate=v6_fit_gate(x,pp,c,mode,epochs)
            with torch.no_grad(): y=v6_combine(a,qq,gate(a))
            candidates.append({"alignment":align,"type":"gate","mode":mode,"score":score(b,y),"state":gate.state_dict()})
    best_single=max((r for r in candidates if r["type"]=="single"),key=lambda r:r["score"])
    best=max(candidates,key=lambda r:r["score"])
    if best["score"]<best_single["score"]+.02: best=best_single
    state=best.get("state")
    return {k:v for k,v in best.items() if k!="state"},[{k:v for k,v in r.items() if k!="state"} for r in candidates],state

def seal_package(root,config):
    write(root/"v6_bundle.json",config)
    old_final.finalize_package(root)
    for n in ("v6_gating.py","v6_offline.py"): shutil.copy2(TOOLS/n,root/n)
    (root/"README.txt").write_text("Trusted noisy-only V6 bundle. Install compatible packages before offline use.\npython v6_offline.py --probe\npython v6_offline.py --input <noisy> --output <new_dir> --zip\nPython network guard is not an OS sandbox.\n",encoding="utf-8")
    write(root/"package_manifest.json",offline.package_manifest(root))
    offline.verify_package(root)

def invoke(root,*args):
    subprocess.run([sys.executable,str(root/"v6_offline.py"),"--package",str(root),*map(str,args)],check=True)

def finalize(args):
    core.no_uncertain_prior_jobs()
    if (core.OUTPUT/"v57_audit_seal.json").exists(): raise RuntimeError("Independent audit already sealed")
    data=args.data_root.resolve();out=args.out.resolve()
    if out.exists() or out.is_relative_to(data) or out.is_relative_to(core.OUTPUT.resolve()): raise ValueError("New output outside dataset and history")
    train=offline.read_manifest(data/"train/samples.csv",1296)
    val=data/"validation/samples.csv";offline.read_manifest(val,164)
    fit_ids,select_ids=train_splits(data)
    train_sha=ctl.sha(data/"train/samples.csv")
    from v6_preflight import ASSETS
    mp_root=args.mp_root or ASSETS
    cmgan_root=args.cmgan_root
    if cmgan_root is None:
        try:
            from v571_preflight import require_report
            cmgan_root=Path(require_report()["cmgan_root"])
        except (OSError,RuntimeError,ValueError,KeyError): pass
    pool=shortlist(args.top)
    if not pool: raise ValueError("No verified development candidates")
    out.mkdir(parents=True)
    root=out/"package";root.mkdir()
    specs=[];models=[];accepted=[];rejected=[]
    torch.set_num_threads(2)
    for i,row in enumerate(pool):
        try:
            spec=package_expert(root,i,row,mp_root,cmgan_root,train_sha,args.allow_public_download)
            model=runner.load_expert(root,spec,i)
            with torch.no_grad():
                probe=model(torch.sin(torch.arange(48000)[None]*.0864)*.03)
            if probe.shape!=(1,48000) or not torch.isfinite(probe).all(): raise ValueError("Expert probe failed")
            specs.append(spec);models.append(model);accepted.append({"family":row[2],"prior_score":row[0],"path":str(row[3])})
        except Exception as exc:
            rejected.append({"family":row[2],"path":str(row[3]),"reason":repr(exc)})
            shutil.rmtree(root/f"expert_{i:02d}",ignore_errors=True)
    if not models: raise RuntimeError("All candidates rejected: "+str(rejected))
    fit=collect(data,train,fit_ids,models);select=collect(data,train,select_ids,models)
    best,candidates,state=search_gate(fit,select,args.gate_epochs)
    config={"protocol":"v6-offline-v1","experts":specs,"alignment":best["alignment"]}
    if state is not None:
        cp=root/"gate.pt";torch.save(state,cp);config["gate"]={"mode":best["mode"],"path":"gate.pt","sha256":ctl.sha(cp)}
    else: config["weights"]=best["weights"]
    seal_package(root,config)
    # Evaluate selected train-only combination and every single model as fallback
    # on identical real PCM16 validation. Freeze the highest development score.
    tests=[("ensemble",root)]
    for i,spec in enumerate(specs):
        single=out/f"single_{i:02d}";shutil.copytree(root,single)
        cp=single/"gate.pt";cp.unlink(missing_ok=True)
        seal_package(single,{"protocol":"v6-offline-v1","experts":[spec],"alignment":"none","weights":[1.]})
        tests.append((f"single_{i:02d}",single))
    scores=[]
    for name,p in tests:
        try:
            invoke(p,"--probe")
            dest=out/("evaluation_"+name)
            invoke(p,"--data-root",data,"--manifest",val,"--count",164,"--output",dest)
            summary=json.loads((dest/"summary.json").read_text(encoding="utf-8"))
            if summary["manifest_sha256"]!=ctl.sha(val) or summary["count"]!=164: raise ValueError("Wrong validation identity")
            scores.append({"name":name,"status":"COMPLETE","pesq":summary["pesq_wb"],"package":str(p),"summary":summary})
        except Exception as exc: scores.append({"name":name,"status":"REJECTED","reason":repr(exc)})
    success=[x for x in scores if x["status"]=="COMPLETE"]
    report={"protocol":"v6-finalize-v1","accepted":accepted,"rejected":rejected,"train_fit_count":len(fit_ids),"train_select_count":len(select_ids),
            "train_manifest_sha256":train_sha,"validation_manifest_sha256":ctl.sha(val),"gate_selection":best,"gate_candidates":candidates,"validation_results":scores,
            "caveat":"Legacy experts may have seen the gate training/selection speakers; independent audit is still necessary."}
    if not success:
        write(out/"final_report.json",report);raise RuntimeError("All offline development evaluations failed")
    winner=max(success,key=lambda r:r["pesq"])
    frozen=out/"champion";shutil.copytree(Path(winner["package"]),frozen);offline.verify_package(frozen)
    report["champion"]={"pesq":winner["pesq"],"name":winner["name"],"package":str(frozen),"manifest_sha256":ctl.sha(frozen/"package_manifest.json")}
    write(out/"final_report.json",report)
    print("V6 CHAMPION",winner["pesq"],frozen,flush=True)

def audit(args):
    core.no_uncertain_prior_jobs()
    final=args.final.resolve();package=final/"champion"
    report=json.loads((final/"final_report.json").read_text(encoding="utf-8"))
    offline.verify_package(package)
    if ctl.sha(package/"package_manifest.json")!=report["champion"]["manifest_sha256"]: raise ValueError("Frozen champion changed")
    data=args.data_root.resolve();audit_root=args.audit_root.resolve();manifest=args.manifest.resolve()
    if data==audit_root or audit_root.is_relative_to(data) or data.is_relative_to(audit_root) or not manifest.is_relative_to(audit_root): raise ValueError("Audit root must be independent")
    if ctl.sha(data/"train/samples.csv")!=report["train_manifest_sha256"] or ctl.sha(data/"validation/samples.csv")!=report["validation_manifest_sha256"]: raise ValueError("Challenge manifests changed")
    if not args.confirm_heldout: raise ValueError("Confirm independent audit provenance before scoring")
    overlap=old_final.audit_disjoint(data,audit_root,manifest)
    marker=core.OUTPUT/"v57_audit_seal.json";marker.parent.mkdir(parents=True,exist_ok=True)
    payload={"status":"STARTED","champion":report["champion"]["manifest_sha256"],"manifest":ctl.sha(manifest),"provenance":args.provenance,"overlap":overlap}
    with marker.open("x",encoding="utf-8") as f: json.dump(payload,f,indent=2)
    invoke(package,"--data-root",audit_root,"--manifest",manifest,"--count",overlap["audit_files"],"--output",final/"independent_audit")
    payload["status"]="COMPLETE";write(marker,payload)
    print("Independent audit recorded; no further training/tuning",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("select");a.add_argument("--data-root",type=Path,required=True);a.add_argument("--out",type=Path,required=True)
    a.add_argument("--top",type=int,default=3,choices=range(1,6));a.add_argument("--gate-epochs",type=int,default=12,choices=range(1,51))
    a.add_argument("--mp-root",type=Path);a.add_argument("--cmgan-root",type=Path);a.add_argument("--allow-public-download",action="store_true")
    b=sub.add_parser("audit");b.add_argument("--final",type=Path,required=True);b.add_argument("--data-root",type=Path,required=True)
    b.add_argument("--audit-root",type=Path,required=True);b.add_argument("--manifest",type=Path,required=True)
    b.add_argument("--confirm-heldout",action="store_true");b.add_argument("--provenance",required=True)
    args=p.parse_args()
    core.no_uncertain_prior_jobs()
    import os
    lock=core.OUTPUT/"active.lock";lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("x",encoding="utf-8") as f: f.write(str(os.getpid()))
    try:
        if args.command=="select": finalize(args)
        else: audit(args)
    finally: lock.unlink(missing_ok=True)
if __name__=="__main__": main()
