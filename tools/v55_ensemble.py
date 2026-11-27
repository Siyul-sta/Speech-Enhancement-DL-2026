"""V5.5 local ensemble search. Train-only fit/selection, frozen noisy-only inference."""
from __future__ import annotations
import argparse
import csv
import io
import itertools
import json
import math
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import ralph_v4 as controller
import submit_v5 as submission
from v51_data_audit import speaker
from v55_gating import fit_gate,mix_outputs

def select_experts(records,limit=3,public_families=()):
    """Shortlist verified native runs, diversify by architecture first.
    Prior validation scores are development shortlist evidence, not gate targets.
    """
    if not 2<=limit<=5: raise ValueError("Expert limit 2–5")
    pool=[]
    for st,r in records:
        if st.get("kind")!="v4_external" or st.get("status")!="COMPLETE" or r.get("evaluation_protocol")!="v4.2-fixed-dev-v2" or r.get("smoke") or r.get("transductive_noisy") or r.get("model_family") not in set(submission.FAMILIES)|set(public_families): continue
        if not st.get("source_path"): continue
        folder=Path(st["source_path"]).parent
        if not all((folder/f).is_file() for f in ("source.py","best_model.pt","result.json")): continue
        if r.get("model_family") in public_families and not r["model_family"].startswith(("cmgan","cascade_cmgan")) and not all((folder/f).is_file() for f in ("offline_hyperparams.yaml","offline_hyperparams.json")): continue
        score=float(r["best_validation_pesq_wb"])
        if not math.isfinite(score): continue
        pool.append((score,r["model_family"],folder))
    pool.sort(key=lambda p:(-p[0],str(p[2])))
    chosen=[];seen_families=set();seen_hashes=set()
    for diverse in (True,False):
        for _,family,folder in pool:
            if folder in chosen or (diverse and family in seen_families): continue
            digest=submission.sha256(folder/"best_model.pt")
            if digest in seen_hashes: continue
            chosen.append(folder);seen_families.add(family);seen_hashes.add(digest)
            if len(chosen)==limit: return chosen
    return chosen

def split_train(rows,fit_count=64,select_count=32,seed=55042):
    """Distinct known speaker groups between fit and selection, exact clip counts."""
    if min(fit_count,select_count)<1: raise ValueError("Positive split sizes required")
    groups={}
    for i,row in enumerate(rows):
        name=speaker(row)
        if name=="unknown": continue
        groups.setdefault(name,[]).append(i)
    rng=np.random.default_rng(seed)
    names=sorted(groups)
    rng.shuffle(names)
    select=[];fit=[];select_names=[];fit_names=[]
    for name in names:
        if len(select)<select_count:
            select.extend(groups[name]);select_names.append(name)
        else:
            fit.extend(groups[name]);fit_names.append(name)
    if len(fit)<fit_count or len(select)<select_count: raise ValueError("Insufficient known disjoint train speakers for ensemble fit/selection")
    fit=sorted(rng.choice(fit,fit_count,replace=False).tolist())
    select=sorted(rng.choice(select,select_count,replace=False).tolist())
    return fit,select

def load_train(root):
    path=root/"train"/"samples.csv"
    with path.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    if len(rows)!=1296 or len({r["segment_id"] for r in rows})!=1296: raise ValueError("Expected original 1296 unique train rows")
    return path,rows

@torch.no_grad()
def collect(root,rows,ids,models,device):
    x=[];c=[];pred=[]
    for i in ids:
        row=rows[i]
        a=torch.from_numpy(submission.read_wav(submission.safe_path(root,row["noisy_path"])))[None].to(device)
        target=torch.from_numpy(submission.read_wav(submission.safe_path(root,row["clean_path"])))
        outputs=[m(a).float() for _,m in models]
        if any(y.shape!=a.shape or not torch.isfinite(y).all() for y in outputs): raise RuntimeError("Invalid expert output")
        x.append(a[0].cpu());c.append(target);pred.append(torch.stack([y[0].cpu() for y in outputs]))
    return torch.stack(x),torch.stack(pred),torch.stack(c)

def fixed_weights(m):
    # Singles, all uniform subsets, and a bounded pairwise 25/75 grid.
    for n in range(1,m+1):
        for subset in itertools.combinations(range(m),n):
            yield tuple(1/n if i in subset else 0. for i in range(m))
    for i,j in itertools.combinations(range(m),2):
        for a in (.25,.75):
            yield tuple(a if k==i else 1-a if k==j else 0. for k in range(m))

def pcm16(y):
    buf=io.BytesIO()
    sf.write(buf,np.clip(y,-1,1),16000,format="WAV",subtype="PCM_16")
    buf.seek(0)
    return sf.read(buf,dtype="float32")[0]

def actual_scores(clean,pred,pesq_fn):
    a,b=clean.detach().cpu().numpy(),pred.detach().cpu().numpy()
    if a.shape!=b.shape or not np.isfinite(b).all(): raise ValueError("Invalid ensemble score input")
    vals=[float(pesq_fn(16000,c,pcm16(y),"wb")) for c,y in zip(a,b)]
    if not vals or not np.isfinite(vals).all(): raise ValueError("Invalid PESQ scores")
    return vals

def choose_candidate(candidates,margin=.02):
    singles=[x for x in candidates if x["type"]=="fixed" and sum(w>0 for w in x["weights"])==1]
    if not singles: raise ValueError("No single-expert fallback")
    baseline=max(singles,key=lambda r:r["pesq"])
    best=max(candidates,key=lambda r:r["pesq"])
    return best if best["pesq"]>baseline["pesq"]+margin else baseline

def bundle_experts(folders,out,public_families=(),cmgan_root=None):
    entries=[];hashes=set()
    for i,folder in enumerate(folders):
        r=controller.verify(folder,"v4_external")
        state=json.loads((folder/"remote_state.json").read_text(encoding="utf-8"))
        controller.verify_v56_metadata(r,state)
        if state.get("status")!="COMPLETE" or state.get("kind")!="v4_external" or r.get("evaluation_protocol")!="v4.2-fixed-dev-v2" or r.get("smoke") or r.get("transductive_noisy") or r.get("model_family") not in set(submission.FAMILIES)|set(public_families):
            raise ValueError("Ineligible expert: "+str(folder))
        digest=submission.sha256(folder/"best_model.pt")
        if digest in hashes: raise ValueError("Duplicate expert checkpoint")
        hashes.add(digest)
        sub=out/f"model_{i+1}";sub.mkdir()
        for name in ("source.py","best_model.pt","result.json"): shutil.copy2(folder/name,sub/name)
        entries.append({"weight":1/len(folders),"source":f"model_{i+1}/source.py","checkpoint":f"model_{i+1}/best_model.pt",
            "source_sha256":submission.sha256(sub/"source.py"),"checkpoint_sha256":digest,
            "individual_validation_pesq":r["best_validation_pesq_wb"],"family":r["model_family"],"remote_ref":state.get("ref")})
        if r["model_family"] in public_families:
            from v57_public_assets import package_public
            meta=package_public(r["model_family"],folder,(sub/"source.py").read_text(encoding="utf-8"),sub/"assets",False,cmgan_root)
            entries[-1]["public_assets"]={**meta,"root":f"model_{i+1}/assets"}
    return entries

def json_write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")

def run(args):
    from pesq import pesq
    root=args.root.resolve();out=args.out.resolve()
    if out.exists() or out.is_relative_to(root): raise ValueError("Use a new output directory outside dataset root")
    if not 2<=args.max_experts<=5 or not 1<=args.gate_epochs<=100: raise ValueError("Invalid ensemble settings")
    public_families=tuple(getattr(args,"public_families",()))
    cmgan_root=getattr(args,"cmgan_root",None)
    folders=[p.resolve() for p in args.run] if args.run else select_experts(controller.history(),args.max_experts,public_families)
    if not 2<=len(folders)<=5 or len(set(folders))!=len(folders): raise ValueError("Need 2–5 distinct verified completed native checkpoints; train experts first")
    if any(out.is_relative_to(p.resolve()) for p in folders): raise ValueError("Output cannot modify expert runs")
    manifest,rows=load_train(root)
    fit,select=split_train(rows)
    device=torch.device(args.device)
    out.mkdir(parents=True)
    entries=bundle_experts(folders,out,public_families,cmgan_root)
    config=out/"bundle.json"
    working=out/"candidate_bundle.json"
    json_write(working,{"models":entries})
    models=submission.load_bundle(working,device)
    for _,m in models: m.eval().requires_grad_(False)
    x,p,c=collect(root,rows,fit,models,device)
    sx,sp,sc=collect(root,rows,select,models,device)
    candidates=[];gates={}
    for w in fixed_weights(len(models)):
        y=(sp*torch.tensor(w,dtype=sp.dtype)[None,:,None]).sum(1)
        values=actual_scores(sc,y,pesq)
        candidates.append({"type":"fixed","weights":list(w),"pesq":float(np.mean(values))})
    for mode in ("clip","frequency"):
        gate=fit_gate(x,p,c,mode,epochs=args.gate_epochs)
        with torch.no_grad():
            y=torch.cat([mix_outputs(a,b,gate(a)) for a,b in zip(sx.split(4),sp.split(4))])
        values=actual_scores(sc,y,pesq)
        candidates.append({"type":mode,"pesq":float(np.mean(values))})
        gates[mode]=gate
    selected=choose_candidate(candidates)
    final={"selection":"train-only speaker-disjoint fit/selection; no validation targets used to fit or choose mixture"}
    if selected["type"]=="fixed":
        final["models"]=[{**entry,"weight":w} for entry,w in zip(entries,selected["weights"]) if w>0]
    else:
        mode=selected["type"]
        target=out/"gate.pt"
        torch.save(gates[mode].state_dict(),target)
        final["models"]=entries
        final["gating"]={"mode":mode,"experts":len(entries),"checkpoint":"gate.pt","checkpoint_sha256":submission.sha256(target)}
    json_write(config,final)
    # Strict reload before publishing a successful report; use ordinary inference contract.
    try:
        selected_models=submission.load_bundle(config,device)
        sample=sx[0].numpy()
        actual=submission.predict(selected_models,sample,device)
        if actual.shape!=sample.shape or not np.isfinite(actual).all(): raise RuntimeError("Invalid final bundle")
    except Exception:
        config.unlink(missing_ok=True)
        raise
    working.unlink()
    report={"protocol":"v55-train-fit-select-v1","train_manifest_sha256":submission.sha256(manifest),
        "fit_indices":fit,"selection_indices":select,"fit_speaker_groups":sorted({speaker(rows[i]) for i in fit}),
        "selection_speaker_groups":sorted({speaker(rows[i]) for i in select}),"gate_epochs":args.gate_epochs,
        "experts":entries,"candidates":candidates,"selected":selected,"fallback_margin":.02,
        "bundle_sha256":submission.sha256(config),
        "limitations":["Prior challenge validation scores may shortlist experts; no validation target is used to fit or choose mixture.",
          "Base experts may have trained on fit/selection clips. Scores are not out-of-fold estimates or independent generalization.",
          "Public experts require explicit compatibility authorization and archived assets.",
          "Final validation is development evidence, not hidden test."]}
    json_write(out/"ensemble_report.json",report)
    print("V55 SELECTED",json.dumps(selected),flush=True)
    if args.validation:
        # Only selected, frozen recipe is evaluated; never reselect on this result.
        submission.evaluate(selected_models,root,device,out)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--run",type=Path,nargs="+",help="Explicit verified native run folders; default uses Ralph history")
    p.add_argument("--max-experts",type=int,default=3)
    p.add_argument("--gate-epochs",type=int,default=12)
    p.add_argument("--device",default="cpu")
    p.add_argument("--validation",action="store_true",help="Evaluate only selected frozen bundle on challenge development validation")
    args=p.parse_args()
    torch.set_num_threads(2)
    run(args)

if __name__=="__main__": main()
