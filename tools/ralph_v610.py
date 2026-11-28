"""MP-SENet DNS-focused continuation. Separate from active V6/V6.9 processes.
Default plans only. Execute ONLY after both existing processes have exited.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import ralph_v6 as ctl
import ralph_v2 as core
import ralph_v4 as v4
import ralph_v6_refine as refine
import v6_refine_analysis as research

RECIPES=("phase_derivatives","hard_noise_first","full_length","residual")
def replace_once(code,needle,value):
    if code.count(needle)!=1:raise ValueError("V6.10 source contract changed: "+needle[:70])
    return code.replace(needle,value,1)

def source():
    code=ctl.source()
    extra=(ctl.ROOT/"src/v610_focus.py").read_text(encoding="utf-8")
    code=replace_once(code,"def v6_loss(pred,clean,cfg):",extra+"\n\ndef v6_loss(pred,clean,cfg):")
    needle='    if cfg.get("v6_phase",True): loss=loss+.15*phase_complex_loss(pred.float(),clean.float())'
    code=replace_once(code,needle,needle+'\n    if cfg.get("v610_phase_derivatives"): loss=loss+.08*v610_phase_derivatives(pred.float(),clean.float())')
    needle='        for epoch in range(1,int(cfg["epochs"])+1):'
    code=replace_once(code,needle,'''        if cfg.get("v610_hard_first") and mix is not None:
            # Training calibration only; seed hard group probabilities before epoch 1.
            _,cal_rows=evaluate(model,cal,device,pesq_fn,save_rows=True)
            mix.curriculum=False
            print("V610_HARD_INIT "+json.dumps(v610_noise_weights(mix,train,cal_ids,cal_rows)),flush=True)
'''+needle)
    compile(code,"v610_runner.py","exec")
    return code

def digest(code):return hashlib.sha256(code.encode()).hexdigest()

def select_recipe(parent,report,entries,forced=None):
    st,r,cfg,_=parent
    diag=report.get("train_oracle_diagnostic")
    if not diag or report.get("snapshot_champion")!=ctl.parent_info(st,r):
        raise ValueError("Need actual champion Train diagnostic")
    attempted=set()
    for s,q,c,_ in entries.values():
        if c and c.get("v610_recipe") and c.get("v6_parent") and c["v6_parent"]["sha256"]==r["v6"]["checkpoint_sha256"]:
            attempted.add(c["v610_recipe"])
    phase=float(diag["phase_gain"]);mag=float(diag["magnitude_gain"])
    if not math.isfinite(phase) or not math.isfinite(mag):raise ValueError("Invalid oracle diagnostic")
    order=["phase_derivatives","hard_noise_first"] if phase>=mag-.1 else ["hard_noise_first","phase_derivatives"]
    order+=["full_length","residual"]
    if forced:order=[forced]
    for name in order:
        if name in attempted:continue
        if name=="phase_derivatives" and cfg.get("v610_phase_derivatives"):continue
        if name=="hard_noise_first" and cfg.get("v610_hard_first"):continue
        if name=="full_length" and cfg.get("v6_crop")==48000:continue
        if name=="residual" and cfg.get("v6_depth"):continue
        return name
    return None

def trial(parent,recipe,sha,cache,epochs,mix,public_mix,smoke=False):
    st,r,old,_=parent
    if (old.get("v6_base_family") or old["architecture"])!="mp_dns":raise ValueError("MP-SENet DNS only")
    if recipe not in RECIPES:raise ValueError("Unknown recipe")
    depth=old.get("v6_depth")
    new_residual=recipe=="residual" and not depth
    if recipe=="residual" and not new_residual:raise ValueError("Residual already exists")
    family="v6_cascade" if new_residual else old["architecture"]
    depth=2 if new_residual else depth
    lr=float(old["lr"])
    if r["v6"]["attempted_epochs"]>0:lr=max(1e-8,min(lr*.5,float(r["v6"]["lr"])))
    if new_residual:lr=1e-4
    c=ctl.config(family,"cascade" if depth else "train",sha,{},cache=None if smoke else cache,
        epochs=1 if smoke else epochs,mix=16 if smoke else mix,public_mix=16 if smoke else public_mix,
        korean=0 if smoke else old["v6_korean"],critic=bool(old.get("v6_critic")),hard=bool(old.get("v6_hard") or recipe=="hard_noise_first"),
        parent=None if smoke else ctl.parent_info(st,r),base="mp_dns" if depth else None,depth=depth,
        coarse=bool(new_residual and not smoke),smoke=smoke,lr=lr)
    c["v6_crop"]=48000 if recipe=="full_length" or old.get("v6_critic") else old["v6_crop"]
    c["v6_phase"]=old.get("v6_phase",True)
    c["v6_external_margin"]=old.get("v6_external_margin",.15)
    if old.get("v6_sisdr"):c["v6_sisdr"]=True
    c["v610_recipe"]=recipe
    c["v610_phase_derivatives"]=bool(old.get("v610_phase_derivatives") or recipe=="phase_derivatives")
    c["v610_hard_first"]=bool(old.get("v610_hard_first") or recipe=="hard_noise_first")
    return refine.resign(c,sha)

def preflight(data,code,cfg,out,parent):
    """Check new objective on actual train, strict archived weights, full 48k forward.
    Does not touch original V6 preflight report or source.
    """
    import torch
    from v6_preflight import require,load_module,environment
    import submit_v5 as wav
    import v57_offline as off
    base=ctl.source()
    accepted=require(digest(base),data,["mp_dns"])
    out.mkdir(parents=True,exist_ok=False)
    runner=out/"source.py";runner.write_bytes(code.encode("utf-8"));m=load_module(runner)
    m.WORK=out/"assets";m.WORK.mkdir()
    torch.set_num_threads(2)
    rows=off.read_manifest(data/"train/samples.csv",1296)
    row=rows[0];x=torch.from_numpy(wav.read_wav(wav.safe_path(data,row["noisy_path"])))[None]
    s=torch.from_numpy(wav.read_wav(wav.safe_path(data,row["clean_path"])))[None]
    m.EXPERIMENT_CONFIG=cfg
    model=m.v6_make_model(cfg,mp_root=Path(accepted["assets"]),pretrained=False)
    target=model.coarse if cfg["v6_parent_coarse"] else model
    cp=parent[3]/"best_model.pt"
    if ctl.sha(cp)!=parent[1]["v6"]["checkpoint_sha256"]:raise ValueError("Parent weights changed")
    target.load_state_dict(torch.load(cp,map_location="cpu",weights_only=True),strict=True)
    model.eval()
    with torch.no_grad():y=model(x)
    if y.shape!=x.shape or y.shape!=(1,48000) or not torch.isfinite(y).all():raise ValueError("Full noisy-only output failed")
    if cfg.get("v6_parent_coarse"):
        with torch.no_grad():z=model.coarse(x)
        if not torch.allclose(y,z,rtol=2e-4,atol=2e-5):raise ValueError("Cascade identity failed")
    model.train()
    if cfg.get("v6_transfer"):m.TransferSchedule(model,cfg["v6_transfer"],float(cfg["lr"]))
    loss=m.v6_loss(model(x[:,:4096]),s[:,:4096],cfg)
    if not torch.isfinite(loss):raise ValueError("Invalid new loss")
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    if not grads or any(not torch.isfinite(g).all() for g in grads) or not any(float(g.abs().max())>0 for g in grads):
        raise ValueError("New objective gradient failed")
    result={"status":"PASS","source_sha256":digest(code),"parent_sha256":ctl.sha(cp),"recipe":cfg["v610_recipe"],
        "environment":environment(),"loss":float(loss.detach()),"samples":48000}
    core.json_write(out/"report.json",result)
    print("V610 PREFLIGHT PASS",json.dumps(result),flush=True)
    return result

def run(a):
    data=a.data_root.resolve()
    if (core.OUTPUT/"v57_audit_seal.json").exists():raise RuntimeError("Independent audit sealed")
    # A planning run can coexist with current jobs. An execute run must start after both
    # existing local processes have exited, and still obey the exclusive lock.
    code=source();sha=digest(code)
    if a.execute:
        if not a.allow_public_assets:raise ValueError("Explicit public asset authorization required")
        refine.wait_idle(3)
    lock=core.OUTPUT/"active.lock"
    def body():
        entries=ctl.ledger()
        done=[e for e in entries.values() if e[1] and not e[1]["smoke"]]
        if not done:raise ValueError("No complete checkpoint")
        parent=sorted(done,key=lambda e:(-e[1]["best_validation_pesq_wb"],e[0]["ref"]))[0]
        if (parent[2].get("v6_base_family") or parent[2]["architecture"])!="mp_dns":raise ValueError("New overall champion is not MP-DNS; reassess rather than discard it")
        if parent[1]["v56_source_sha"] not in (digest(ctl.source()),sha):raise ValueError("Unrecognized parent training source")
        # Use cached Train predictions when current champion matches; otherwise regenerate.
        report=None
        if a.analysis:
            report=refine.load_report(a.analysis.resolve(),data)
            if report.get("snapshot_champion")!=ctl.parent_info(parent[0],parent[1]):report=None
        if report is None:
            candidates=sorted(research.ROOT.glob("*/report.json"),reverse=True)
            for p in candidates:
                try:
                    candidate=refine.load_report(p.parent,data)
                    if candidate.get("snapshot_champion")==ctl.parent_info(parent[0],parent[1]) and candidate.get("train_oracle_diagnostic"):
                        report=candidate;break
                except (OSError,ValueError,KeyError):continue
        if report is None:
            report=research.analyze(data,research.ROOT/("focus_"+research.stamp()),1,2,8,a.allow_public_assets)
        recipe=select_recipe(parent,report,entries,a.recipe)
        if recipe is None:print("V610 STOP no untried focused hypothesis",flush=True);return
        old=parent[2]
        mix=min(int(old["v6_mix"]),a.mix) if a.mix else int(old["v6_mix"])
        public=min(int(old["v6_public_mix"]),a.public_mix) if a.public_mix else int(old["v6_public_mix"])
        # Smaller budget is a compound recipe change; do not claim isolated causal gains.
        cache=old.get("public_cache") or v4.find_public_cache() if a.execute else old.get("public_cache")
        cfg=trial(parent,recipe,sha,cache,a.epochs,mix,public)
        sm=trial(parent,recipe,sha,cache,a.epochs,mix,public,True)
        estimate=refine.cost(cfg,entries);smoke_est=refine.cost(sm,entries)
        print("V610 PLAN",json.dumps({"recipe":recipe,"parent":cfg["v6_parent"],"config":cfg,
             "estimated_hours":estimate+smoke_est,"smoke_hours":smoke_est,"budget_hours":a.budget_hours,
             "pair_count_changed":mix!=old["v6_mix"] or public!=old["v6_public_mix"]}),flush=True)
        if not a.execute:return
        core.no_uncertain_prior_jobs()
        if cfg["v56_key"] in entries:print("V610 STOP already attempted",flush=True);return
        kaggle=shutil.which("kaggle")
        if not kaggle:raise RuntimeError("Kaggle CLI missing")
        start=time.monotonic()
        def remaining():return min(a.budget_hours-(time.monotonic()-start)/3600,ctl.quota_retry(kaggle)-a.reserve_hours)
        prev=entries.get(sm["v56_key"])
        if prev and not prev[1]:print("V610 STOP prior smoke failed",flush=True);return
        if estimate+(0 if prev else smoke_est)>remaining() or max(estimate,smoke_est)>a.max_job_minutes/60:
            print("V610 STOP insufficient GPU budget",flush=True);return
        preflight(data,code,cfg,research.ROOT/("v610_preflight_"+research.stamp()),parent)
        if estimate+(0 if prev else smoke_est)>remaining():print("V610 STOP after CPU acceptance, insufficient budget",flush=True);return
        tag="v4"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        step=0
        if not prev:
            step+=1;ctl.submit(kaggle,code,tag,step,sm,a.max_job_minutes)
        core.no_uncertain_prior_jobs()
        if refine.cost(cfg,ctl.ledger())>remaining():print("V610 STOP after smoke, insufficient GPU budget",flush=True);return
        step+=1;st,r=ctl.submit(kaggle,code,tag,step,cfg,a.max_job_minutes)
        core.json_write(research.ROOT/("v610_result_"+tag+".json"),{"recipe":recipe,"parent":cfg["v6_parent"],"state":st,"result":r})
        print("V610 RESULT",r["best_validation_pesq_wb"],flush=True)
        research.analyze(data,research.ROOT/("v610_post_"+research.stamp()),2,2,8,a.allow_public_assets)
    if not a.execute:body();return
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("x",encoding="utf-8") as f:f.write(str(os.getpid()))
    try:
        core.no_uncertain_prior_jobs()
        body()
    finally:lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True);p.add_argument("--analysis",type=Path)
    p.add_argument("--recipe",choices=RECIPES);p.add_argument("--execute",action="store_true")
    p.add_argument("--allow-public-assets",action="store_true");p.add_argument("--epochs",type=int,choices=(1,2,3),default=1)
    p.add_argument("--mix",type=int,default=0,help="Optional explicit cap, 0 preserves parent")
    p.add_argument("--public-mix",type=int,default=0,help="Optional explicit cap, 0 preserves parent")
    p.add_argument("--budget-hours",type=float,default=3.);p.add_argument("--reserve-hours",type=float,default=2.)
    p.add_argument("--max-job-minutes",type=int,default=360)
    a=p.parse_args()
    if not (0<=a.mix<=32768 and 0<=a.public_mix<=8192 and math.isfinite(a.budget_hours) and 0<a.budget_hours<=6 and math.isfinite(a.reserve_hours) and a.reserve_hours>=2 and 15<=a.max_job_minutes<=360):p.error("Invalid bounded budget")
    run(a)
if __name__=="__main__":main()
