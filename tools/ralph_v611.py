"""Unified V6.11 Ralph loop: diagnose -> choose -> smoke -> continue -> verify.
No new trainer topology/evaluator. Uses V6.10 phase/noise objectives and V6 critic/cascade.
Default dry plan. Run only after existing V6 and V6.9 processes exit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import ralph_v6 as ctl
import ralph_v6_refine as refine
import ralph_v610 as focus
import v6_refine_analysis as research
import ralph_v2 as core
import ralph_v4 as v4

ROOT=research.ROOT/"unified"
RECIPES=("continue","phase_derivatives","hard_noise_first","full_length","critic","residual")
MIN_GAIN=.002

def champion(entries):
    pool=[e for e in entries.values() if e[1] and not e[1]["smoke"]]
    if not pool:raise ValueError("No verified complete checkpoint")
    # Prefer an earlier checkpoint on exact ties, rather than repeatedly adopting fallback weights.
    return sorted(pool,key=lambda e:(-e[1]["best_validation_pesq_wb"],e[1]["v6"]["attempted_epochs"],e[0]["ref"]))[0]

def tried(entries,parent):
    """Count terminal attempts including failed jobs; identify exact parent and hypothesis."""
    h=parent[1]["v6"]["checkpoint_sha256"]
    records={}
    for st,r,c,folder in entries.values():
        if st.get("status") not in ("COMPLETE","ERROR_TERMINAL"):continue
        if c is None:
            try:c=ctl.config_of(folder)
            except (OSError,ValueError,KeyError,SyntaxError):continue
        if c.get("smoke") or not c.get("v6_parent") or c["v6_parent"]["sha256"]!=h:continue
        recipe=c.get("v611_recipe") or c.get("v610_recipe") or c.get("v6_refine_hypothesis")
        if recipe:
            key=(recipe,int(c.get("v611_lr_step",0)))
            records[key]=records.get(key,0)+1
    return records

def ranked_recipes(parent,diag,entries,max_epochs):
    st,r,c,_=parent
    if r["v6"]["attempted_epochs"]>=max_epochs:return []
    attempted=tried(entries,parent)
    phase=float(diag["phase_gain"]);mag=float(diag["magnitude_gain"])
    if not math.isfinite(phase) or not math.isfinite(mag):raise ValueError("Invalid Train diagnosis")
    order=[]
    # First exploit a verified improvement. After a no-gain block, explore orthogonal hypotheses.
    if r["v6"]["attempted_epochs"]==0 or (r["v6"].get("gain",0.)>=MIN_GAIN and r["best_epoch"]>0):order.append("continue")
    order+=["phase_derivatives","hard_noise_first"] if phase>=mag-.1 else ["hard_noise_first","phase_derivatives"]
    order+=["full_length","critic","residual","continue"]
    out=[]
    for recipe in dict.fromkeys(order):
        if recipe=="phase_derivatives" and c.get("v610_phase_derivatives"):continue
        if recipe=="hard_noise_first" and c.get("v610_hard_first"):continue
        if recipe=="full_length" and c.get("v6_crop")==48000:continue
        if recipe=="critic" and c.get("v6_critic"):continue
        if recipe=="residual" and c.get("v6_depth"):continue
        for lr_step in range(2 if recipe=="continue" else 1):
            if (recipe,lr_step) not in attempted:
                out.append((recipe,lr_step))
                break
    return out

def build(parent,recipe,lr_step,sha,cache,epochs,mix,public_mix,smoke=False):
    st,r,old,_=parent
    if recipe not in RECIPES:raise ValueError("Unknown unified recipe")
    if recipe in focus.RECIPES:
        c=focus.trial(parent,recipe,sha,cache,epochs,mix,public_mix,smoke)
    else:
        depth=old.get("v6_depth")
        critic=bool(old.get("v6_critic") or recipe=="critic")
        lr=float(old["lr"])
        if r["v6"]["attempted_epochs"]>0:
            # Continue productive training at the actual final LR; lower on second attempt.
            lr=min(lr,float(r["v6"]["lr"]))
        c=ctl.config(old["architecture"],"cascade" if depth else "train",sha,{},
            cache=None if smoke else cache,epochs=1 if smoke else epochs,mix=16 if smoke else mix,
            public_mix=16 if smoke else public_mix,korean=0 if smoke else old["v6_korean"],
            critic=critic,hard=bool(old.get("v6_hard")),parent=None if smoke else ctl.parent_info(st,r),
            base=old.get("v6_base_family"),depth=depth,smoke=smoke,lr=lr)
        c["v6_crop"]=48000 if critic else old["v6_crop"]
        c["v6_phase"]=old.get("v6_phase",True);c["v6_external_margin"]=old.get("v6_external_margin",.15)
        if old.get("v6_sisdr"):c["v6_sisdr"]=True
        c["v610_phase_derivatives"]=bool(old.get("v610_phase_derivatives"))
        c["v610_hard_first"]=bool(old.get("v610_hard_first"))
    c["lr"]=max(1e-8,float(c["lr"])*(.5**lr_step))
    c["v611_recipe"]=recipe;c["v611_lr_step"]=lr_step
    c["v610_recipe"]=recipe # preflight provenance; no model code depends on this string
    if c["v6_critic"]:
        c["v6_crop"]=48000
        # Existing gate cannot enable metric gradients before epoch 3.
        # Smoke tests plumbing only; full trial must allow at least 3 epochs.
        if not smoke and epochs<3:raise ValueError("Critic requires >=3 epochs for possible metric gradient")
    return refine.resign(c,sha)

def estimate(cfg,entries):
    n=refine.cost(cfg,entries)
    if cfg.get("v6_critic"):n*=1.35
    if cfg.get("v6_depth"):n*=1.15
    if cfg.get("v610_hard_first") and not cfg["smoke"]:n+=.12
    return n

def profiles(old,allow_reduce):
    a,b=int(old["v6_mix"]),int(old["v6_public_mix"])
    vals=[(a,b)]
    if allow_reduce:
        vals.extend((min(a,x),min(b,y)) for x,y in ((1024,256),(512,128),(256,64)))
    return list(dict.fromkeys(vals))

def pick(parent,diag,entries,sha,cache,remaining,max_job,max_epochs,chunk,allow_reduce):
    old=parent[2];left=max_epochs-parent[1]["v6"]["attempted_epochs"]
    for recipe,lr_step in ranked_recipes(parent,diag,entries,max_epochs):
        critic=recipe=="critic" or old.get("v6_critic")
        minimum=3 if critic else 1
        if left<minimum:continue
        # Longer productive blocks when affordable, shorter blocks for uncertain hypotheses.
        upper=min(left,max(chunk,minimum) if recipe=="continue" else minimum)
        for n in range(upper,minimum-1,-1):
            for mix,pub in profiles(old,allow_reduce):
                c=build(parent,recipe,lr_step,sha,cache,n,mix,pub)
                s=build(parent,recipe,lr_step,sha,cache,n,mix,pub,True)
                if c["v56_key"] in entries:continue
                prev=entries.get(s["v56_key"])
                if prev and not prev[1]:continue
                t=estimate(c,entries);sm=0. if prev else estimate(s,entries)
                if t+sm<=remaining and max(t,sm)<=max_job:
                    return {"config":c,"smoke":s,"reuse_smoke":bool(prev),"hours":t+sm,
                            "compound_data_change":mix!=old["v6_mix"] or pub!=old["v6_public_mix"]}
    return None

def train_diagnostic(data,parent,allow_download):
    """Fast per-champion Train-only diagnostic; never run 164-file CPU validation every round."""
    import torch
    import v6_finalize as fin
    import v6_offline as off
    import v57_offline as legacy
    from v6_preflight import ASSETS
    st,r,cfg,folder=parent
    # Existing full report can provide exact champion diagnosis.
    for p in sorted(research.ROOT.glob("*/report.json"),reverse=True):
        try:
            report=refine.load_report(p.parent,data)
            if report.get("snapshot_champion")==ctl.parent_info(st,r) and report.get("train_oracle_diagnostic"):
                return report["train_oracle_diagnostic"]
        except (OSError,ValueError,KeyError,RuntimeError):continue
    train=data/"train/samples.csv"
    ids=fin.train_splits(data)[1];rows=legacy.read_manifest(train,1296)
    x,s,meta=research.load_pairs(data,rows,ids)
    d=ROOT/"diagnostics"/(r["v6"]["checkpoint_sha256"]+"_"+ctl.sha(train)[:16])
    report=d/"diagnostic.json"
    identity={"parent":ctl.parent_info(st,r),"train_manifest":ctl.sha(train),"source":ctl.sha(folder/"source.py"),
              "torch":str(torch.__version__),"ids":ids}
    if report.exists():
        saved=json.loads(report.read_text(encoding="utf-8"))
        if saved.get("identity")==identity:return saved["diagnostic"]
    # Unique scratch directory; no modification of prior packages or datasets.
    tmp=d/("probe_"+research.stamp());tmp.mkdir(parents=True)
    torch.set_num_threads(2)
    spec=fin.package_expert(tmp,0,(r["best_validation_pesq_wb"],"v6",r["model_family"],folder),ASSETS,None,ctl.sha(train),allow_download)
    model=off.load_expert(tmp,spec,0)
    y=research.cached_predictions(research.ROOT/"predictions",model,x,
        {"ref":st["ref"],"checkpoint":r["v6"]["checkpoint_sha256"],"source":spec["source_sha256"],"torch":str(torch.__version__),"threads":2,"split":"select"})
    diag=research.oracle_diagnostic(s,y)
    # Store only aggregate diagnostics, no raw clips outside the ignored cache.
    core.json_write(report,{"identity":identity,"diagnostic":diag})
    shutil.rmtree(tmp)
    print("V611 DIAGNOSTIC",json.dumps(diag),flush=True)
    return diag

def run(a):
    data=a.data_root.resolve()
    code=focus.source();sha=focus.digest(code)
    if (core.OUTPUT/"v57_audit_seal.json").exists():raise RuntimeError("Independent audit sealed")
    if not a.execute:
        e=ctl.ledger();p=champion(e)
        print("V611 DRY PLAN",json.dumps({"parent":ctl.parent_info(p[0],p[1]),"recipes":RECIPES,"max_rounds":a.max_rounds,
              "source":sha,"note":"No GPU; actual Train diagnosis and live quota used in execute mode."}),flush=True)
        return
    if not a.allow_public_assets:raise ValueError("Explicit public permission required")
    kaggle=shutil.which("kaggle")
    if not kaggle:raise RuntimeError("Kaggle CLI missing")
    # Both old local processes must have exited before starting. The lock protects
    # against active submissions; cannot discover a separate process waiting outside it.
    refine.wait_idle(3)
    lock=core.OUTPUT/"active.lock";lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("x",encoding="utf-8") as f:f.write(str(os.getpid()))
    try:
        if a.resume:ctl.resume(kaggle,a.max_job_minutes)
        else:core.no_uncertain_prior_jobs()
        from v6_preflight import require
        require(focus.digest(ctl.source()),data,["mp_dns"])
        ROOT.mkdir(parents=True,exist_ok=True)
        start=time.monotonic();initial=ctl.quota_retry(kaggle)
        budget=min(a.budget_hours if a.budget_hours is not None else initial-a.reserve_hours,initial-a.reserve_hours)
        def remaining():return min(budget-(time.monotonic()-start)/3600,ctl.quota_retry(kaggle)-a.reserve_hours)
        tag="v4"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        state_path=ROOT/("run_"+tag+".json")
        rounds=jobs=failures=stale=0;events=[];reason="round_limit"
        def save():
            core.json_write(state_path,{"protocol":"v611-loop-v1","rounds":rounds,"jobs":jobs,"events":events,"stop_reason":reason,"budget_hours":budget})
        while not a.max_rounds or rounds<a.max_rounds:
            core.no_uncertain_prior_jobs()
            if (core.OUTPUT/"v57_audit_seal.json").exists():raise RuntimeError("Independent audit sealed")
            if remaining()<=0:reason="budget";break
            entries=ctl.ledger();p=champion(entries)
            if (p[2].get("v6_base_family") or p[2]["architecture"])!="mp_dns":reason="non_mp_dns_champion";break
            if p[1]["v56_source_sha"] not in (focus.digest(ctl.source()),sha):raise ValueError("Unknown parent source")
            if p[1]["v6"]["attempted_epochs"]>=a.max_epochs:reason="lineage_epoch_cap";break
            diag=train_diagnostic(data,p,a.allow_public_assets)
            cache=p[2].get("public_cache") or v4.find_public_cache()
            plan=pick(p,diag,entries,sha,cache,remaining(),a.max_job_minutes/60,a.max_epochs,a.chunk_epochs,a.allow_reduce_mix)
            if not plan:reason="no_affordable_untried_recipe";break
            c,s=plan["config"],plan["smoke"]
            print("V611 PLAN",json.dumps({"round":rounds+1,"remaining":remaining(),**plan}),flush=True)
            if plan["hours"]>remaining():reason="budget_before_preflight";break
            focus.preflight(data,code,c,ROOT/("preflight_"+research.stamp()),p)
            if plan["hours"]>remaining():reason="budget_after_preflight";break
            rounds+=1
            event={"round":rounds,"recipe":c["v611_recipe"],"parent":c["v6_parent"],"plan_hours":plan["hours"],
                   "compound_data_change":plan["compound_data_change"],"status":"started"}
            events.append(event);save()
            try:
                if not plan["reuse_smoke"]:
                    jobs+=1;save();ctl.submit(kaggle,code,tag,jobs,s,a.max_job_minutes)
                core.no_uncertain_prior_jobs()
                if estimate(c,ctl.ledger())>min(remaining(),a.max_job_minutes/60):
                    reason="budget_after_smoke";event["status"]="smoke_only";break
                jobs+=1;save()
                st,r=ctl.submit(kaggle,code,tag,jobs,c,a.max_job_minutes)
                gain=float(r["best_validation_pesq_wb"])-float(p[1]["best_validation_pesq_wb"])
                event.update(status="complete",ref=st["ref"],pesq=r["best_validation_pesq_wb"],gain=gain,
                    metric_steps=r["v6"]["metric_steps"],critic_calibration=r["v6"]["critic_calibration"])
                print("V611 SCORE",json.dumps(event),flush=True)
                failures=0;stale=0 if gain>=MIN_GAIN else stale+1
            except ctl.legacy.KaggleTerminalFailure as exc:
                failures+=1;stale+=1;event.update(status="terminal_failure",error=str(exc))
                if failures>=3:reason="failure_budget";save();break
            save()
            if a.max_stale and stale>=a.max_stale:reason="stagnation";break
        save();print("V611 STOP",reason,flush=True)
        core.no_uncertain_prior_jobs()
        # Once at the end, evaluate/compare actual PCM16 checkpoint mixtures and seal local winner.
        if not a.skip_final:
            report=research.analyze(data,research.ROOT/("unified_final_"+research.stamp()),a.top,2,8,a.allow_public_assets,prefer_diverse=False)
            print("V611 FINAL",json.dumps({"winner":report["winner"],"pesq":report["winner_pesq"]}),flush=True)
    finally:lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True);p.add_argument("--execute",action="store_true")
    p.add_argument("--resume",action="store_true");p.add_argument("--allow-public-assets",action="store_true")
    p.add_argument("--max-rounds",type=int,default=0);p.add_argument("--max-stale",type=int,default=0)
    p.add_argument("--max-epochs",type=int,default=200);p.add_argument("--chunk-epochs",type=int,default=3)
    p.add_argument("--budget-hours",type=float);p.add_argument("--reserve-hours",type=float,default=2.)
    p.add_argument("--max-job-minutes",type=int,default=360);p.add_argument("--allow-reduce-mix",action="store_true")
    p.add_argument("--top",type=int,choices=range(1,6),default=2);p.add_argument("--skip-final",action="store_true")
    a=p.parse_args()
    if not (a.max_rounds>=0 and a.max_stale>=0 and 1<=a.max_epochs<=200 and 1<=a.chunk_epochs<=10 and
            math.isfinite(a.reserve_hours) and a.reserve_hours>=2 and 15<=a.max_job_minutes<=360 and
            (a.budget_hours is None or math.isfinite(a.budget_hours) and 0<a.budget_hours<=30)):p.error("Invalid budget/limits")
    run(a)
if __name__=="__main__":main()
