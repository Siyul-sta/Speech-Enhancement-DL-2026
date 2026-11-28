"""Evidence-driven V6 continuation; never interrupts the active controller.
Existing V6 source and preflight identity remain unchanged.
Default is analysis/plan only; --execute-after-idle authorizes bounded GPU continuation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from datetime import datetime,timezone
from pathlib import Path
import ralph_v6 as ctl
import ralph_v2 as core
import ralph_v56 as v56
import ralph_v4 as v4
import v6_refine_analysis as research

def resign(cfg,code_sha):
    c={k:v for k,v in cfg.items() if k not in ("v56_key","v56_source_sha","v56_action")}
    action="smoke" if c["smoke"] else "v6_"+c["v6_mode"]
    key=v56.signature([code_sha,action,c])
    return {**c,"v56_key":key,"v56_source_sha":code_sha,"v56_action":action}

def build_trial(champion,report,code_sha,policies,locks,cache,epochs,smoke=False):
    st,r,old,_=champion
    hp=report["hypothesis"]["name"]
    if hp not in ("full_length","hard_noise"):raise ValueError("Unknown refinement")
    # One change to the scientific recipe at a time. Start from verified weights.
    # Smoke deliberately uses the official initial weights, not validation-informed targets.
    family=old["architecture"];base=old.get("v6_base_family");depth=old.get("v6_depth")
    if old.get("v6_critic"):
        # A calibrated critic is not restored across blocks. Preserve its flag but report restart.
        critic=True
    else:critic=False
    if hp=="full_length" and old.get("v6_crop")==48000:hp="hard_noise"
    if hp=="hard_noise" and old.get("v6_hard"):
        if old.get("v6_crop")!=48000:hp="full_length"
        else:raise ValueError("Both targeted hypotheses already present; no untested change")
    mode="cascade" if depth else "train"
    parent=None if smoke else ctl.parent_info(st,r)
    lr=float(old["lr"])
    # For zero-shot first learning uses original LR; continuations use <= half prior start/final.
    if r["v6"]["attempted_epochs"]>0:lr=max(1e-8,min(lr*.5,float(r["v6"]["lr"])))
    cfg=ctl.config(family,mode,code_sha,policies,cache=None if smoke else cache,
        epochs=2 if smoke else epochs,mix=16 if smoke else old["v6_mix"],
        public_mix=16 if smoke else old["v6_public_mix"],korean=0 if smoke else old["v6_korean"],
        critic=critic,hard=bool(old["v6_hard"] or hp=="hard_noise"),parent=parent,base=base,depth=depth,
        smoke=smoke,lr=lr,asset_locks=locks)
    cfg["v6_crop"]=48000 if hp=="full_length" else old["v6_crop"]
    cfg["v6_phase"]=old.get("v6_phase",True)
    cfg["v6_external_margin"]=old.get("v6_external_margin",.15)
    if old.get("v6_sisdr"):cfg["v6_sisdr"]=True
    cfg["v6_refine_hypothesis"]=hp
    return resign(cfg,code_sha)

def cost(cfg,entries):
    """Conservative full-length projection from observed epoch cost, never guaranteed billing."""
    guess=ctl.estimate(cfg,entries)
    if cfg["smoke"]:return max(guess,.6 if cfg["v6_crop"]==48000 else .35)
    rates=[]
    for st,r,c,_ in entries.values():
        if not r or r.get("smoke") or c["architecture"]!=cfg["architecture"] or c.get("v6_base_family")!=cfg.get("v6_base_family"):continue
        n=r["v6"].get("training_pairs_per_epoch",0)
        if not n:continue
        ratio=max(1.,cfg["v6_crop"]/c["v6_crop"])
        rates.extend(float(t)/n*ratio for t in r.get("v6_train_seconds",[]) if t>0)
    if rates:
        pairs=1296+cfg["v6_mix"]+cfg["v6_public_mix"]
        guess=max(guess,.6+1.4*(statistics.median(rates)*pairs*cfg["epochs"]+300)/3600)
    elif cfg["v6_crop"]==48000:guess=max(guess,1.5*guess)
    return guess

def wait_idle(minutes,quiet_seconds=90,clock=time.monotonic,sleep=time.sleep):
    # Original V6 briefly releases and reacquires its lock before offline finalization.
    # Require a quiet interval so this continuation cannot steal that handoff.
    lock=core.OUTPUT/"active.lock";deadline=clock()+minutes*60;idle_since=None
    while True:
        now=clock()
        if lock.exists():
            idle_since=None
            print("V6 REFINE WAIT existing controller (not interrupted)",flush=True)
        else:
            if idle_since is None:idle_since=now
            if now-idle_since>=quiet_seconds:return
        if now>=deadline:raise TimeoutError("Existing controller not stably idle; no new GPU job")
        sleep(min(30,max(0.,deadline-now)))

def load_report(path,data):
    report=json.loads((path/"report.json").read_text(encoding="utf-8"))
    if report.get("protocol")!="v6-refine-v1" or report.get("count")!=164:raise ValueError("Incomplete refinement analysis")
    if report["train_manifest_sha256"]!=ctl.sha(data/"train/samples.csv") or report["validation_manifest_sha256"]!=ctl.sha(data/"validation/samples.csv"):raise ValueError("Dataset manifest changed")
    if ctl.sha(path/"package/package_manifest.json")!=report["winner_package_manifest"]:raise ValueError("Analysis winner package changed")
    import v57_offline
    v57_offline.verify_package(path/"package")
    return report

def run(a):
    data=a.data_root.resolve();out=a.analysis.resolve() if a.analysis else research.ROOT/("analysis_"+research.stamp())
    if a.analysis:report=load_report(out,data)
    else:report=research.analyze(data,out,a.top,a.threads,a.gate_epochs,a.allow_public_assets)
    print("V6 REFINE PLAN",json.dumps({"parent":report["snapshot_champion"],"hypothesis":report["hypothesis"],"analysis":str(out)}),flush=True)
    if not a.execute_after_idle:return
    if not a.allow_public_assets:raise ValueError("Public asset permission required")
    # No automatic cancellation, no removal of another process's lock.
    wait_idle(a.wait_minutes)
    lock=core.OUTPUT/"active.lock";lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("x",encoding="utf-8") as f:f.write(str(os.getpid()))
    try:
        core.no_uncertain_prior_jobs()
        if (core.OUTPUT/"v57_audit_seal.json").exists():raise RuntimeError("Final independent audit sealed")
        code=ctl.source();code_sha=hashlib.sha256(code.encode()).hexdigest()
        from v571_preflight import require_report
        from v572_scale_audit import require_report as require_scale
        from v6_preflight import require
        policies=require_scale(data_root=data)
        if require_report()["sepformer_scale_policies"]!=policies:raise ValueError("Stale scale preflight")
        entries=ctl.ledger()
        done=[e for e in entries.values() if e[1] and not e[1]["smoke"]]
        if not done:raise ValueError("No complete checkpoint")
        champion=sorted(done,key=lambda e:(-e[1]["best_validation_pesq_wb"],e[0]["ref"]))[0]
        if ctl.parent_info(champion[0],champion[1])!=report["snapshot_champion"]:
            # Reuse cached predictions; re-analyze the current best instead of stale parent.
            out=research.ROOT/("analysis_"+research.stamp())
            report=research.analyze(data,out,a.top,a.threads,a.gate_epochs,a.allow_public_assets)
            if ctl.parent_info(champion[0],champion[1])!=report["snapshot_champion"]:raise RuntimeError("Champion identity changed unexpectedly")
        if champion[1]["v56_source_sha"]!=code_sha:raise ValueError("Champion source differs from accepted trainer; no silent migration")
        family=champion[2].get("v6_base_family") or champion[2]["architecture"]
        accepted=require(code_sha,data,[family]);locks=accepted.get("hf_locks",{})
        if family in ("semambapp","rose_cd") and champion[2].get("v6_hf_lock")!=locks.get(family):raise ValueError("Pretrained identity changed")
        kaggle=shutil.which("kaggle")
        if not kaggle:raise RuntimeError("Kaggle CLI missing")
        cache=champion[2].get("public_cache") or v4.find_public_cache()
        tag="v4"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        start=time.monotonic()
        def remaining():return min(a.budget_hours-(time.monotonic()-start)/3600,ctl.quota_retry(kaggle)-a.reserve_hours)
        smoke=build_trial(champion,report,code_sha,policies,locks,cache,a.epochs,smoke=True)
        trial=build_trial(champion,report,code_sha,policies,locks,cache,a.epochs)
        if trial["v56_key"] in entries:print("V6 REFINE STOP already attempted",flush=True);return
        smoke_entry=entries.get(smoke["v56_key"])
        if smoke_entry and not smoke_entry[1]:print("V6 REFINE STOP prior smoke failed",flush=True);return
        needed=cost(trial,entries)+(0 if smoke_entry else cost(smoke,entries))
        if needed>remaining() or max(cost(trial,entries),cost(smoke,entries))>a.max_job_minutes/60:
            print("V6 REFINE STOP insufficient budget",json.dumps({"estimated_hours":needed}),flush=True);return
        step=0
        if not smoke_entry:
            step+=1;print("V6 REFINE SMOKE",json.dumps(smoke),flush=True)
            ctl.submit(kaggle,code,tag,step,smoke,a.max_job_minutes)
        core.no_uncertain_prior_jobs()
        entries=ctl.ledger()
        if cost(trial,entries)>remaining() or cost(trial,entries)>a.max_job_minutes/60:
            print("V6 REFINE STOP after smoke, insufficient remaining budget",flush=True);return
        step+=1;print("V6 REFINE TRAIN",json.dumps(trial),flush=True)
        st,result=ctl.submit(kaggle,code,tag,step,trial,a.max_job_minutes)
        write_result={"parent":report["snapshot_champion"],"hypothesis":trial["v6_refine_hypothesis"],"result":result,"state":st}
        research.write(research.ROOT/("continuation_"+tag+".json"),write_result)
        print("V6 REFINE TRAIN RESULT",result["best_validation_pesq_wb"],flush=True)
        # Compare against all completed single experts and freeze new offline winner.
        # Existing artifacts remain unchanged, even if new training regresses.
        research.analyze(data,research.ROOT/("post_"+research.stamp()),a.top,a.threads,a.gate_epochs,a.allow_public_assets)
    finally:lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True);p.add_argument("--analysis",type=Path)
    p.add_argument("--top",type=int,choices=range(1,6),default=3);p.add_argument("--threads",type=int,choices=range(1,5),default=2)
    p.add_argument("--gate-epochs",type=int,choices=range(1,31),default=8)
    p.add_argument("--allow-public-assets",action="store_true");p.add_argument("--execute-after-idle",action="store_true")
    p.add_argument("--wait-minutes",type=int,default=720);p.add_argument("--max-job-minutes",type=int,default=360)
    p.add_argument("--epochs",type=int,choices=range(1,4),default=1)
    p.add_argument("--budget-hours",type=float,default=2.);p.add_argument("--reserve-hours",type=float,default=2.)
    a=p.parse_args()
    if not (0<=a.wait_minutes<=1440 and 15<=a.max_job_minutes<=360 and math.isfinite(a.budget_hours) and 0<a.budget_hours<=6 and math.isfinite(a.reserve_hours) and a.reserve_hours>=2):p.error("Invalid bounded budget")
    run(a)
if __name__=="__main__":main()
