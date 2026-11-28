"""V7 autonomous bounded research loop, local CPU + private Kaggle GPU.
No external code execution or automatic paper ingestion. Dry-run default.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
import ralph_v6 as ctl
import ralph_v610 as focus
import ralph_v611 as oldloop
import ralph_v6_refine as refine
import ralph_v2 as core
import ralph_v4 as v4
import v6_refine_analysis as research
import v7_search as search

ROOT=research.ROOT/"v7"
WAIT_REASONS={"budget","budget_before_preflight","budget_after_preflight","budget_after_smoke","no_affordable_recipe"}

def parse_quota(output):
    """Read verified Kaggle CLI GPU row; never infer available hours from the calendar."""
    for line in output.splitlines():
        cells=line.strip().split()
        if len(cells)<4 or cells[0]!="GPU":continue
        try:used,remaining,total=[float(v.rstrip("h")) for v in cells[1:4]]
        except ValueError:continue
        if not all(math.isfinite(x) and x>=0 for x in (used,remaining,total)) or total<=0 or remaining>total+.05 or abs(used+remaining-total)>.15:
            continue
        m=re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})",line)
        refresh=None
        if m:
            refresh=datetime.fromisoformat(m.group().replace("Z","+00:00")).astimezone(timezone.utc).isoformat()
        return {"remaining":remaining,"used":used,"total":total,"refresh":refresh}
    raise ValueError("No valid GPU quota row")

def quota(kaggle,timeout_minutes=15,clock=time.monotonic,sleep=time.sleep):
    end=clock()+timeout_minutes*60
    while True:
        try:
            p=ctl.command(kaggle,"quota")
            if p.returncode:raise RuntimeError("Kaggle quota command failed")
            return parse_quota(p.stdout)
        except (OSError,RuntimeError,ValueError,subprocess.SubprocessError):
            if clock()>=end:raise
            print("V7 QUOTA_NETWORK_WAIT",flush=True);sleep(min(60,max(0.,end-clock())))

def refreshed(previous,current):
    # Require actual available GPU hours, not just an advertised reset timestamp.
    increased=current["remaining"]>previous["remaining"]+.25
    reset=bool(previous.get("refresh") and current.get("refresh") and current["refresh"]>previous["refresh"])
    return increased or (reset and current["used"]<previous["used"]-.25)

def await_refresh(kaggle,previous,deadline,reserve,poll_minutes=10,clock=time.monotonic,sleep=time.sleep,reader=None):
    reader=reader or (lambda:quota(kaggle,timeout_minutes=2))
    while clock()<deadline:
        sleep(min(poll_minutes*60,max(0.,deadline-clock())))
        if clock()>=deadline:break
        core.no_uncertain_prior_jobs()
        if (core.OUTPUT/"v57_audit_seal.json").exists():raise RuntimeError("Independent audit sealed")
        try:q=reader()
        except (OSError,RuntimeError,ValueError,subprocess.SubprocessError):
            print("V7 REFRESH_NETWORK_WAIT",flush=True);continue
        if refreshed(previous,q) and q["remaining"]>reserve:
            print("V7 QUOTA_REFRESHED",json.dumps(q),flush=True);return q
        print("V7 QUOTA_WAIT",json.dumps(q),flush=True)
    return None

def verified_entries():
    e=ctl.ledger()
    for st,r,c,folder in e.values():
        if r and c.get("v7_recipe"):search.verify_result(r,c)
    return e

def check_parent(p,base_sha,focus_sha,sha):
    if (p[2].get("v6_base_family") or p[2]["architecture"])!="mp_dns":raise ValueError("Overall champion no longer MP-DNS; explicit reassessment required")
    if p[1]["v56_source_sha"] not in (base_sha,focus_sha,sha):raise ValueError("Unrecognized parent source")

def run(a):
    data=a.data_root.resolve();code=search.source();sha=search.digest(code)
    base_sha=search.digest(ctl.source());focus_sha=search.digest(focus.source())
    if (core.OUTPUT/"v57_audit_seal.json").exists():raise RuntimeError("Independent audit sealed")
    if not a.execute:
        p=oldloop.champion(verified_entries())
        print("V7 DRY PLAN",json.dumps({"parent":ctl.parent_info(p[0],p[1]),"catalog":[x[0] for x in search.CATALOG],
            "quota_wait":a.wait_for_quota,"source":sha,"note":"No GPU; no remote quota call."}),flush=True)
        return
    if not a.allow_public_assets:raise ValueError("Explicit public asset authorization required")
    kaggle=shutil.which("kaggle")
    if not kaggle:raise RuntimeError("Kaggle CLI missing")
    # Old controllers must have exited. Never remove or steal another active lock.
    refine.wait_idle(3)
    lock=core.OUTPUT/"active.lock";lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("x",encoding="utf-8") as f:f.write(str(os.getpid()))
    try:
        if a.resume:ctl.resume(kaggle,a.max_job_minutes)
        else:core.no_uncertain_prior_jobs()
        from v6_preflight import require
        require(base_sha,data,["mp_dns"])
        ROOT.mkdir(parents=True,exist_ok=True)
        tag="v4"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        state=ROOT/("run_"+tag+".json")
        rounds=jobs=failures=stale=0;cycle=1;events=[];cycles=[];reason="running";wait_used=0.
        q=quota(kaggle)
        def begin(q):
            budget=max(0.,min(q["remaining"]-a.reserve_hours,a.budget_hours if a.budget_hours is not None else q["remaining"]))
            cycles.append({"cycle":cycle,"start_quota":q,"budget_hours":budget})
            return time.monotonic(),budget
        started,budget=begin(q);last_quota=q;observed_reset=False
        def observe_quota():
            nonlocal last_quota,observed_reset
            now=quota(kaggle)
            observed_reset=observed_reset or refreshed(last_quota,now)
            last_quota=now
            return now
        def save():
            core.json_write(state,{"protocol":"v7-controller-v1","source":sha,"rounds":rounds,"jobs":jobs,"cycle":cycle,
                "events":events,"cycles":cycles,"stop_reason":reason,"wait_hours":wait_used/3600})
        def remaining():
            return min(budget-(time.monotonic()-started)/3600,observe_quota()["remaining"]-a.reserve_hours)
        def next_cycle():
            nonlocal q,cycle,started,budget,wait_used,reason,last_quota,observed_reset
            if not a.wait_for_quota or cycle>=a.max_quota_cycles or wait_used>=a.max_wait_hours*3600:return False
            if a.max_rounds and rounds>=a.max_rounds:return False
            core.no_uncertain_prior_jobs()
            previous=observe_quota();cycles[-1]["end_quota"]=previous;cycles[-1]["reason"]=reason
            if observed_reset and previous["remaining"]>a.reserve_hours:
                # A reset may already have happened during the previous job or CPU analysis.
                new=previous
                print("V7 QUOTA_REFRESH_ALREADY_OBSERVED",json.dumps(new),flush=True)
            else:
                wait_start=time.monotonic()
                deadline=wait_start+a.max_wait_hours*3600-wait_used
                print("V7 QUOTA_WAIT_START",json.dumps({"previous":previous,"cycle":cycle}),flush=True);save()
                new=await_refresh(kaggle,previous,deadline,a.reserve_hours,a.poll_minutes)
                wait_used+=time.monotonic()-wait_start
            if new is None:reason="quota_wait_limit";save();return False
            q=new;last_quota=new;observed_reset=False;cycle+=1;started,budget=begin(q);reason="running";save();return True
        try:
            while not a.max_rounds or rounds<a.max_rounds:
                core.no_uncertain_prior_jobs()
                if (core.OUTPUT/"v57_audit_seal.json").exists():raise RuntimeError("Independent audit sealed")
                e=verified_entries();p=oldloop.champion(e);check_parent(p,base_sha,focus_sha,sha)
                if p[1]["v6"]["attempted_epochs"]>=a.max_epochs:reason="lineage_epoch_cap";break
                diag=oldloop.train_diagnostic(data,p,a.allow_public_assets)
                cache=p[2].get("public_cache") or v4.find_public_cache()
                rem=remaining()
                plan=search.pick(p,diag,e,sha,cache,rem,a.max_job_minutes/60,a.max_epochs,a.chunk_epochs,a.allow_reduce_mix,a.enable_korean)
                if not plan:
                    possible=search.pick(p,diag,e,sha,cache,min(q["total"]-a.reserve_hours,a.budget_hours or q["total"]),
                        a.max_job_minutes/60,a.max_epochs,a.chunk_epochs,a.allow_reduce_mix,a.enable_korean)
                    reason=("budget" if rem<=0 else "no_affordable_recipe") if possible else "no_viable_untried_recipe"
                    if possible and next_cycle():continue
                    break
                c,s=plan["config"],plan["smoke"]
                print("V7 PLAN",json.dumps({"round":rounds+1,"cycle":cycle,"remaining":rem,**plan}),flush=True)
                if plan["hours"]>remaining():
                    reason="budget_before_preflight"
                    if next_cycle():continue
                    break
                focus.preflight(data,code,c,ROOT/("preflight_"+research.stamp()),p)
                if plan["hours"]>remaining():
                    reason="budget_after_preflight"
                    if next_cycle():continue
                    break
                rounds+=1
                event={"round":rounds,"cycle":cycle,"recipe":c["v7_recipe"],"parent":c["v6_parent"],
                    "estimate_hours":plan["hours"],"selection":plan["reason"],"compound_data_change":plan["compound_data_change"],"status":"started"}
                events.append(event);save()
                try:
                    if not plan["reuse_smoke"]:
                        jobs+=1;save()
                        _,sm=ctl.submit(kaggle,code,tag,jobs,s,a.max_job_minutes);search.verify_result(sm,s)
                    core.no_uncertain_prior_jobs()
                    if search.cost(c,verified_entries())>min(remaining(),a.max_job_minutes/60):
                        reason="budget_after_smoke";event["status"]="smoke_only";save()
                        if next_cycle():continue
                        break
                    jobs+=1;save()
                    st,r=ctl.submit(kaggle,code,tag,jobs,c,a.max_job_minutes);search.verify_result(r,c)
                    gain=float(r["best_validation_pesq_wb"])-float(p[1]["best_validation_pesq_wb"])
                    event.update(status="complete",ref=st["ref"],pesq=r["best_validation_pesq_wb"],gain=gain,
                        metric_steps=r["v6"]["metric_steps"],critic_calibration=r["v6"]["critic_calibration"],
                        warmup_calibration=r["v7"]["warmup_calibration"])
                    print("V7 SCORE",json.dumps(event),flush=True)
                    failures=0;stale=0 if gain>=.002 else stale+1
                except ctl.legacy.KaggleTerminalFailure as exc:
                    failures+=1;stale+=1;event.update(status="terminal_failure",error=str(exc))
                    if failures>=3:reason="failure_budget";save();break
                save()
                if a.max_stale and stale>=a.max_stale:reason="stagnation";break
            else:reason="round_limit"
            save();print("V7 STOP",reason,flush=True)
            core.no_uncertain_prior_jobs()
            if not a.skip_final:
                report=research.analyze(data,research.ROOT/("v7_final_"+research.stamp()),a.top,2,8,a.allow_public_assets,prefer_diverse=False)
                print("V7 FINAL",json.dumps({"winner":report["winner"],"pesq":report["winner_pesq"]}),flush=True)
        except KeyboardInterrupt:
            reason="interrupted";save();raise
        except Exception:
            reason="error_or_unresolved";save();raise
    finally:lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True);p.add_argument("--execute",action="store_true")
    p.add_argument("--resume",action="store_true");p.add_argument("--allow-public-assets",action="store_true")
    p.add_argument("--max-rounds",type=int,default=0);p.add_argument("--max-stale",type=int,default=0)
    p.add_argument("--max-epochs",type=int,default=200);p.add_argument("--chunk-epochs",type=int,default=3)
    p.add_argument("--budget-hours",type=float);p.add_argument("--reserve-hours",type=float,default=2.)
    p.add_argument("--max-job-minutes",type=int,default=360);p.add_argument("--allow-reduce-mix",action="store_true")
    p.add_argument("--enable-korean",action="store_true");p.add_argument("--wait-for-quota",action="store_true")
    p.add_argument("--max-quota-cycles",type=int,default=2);p.add_argument("--max-wait-hours",type=float,default=168.)
    p.add_argument("--poll-minutes",type=int,default=10);p.add_argument("--top",type=int,choices=range(1,6),default=3)
    p.add_argument("--skip-final",action="store_true")
    a=p.parse_args()
    if not (a.max_rounds>=0 and a.max_stale>=0 and 1<=a.max_epochs<=200 and 1<=a.chunk_epochs<=10 and
        math.isfinite(a.reserve_hours) and a.reserve_hours>=2 and 15<=a.max_job_minutes<=360 and
        (a.budget_hours is None or math.isfinite(a.budget_hours) and 0<a.budget_hours<=30) and
        1<=a.max_quota_cycles<=52 and math.isfinite(a.max_wait_hours) and 0<a.max_wait_hours<=720 and 1<=a.poll_minutes<=60):p.error("Invalid bounded limits")
    run(a)
if __name__=="__main__":main()
