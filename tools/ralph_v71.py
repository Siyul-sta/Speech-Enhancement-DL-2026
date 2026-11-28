"""V7.1 single-allocation research sessions; safe dry-run by default.
Each session evaluates up to four reviewed same-parent challengers. The best
fully verified candidate becomes eligible for the NEXT session's warm start.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone
import ralph_v2 as core
import ralph_v4 as v4
import ralph_v6 as ctl
import ralph_v6_refine as refine
import ralph_v610 as focus
import ralph_v611 as oldloop
import ralph_v7 as v7
import v7_search as search
import v6_refine_analysis as research

sys.path.insert(0,str(ctl.ROOT/"src"))
from v71_session import digest

TERMINAL={"ERROR","CANCELLED","CANCELED","CANCEL_ACKNOWLEDGED"}
ROOT=research.ROOT/"v71"

def archived_source(base,cfg,dataset):
    for name,val in (("EXPERIMENT_CONFIG",cfg),("DATASET_SLUG",dataset)):
        base,n=re.subn(r"^"+name+r" = .+$",lambda _:name+" = "+repr(val),base,count=1,flags=re.M)
        if n!=1:raise ValueError("Missing source config")
    compile(base,"v71_archived.py","exec");return base

def make_plan(parent,diag,entries,base,cache,hours,max_candidates,allow_reduce,enable_korean,chunk,max_epochs):
    sha=search.digest(base);old=parent[2];left=max_epochs-parent[1]["v6"]["attempted_epochs"]
    candidates=[];used=.15
    if left<1 or hours<=.15:return None
    for name,why in search.rank(parent,diag,entries,enable_korean):
        if len(candidates)>=max_candidates:break
        s=search.settings(old);s.update({k:v for k,v in search.BY_NAME[name][1].items() if k in s})
        minimum=3 if s["v6_critic"] and not s["v7_critic_warm_count"] else 1
        upper=min(left,max(minimum,chunk) if name in ("continue","lr_half","lr_quarter") else minimum)
        selected=None
        for epochs in range(upper,minimum-1,-1):
            for mix,pub in oldloop.profiles(old,allow_reduce):
                c=search.build(parent,name,sha,cache,epochs,mix,pub);sm=search.build(parent,name,sha,cache,epochs,mix,pub,True)
                if c["v56_key"] in entries:continue
                prior=entries.get(sm["v56_key"])
                if prior:
                    if not prior[1]:continue
                    search.verify_result(prior[1],sm)
                train_hours=search.cost(c,entries)
                estimate=train_hours+(0 if prior else search.cost(sm,entries))
                if not math.isfinite(estimate) or estimate<=0 or used+estimate>hours:continue
                selected={"config":c,"smoke":sm,"reuse_smoke":bool(prior),"train_hours":train_hours,"estimate_hours":estimate,
                          "family":why["family"],"priority":why["priority"],"selection":why}
                break
            if selected:break
        if selected:candidates.append(selected);used+=selected["estimate_hours"]
    if not candidates:return None
    return {"protocol":"v71-plan-v1","source":sha,"dataset":v4.DATASET,"hours":hours,"estimate_hours":used,
            "parent":ctl.parent_info(parent[0],parent[1]),"candidates":candidates}

def state_for(ref,cfg):
    return {"ref":ref,"kind":"v4_smoke" if cfg["smoke"] else "v4_external","status":"COMPLETE","v6":True,
            **{k:cfg[k] for k in ("v56_key","v56_source_sha","v56_action","architecture","epochs","sepformer_scale_policies")}}

def session_files(folder):
    plan=json.loads((folder/"v71_plan.json").read_text(encoding="utf-8"))
    base=(folder/"v71_base.txt").read_text(encoding="utf-8")
    if plan.get("protocol")!="v71-plan-v1" or search.digest(base)!=plan["source"]:raise ValueError("Archived session source mismatch")
    report=json.loads((folder/"v71_report.json").read_text(encoding="utf-8"))
    if report.get("protocol")!="v71-session-v1" or report["plan_sha"]!=digest(plan) or report.get("reason") not in ("finished","budget","stage_error"):
        raise ValueError("Unfinished or mismatched session report")
    return plan,base,report

def verify_stages(folder,ref):
    plan,base,report=session_files(folder)
    rows=[];trained=[];seen=set()
    for row in report["records"]:
        i=row["index"]
        if type(i) is not int or not 0<=i<len(plan["candidates"]) or i in seen:raise ValueError("Invalid stage index")
        seen.add(i);item=plan["candidates"][i]
        if row.get("smoke") not in ("reused","not_started","complete") or row.get("train") not in ("not_started","complete"):
            raise ValueError("Invalid stage status")
        if (row["smoke"]=="reused")!=bool(item["reuse_smoke"]):raise ValueError("Unverified reused smoke")
        if row["train"]=="complete" and row["smoke"] not in ("complete","reused"):raise ValueError("Training without smoke")
        for stage in ("smoke","train"):
            if row[stage]!="complete":continue
            cfg=item["smoke" if stage=="smoke" else "config"]
            if stage=="train" and cfg["v6_parent"]!=plan["parent"]:raise ValueError("Session parent changed")
            p=folder/"v71_trials"/f"trial_{i:02d}"/stage
            source=archived_source(base,cfg,plan["dataset"])
            (p/"source.py").write_text(source,encoding="utf-8")
            # Local temporary filename required by unchanged V6 verifier.
            cp=p/"best_model.pt"
            if cp.exists():
                if ctl.sha(cp)!=ctl.sha(p/"checkpoint.pt"):raise ValueError("Unexpected extra checkpoint")
            else:
                try:os.link(p/"checkpoint.pt",cp)
                except OSError:shutil.copy2(p/"checkpoint.pt",cp)
            st=state_for(ref,cfg)
            try:
                r,actual=ctl.verify(p,st);search.verify_result(r,cfg)
            finally:cp.unlink(missing_ok=True)
            if actual!=cfg:raise ValueError("Stage configuration changed")
            rows.append((st,r,cfg,p))
            if stage=="train":
                if abs(float(row["score"])-r["best_validation_pesq_wb"])>1e-6:raise ValueError("Stage score mismatch")
                trained.append((i,st,r,cfg,p))
    winner=max(trained,key=lambda t:t[2]["best_validation_pesq_wb"],default=None)
    if report["winner"]!=(winner[0] if winner else None):raise ValueError("Invalid session winner")
    if winner and ctl.sha(folder/"best_model.pt")!=winner[2]["v6"]["checkpoint_sha256"]:raise ValueError("Exported winner checkpoint mismatch")
    return rows,winner

def import_session(folder,state):
    rows,winner=verify_stages(folder,state["ref"])
    if winner:
        _,st,r,cfg,p=winner
        # Copy only canonical evidence for the winner, with a single real remote ref.
        for f in p.iterdir():
            if f.is_file() and f.name!="checkpoint.pt":shutil.copy2(f,folder/f.name)
        ctl.verify(folder,st);search.verify_result(r,cfg)
        state.update(st,v71_session=True)
    else:
        state.update(status="ERROR_TERMINAL",v71_no_candidate=True)
    core.json_write(folder/"remote_state.json",state)
    print("V71 IMPORT "+json.dumps({"ref":state["ref"],"verified_stages":len(rows),"winner":winner[3]["v7_recipe"] if winner else None,
        "pesq":winner[2]["best_validation_pesq_wb"] if winner else None}),flush=True)
    return winner

def session_history(entries):
    # Non-winning candidates are evidence, NOT eligible external warm-start parents.
    out=dict(entries)
    for path in sorted(core.OUTPUT.glob("run_v4*/round_*/*/remote_state.json")):
        st=json.loads(path.read_text(encoding="utf-8"))
        if not st.get("v71_session") or not (st.get("v6") or st.get("v71_no_candidate")):continue
        if st["status"] not in ("COMPLETE","ERROR_TERMINAL"):continue
        rows,_=verify_stages(path.parent,st["ref"])
        for row in rows:out.setdefault(row[2]["v56_key"],row)
    return out

def poll(kaggle,folder,state,minutes,clock=time.monotonic,sleep=time.sleep):
    deadline=clock()+minutes*60
    while clock()<deadline:
        try:
            p=ctl.command(kaggle,"kernels","status",state["ref"])
            m=re.search(r"KernelWorkerStatus\.([A-Z_]+)",p.stdout+" "+p.stderr) if p.returncode==0 else None
            worker=m[1] if m else None
        except (OSError,subprocess.TimeoutExpired):worker=None
        if worker=="COMPLETE":
            try:
                p=ctl.command(kaggle,"kernels","output",state["ref"],"-p",folder,"-o",timeout=300)
                if p.returncode:raise OSError("Output download failed")
            except (OSError,subprocess.TimeoutExpired):
                print("V71 OUTPUT_RETRY "+state["ref"],flush=True);sleep(60);continue
            return import_session(folder,state)
        if worker in TERMINAL:
            state.update(status="ERROR_TERMINAL",terminal_remote_status=worker)
            core.json_write(folder/"remote_state.json",state)
            print("V71 TERMINAL "+worker+" "+state["ref"],flush=True)
            return None
        print("V71 "+("REMOTE "+worker if worker else "NETWORK_WAIT")+" "+state["ref"],flush=True);sleep(60)
    raise TimeoutError("Unresolved V7.1 session; resume same ref, never push again: "+state["ref"])

def submit(kaggle,base,plan,minutes):
    tag="v4"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    folder=core.OUTPUT/("run_"+tag)/"round_0001"/"v4_external"
    job=ctl.ROOT/"kaggle_run"/"ralph_v4"/tag/"job_0001"
    folder.mkdir(parents=True);job.mkdir(parents=True)
    ref="siyul1122/research-speech-"+tag+"-job-1"
    script=(ctl.ROOT/"src/v71_session.py").read_text(encoding="utf-8")+"\nif __name__ == '__main__':\n    run_session("+repr(base)+","+repr(plan)+")\n"
    compile(script,"v71_remote.py","exec")
    (job/"run.py").write_text(script,encoding="utf-8")
    (folder/"v71_base.txt").write_text(base,encoding="utf-8")
    core.json_write(folder/"v71_plan.json",plan)
    sources=list(dict.fromkeys([plan["parent"]["ref"]]+[item["config"]["public_cache"] for item in plan["candidates"] if item["config"]["public_cache"]]))
    core.json_write(job/"kernel-metadata.json",{"id":ref,"title":"Research Speech "+tag+" job 1","code_file":"run.py","language":"python","kernel_type":"script","is_private":"true","enable_gpu":"true","enable_internet":"true","dataset_sources":[plan["dataset"]],"competition_sources":[],"kernel_sources":sources})
    state={"ref":ref,"status":"SUBMIT_PENDING","kind":"v71_session","v71_session":True,"plan_sha":digest(plan)}
    core.json_write(folder/"remote_state.json",state)
    try:
        p=ctl.command(kaggle,"kernels","push","-p",job,"--accelerator","NvidiaTeslaT4",timeout=180)
        print(p.stdout,p.stderr,flush=True)
    except (OSError,subprocess.TimeoutExpired) as exc:print("V71 PUSH UNKNOWN",repr(exc),flush=True)
    return poll(kaggle,folder,state,minutes)

def resume(kaggle,minutes):
    # Session records are intentionally not V6 candidates before complete verification.
    for p,st in list(ctl.recovery.pending_states()):
        if st.get("v71_session"):poll(kaggle,p.parent,st,minutes)
    ctl.resume(kaggle,minutes)

def run(a):
    data=a.data_root.resolve();base=search.source();sha=search.digest(base)
    base_sha=search.digest(ctl.source());focus_sha=search.digest(focus.source())
    seal=core.OUTPUT/"v57_audit_seal.json"
    if seal.exists():raise RuntimeError("Independent audit sealed")
    if not a.execute:
        p=oldloop.champion(v7.verified_entries());print("V71 DRY PLAN",json.dumps({"parent":ctl.parent_info(p[0],p[1]),"max_candidates":a.max_candidates,"session_hours":a.session_hours,"note":"No GPU submission"}),flush=True);return
    if not a.allow_public_assets:raise ValueError("Explicit public asset authorization required")
    kaggle=shutil.which("kaggle")
    if not kaggle:raise RuntimeError("Kaggle CLI missing")
    refine.wait_idle(3);lock=core.OUTPUT/"active.lock";lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("x",encoding="utf-8") as f:f.write(str(os.getpid()))
    try:
        if a.resume:resume(kaggle,a.max_job_minutes)
        else:core.no_uncertain_prior_jobs()
        from v6_preflight import require
        require(base_sha,data,["mp_dns"])
        count=0;cycle=1;wait_deadline=time.monotonic()+a.max_wait_hours*3600
        q=v7.quota(kaggle)
        while not a.max_sessions or count<a.max_sessions:
            core.no_uncertain_prior_jobs()
            if seal.exists():raise RuntimeError("Independent audit sealed")
            e=v7.verified_entries();p=oldloop.champion(e);v7.check_parent(p,base_sha,focus_sha,sha)
            if p[1]["v6"]["attempted_epochs"]>=a.max_epochs:break
            diag=oldloop.train_diagnostic(data,p,a.allow_public_assets)
            cache=p[2].get("public_cache") or v4.find_public_cache()
            q=v7.quota(kaggle);hours=min(a.session_hours,q["remaining"]-a.reserve_hours)
            plan=make_plan(p,diag,session_history(e),base,cache,hours,a.max_candidates,a.allow_reduce_mix,a.enable_korean,a.chunk_epochs,a.max_epochs)
            if not plan:
                if a.wait_for_quota and cycle<a.max_quota_cycles and hours<a.session_hours and time.monotonic()<wait_deadline:
                    new=v7.await_refresh(kaggle,q,wait_deadline,a.reserve_hours)
                    if new:cycle+=1;continue
                print("V71 STOP no affordable untried session",flush=True);break
            print("V71 PLAN "+json.dumps(plan),flush=True)
            for item in plan["candidates"]:
                focus.preflight(data,base,item["config"],ROOT/("preflight_"+research.stamp()),p)
            q=v7.quota(kaggle)
            if q["remaining"]-a.reserve_hours<plan["hours"]:
                print("V71 STOP quota changed before submit",flush=True);break
            count+=1
            winner=submit(kaggle,base,plan,a.max_job_minutes)
            if winner is None:
                print("V71 STOP no verified training result; inspect session",flush=True);break
        if not a.skip_final:
            core.no_uncertain_prior_jobs()
            report=research.analyze(data,research.ROOT/("v7_final_"+research.stamp()),a.top,2,8,a.allow_public_assets,prefer_diverse=False)
            print("V71 FINAL "+json.dumps({"winner":report["winner"],"pesq":report["winner_pesq"]}),flush=True)
    finally:lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True)
    for flag in ("execute","resume","allow-public-assets","allow-reduce-mix","enable-korean","wait-for-quota","skip-final"):
        p.add_argument("--"+flag,action="store_true")
    p.add_argument("--max-sessions",type=int,default=0);p.add_argument("--max-candidates",type=int,default=3)
    p.add_argument("--session-hours",type=float,default=4.5);p.add_argument("--max-job-minutes",type=int,default=360)
    p.add_argument("--reserve-hours",type=float,default=2.);p.add_argument("--max-epochs",type=int,default=200);p.add_argument("--chunk-epochs",type=int,default=3)
    p.add_argument("--max-quota-cycles",type=int,default=2);p.add_argument("--max-wait-hours",type=float,default=168.);p.add_argument("--top",type=int,choices=range(1,6),default=3)
    a=p.parse_args()
    if not (a.max_sessions>=0 and 1<=a.max_candidates<=4 and 0<a.session_hours<=5 and 15<=a.max_job_minutes<=720 and math.isfinite(a.reserve_hours) and a.reserve_hours>=2 and 1<=a.max_epochs<=200 and 1<=a.chunk_epochs<=10 and 1<=a.max_quota_cycles<=52 and 0<a.max_wait_hours<=720):p.error("Invalid bounded limits")
    run(a)
if __name__=="__main__":main()
