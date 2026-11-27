"""V6 unified maximum-PESQ controller. Dry run by default, no implicit cloud jobs."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import ralph_v4 as v4
import ralph_v56 as v56
import ralph_v2 as core
import ralph_loop as legacy
import reconcile_ralph_v2 as recovery

ROOT=v4.ROOT
FAMILIES=("mp_dns","mp_vb","sepformer_dns_transfer","cmgan_transfer","complex_residual","metricgan","sepformer_wham_transfer","dual_axis","residual_relaxed")
PRETRAINED=set(FAMILIES)-{"complex_residual","dual_axis","residual_relaxed"}
MILESTONES=(25,50,100,200)
TERMINAL={"ERROR","CANCELLED","CANCELED","CANCEL_ACKNOWLEDGED"}
ACTIVE={"RUNNING","QUEUED","PENDING","INITIALIZING","CANCEL_REQUESTED"}

def source():
    base=(ROOT/"src/kaggle_v4_external.py").read_text(encoding="utf-8")
    marker='if __name__ == "__main__":'
    if base.count(marker)!=1: raise ValueError("Base runner contract changed")
    base=base.split(marker)[0]
    marker='    if isinstance(model,CMGANTransfer):\n        n=model.net'
    if base.count(marker)!=1: raise ValueError("Transfer group contract changed")
    base=base.replace(marker,'    if isinstance(model,V6MPNet): return v6_mp_groups(model)\n'+marker,1)
    code=base+"\n"+"\n".join((ROOT/"src"/p).read_text(encoding="utf-8") for p in ("v6_mpsenet.py","v6_data.py","v6_training.py"))
    compile(code,"v6_runner.py","exec")
    return code

def sha(p):
    with Path(p).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()

def config_of(folder):
    rows=[x for x in (folder/"source.py").read_text(encoding="utf-8").splitlines() if x.startswith("EXPERIMENT_CONFIG = ")]
    if len(rows)!=1: raise ValueError("Missing unique source configuration")
    return ast.literal_eval(rows[0].split("=",1)[1].strip())

def verify(folder,state):
    kind=state["kind"];r=v4.verify(folder,kind);v4.verify_v56_metadata(r,state)
    cfg=config_of(folder);m=r.get("v6",{})
    if m.get("protocol")!="v6-max-pesq-v1" or m.get("mode")!=cfg.get("v6_mode") or m.get("parent")!=cfg.get("v6_parent"):
        raise ValueError("V6 identity mismatch")
    if m.get("checkpoint_sha256")!=sha(folder/"best_model.pt"): raise ValueError("V6 checkpoint corrupt")
    n=int(r["epochs_completed"]);b=int(r["best_epoch"]);p=m.get("parent")
    if not 0<=b<=n<=int(cfg["epochs"]) or (cfg["v6_mode"]!="zero" and n<1): raise ValueError("V6 epoch count")
    if m["attempted_epochs"]!=(p["attempted_epochs"] if p else 0)+n or m["checkpoint_epoch"]!=(p["checkpoint_epoch"] if p else 0)+b: raise ValueError("V6 lineage count")
    base=r["zero_shot"];bd=base["external_domains"]
    v4.verify_rows(folder/"zero_shot_validation.csv",164,base["challenge"])
    er=v4.verify_rows(folder/"zero_shot_external.csv",64,sum(bd)/2)
    if any(abs(sum(er[j*32:(j+1)*32])/32-bd[j])>1e-6 for j in range(2)): raise ValueError("V6 baseline domain rows")
    if abs(m["start_pesq"]-base["challenge"])>1e-6 or abs(m["gain"]-(r["best_validation_pesq_wb"]-base["challenge"]))>1e-6 or r["best_validation_pesq_wb"]<base["challenge"]-1e-5: raise ValueError("V6 best checkpoint regression")
    if m["root_external_domains"]!=(p["root_external_domains"] if p else bd): raise ValueError("V6 root domains")
    if p and (abs(base["challenge"]-p["score"])>.025 or any(abs(a-b)>.025 for a,b in zip(bd,p["external_domains"]))): raise ValueError("V6 parent score mismatch")
    if any(r["domain_scores"][k]<m["root_external_domains"][i]-float(cfg.get("v6_external_margin",.15))-1e-5 for i,k in enumerate(("external_normal","external_stress"))): raise ValueError("V6 root guard")
    if cfg.get("v6_depth"):
        v4.verify_rows(folder/"coarse_validation.csv",164,m["coarse_score"])
    if cfg.get("v6_critic") and (m["critic_updates"]<1 or not m["critic_calibration"]): raise ValueError("Missing V6 actual critic calibration")
    return r,cfg

def parent_info(st,r):
    m=r["v6"]
    return {"ref":st["ref"],"sha256":m["checkpoint_sha256"],"attempted_epochs":m["attempted_epochs"],
        "checkpoint_epoch":m["checkpoint_epoch"],"score":float(r["best_validation_pesq_wb"]),
        "external_domains":[r["domain_scores"][k] for k in ("external_normal","external_stress")],
        "root_external_domains":m["root_external_domains"]}

def ledger():
    out={}
    for p in sorted(core.OUTPUT.glob("run_v4*/round_*/*/remote_state.json")):
        st=json.loads(p.read_text(encoding="utf-8"))
        if not st.get("v6"): continue
        key=st["v56_key"]
        if key in out: raise ValueError("Duplicate V6 identity")
        r=cfg=None
        if st["status"]=="COMPLETE": r,cfg=verify(p.parent,st)
        out[key]=(st,r,cfg,p.parent)
    complete={st["ref"]:(st,r,cfg) for st,r,cfg,_ in out.values() if r}
    for st,r,cfg,_ in out.values():
        if not r or not cfg.get("v6_parent"): continue
        p=cfg["v6_parent"]
        if p["ref"] not in complete: raise ValueError("Missing verified V6 parent")
        ps,pr,pc=complete[p["ref"]]
        if parent_info(ps,pr)!=p: raise ValueError("Changed V6 parent checkpoint")
        if cfg.get("v6_parent_coarse"):
            if pc["architecture"]!=cfg["v6_base_family"] or pc.get("v6_depth"): raise ValueError("Invalid cascade parent family")
        elif pc["architecture"]!=cfg["architecture"] or pc.get("v6_base_family")!=cfg.get("v6_base_family") or pc.get("v6_depth")!=cfg.get("v6_depth"): raise ValueError("Wrong continuation family")
    return out

def config(family,mode,code_sha,policies,cache=None,epochs=0,mix=2048,public_mix=512,korean=0,critic=False,hard=False,parent=None,base=None,depth=None,coarse=False,smoke=False,lr=None):
    base_family=base or family
    cfg={"id":"V3_RESIDUAL_PHASE","architecture":family,"epochs":epochs,"seed":42,"batch_size":4 if base_family in ("complex_residual","residual_relaxed","dual_axis") else 1,
         "patience":5,"selection_target":"challenge","public_cache":cache,"warm_ref":parent["ref"] if parent else None,
         "v6_parent":parent,"v6_parent_coarse":coarse,"v6_mode":mode,"v6_base_family":base_family if depth else None,
         "v6_depth":depth,"v6_mix":mix,"v6_public_mix":public_mix,"v6_korean":korean,"v6_critic":critic,"v6_hard":hard,"v6_phase":True,
         "v6_crop":48000 if critic else 16000,"v6_external_margin":.15,"v6_transfer":"progressive" if mode=="train" and family in PRETRAINED else None,
         "lr":lr if lr is not None else (3e-5 if base_family in PRETRAINED else 1e-4),"smoke":smoke}
    cfg["sepformer_scale_policies"]={base_family:policies[base_family]} if base_family in policies else {}
    action="smoke" if smoke else "v6_"+mode
    cfg.update(v56_source_sha=code_sha,v56_action=action,v56_key=v56.signature([code_sha,action,cfg]))
    return cfg

def estimate(cfg,entries):
    if cfg["v6_mode"]=="zero": return .65 if cfg["architecture"].startswith("mp_") else .4
    if cfg["smoke"]: return .35
    arch=cfg["architecture"];pairs=1100+cfg["v6_mix"]+cfg["v6_public_mix"]
    base=cfg.get("v6_base_family") or arch
    rates=[];evals=[]
    for st,r,c,_ in entries.values():
        if not r or c["architecture"]!=arch or c.get("v6_base_family")!=cfg.get("v6_base_family") or c.get("v6_crop")!=cfg.get("v6_crop") or c.get("v6_critic")!=cfg.get("v6_critic"): continue
        ts=r.get("v6_train_seconds",[]);total=r.get("epoch_seconds",[]);n=r["v6"]["training_pairs_per_epoch"]
        if n and ts:
            rates.extend(t/n for t in ts if t>0)
            evals.extend(max(0.,a-b) for a,b in zip(total,ts))
    seconds=statistics.median(rates) if rates else (1.2 if base.startswith(("mp_","sepformer_")) else .7 if base in PRETRAINED else .12)
    val=statistics.median(evals) if evals else (100 if base in PRETRAINED else 40)
    return (.25 if cfg.get("public_cache") else .6)+1.4*(2*val+cfg["epochs"]*(seconds*pairs+val))/3600

def choose(entries,code_sha,policies,cache,remaining,job_limit,args):
    current=[(st,r,c) for st,r,c,_ in entries.values() if r and r["v56_source_sha"]==code_sha and not r["smoke"]]
    by_ref={st["ref"]:(st,r,c) for st,r,c in current}
    limit=min(remaining,job_limit)
    def affordable(c): return c["v56_key"] not in entries and estimate(c,entries)<=limit
    def with_smoke(c):
        # Same architecture/depth/critic/crop; synthetic smoke does not claim parent reproduction.
        s=config(c["architecture"],c["v6_mode"],code_sha,policies,epochs=2,mix=16,public_mix=16,
            critic=c["v6_critic"],base=c.get("v6_base_family"),depth=c.get("v6_depth"),smoke=True)
        if s["v56_key"] in entries:
            st,r,_,_=entries[s["v56_key"]]
            return c if r and st["status"]=="COMPLETE" and affordable(c) else None
        return s if affordable(s) and affordable(c) and estimate(s,entries)+estimate(c,entries)<=remaining else None
    # Always benchmark the new pretrained MP models first. Actual score, not literature score.
    for family in ("mp_dns","mp_vb"):
        if family not in args.families: continue
        c=config(family,"zero",code_sha,policies,cache)
        if affordable(c): return c
    zeros={c["architecture"]:(st,r,c) for st,r,c in current if c["v6_mode"]=="zero"}
    trained=[(st,r,c) for st,r,c in current if c["v6_mode"] in ("train","cascade")]
    trained_families={c["architecture"] for _,_,c in trained}
    ranked=sorted(args.families,key=lambda f:(-zeros[f][1]["best_validation_pesq_wb"] if f in zeros else -2.32 if f=="sepformer_dns_transfer" else -2.05 if f=="cmgan_transfer" else -1.8, FAMILIES.index(f)))
    # Give an initial native pilot independent of public-model score.
    if len(trained_families)>=1 and "complex_residual" not in trained_families and "complex_residual" in ranked: ranked.remove("complex_residual");ranked.insert(0,"complex_residual")
    pilots=[]
    for family in ranked:
        if family in trained_families: continue
        parent=parent_info(*zeros[family][:2]) if family in zeros else None
        c=config(family,"train",code_sha,policies,cache,epochs=args.pilot_epochs,mix=args.mix,public_mix=args.public_mix,parent=parent)
        trial=with_smoke(c)
        if trial: pilots.append(trial)
    # One continuation per exact parent/recipe, including terminal failures.
    tried=set()
    for st,r,c,folder in entries.values():
        if st.get("v56_source_sha")!=code_sha: continue
        if c is None: c=config_of(folder)
        p=c.get("v6_parent")
        if p and not c["smoke"]:
            tried.add((p["ref"],c["architecture"],c.get("v6_base_family"),c.get("v6_depth"),c["v6_mix"],c["v6_public_mix"],c["v6_korean"],c["v6_hard"],c["v6_critic"]))
    def identity(c):
        return (c["v6_parent"]["ref"],c["architecture"],c.get("v6_base_family"),c.get("v6_depth"),c["v6_mix"],c["v6_public_mix"],c["v6_korean"],c["v6_hard"],c["v6_critic"])
    extensions=[]
    for st,r,c in sorted(trained,key=lambda a:-a[1]["best_validation_pesq_wb"]):
        m=r["v6"];p=parent_info(st,r)
        if m["attempted_epochs"]>=args.max_epochs or m["gain"]<.005 or m["checkpoint_epoch"]==0: continue
        if trained and r["best_validation_pesq_wb"]<max(a[1]["best_validation_pesq_wb"] for a in trained)-.2: continue
        # Stable data strategy; optional new controlled hypotheses after improvement.
        cap=min(args.chunk_epochs,args.max_epochs-m["attempted_epochs"])
        if cap<1: continue
        target=next((n for n in MILESTONES if n>m["attempted_epochs"]),args.max_epochs)
        cap=min(cap,target-m["attempted_epochs"])
        for n in range(cap,max(0,min(2,cap)-1),-1):
            nxt=config(c["architecture"],c["v6_mode"],code_sha,policies,cache,epochs=n,mix=c["v6_mix"],public_mix=c["v6_public_mix"],
                korean=c["v6_korean"],critic=c["v6_critic"],hard=c["v6_hard"],parent=p,base=c.get("v6_base_family"),depth=c.get("v6_depth"),
                lr=max(1e-8,min(float(c["lr"])*.5,float(m["lr"]))))
            if identity(nxt) in tried: break
            trial=with_smoke(nxt)
            if trial:
                extensions.append((r["best_validation_pesq_wb"]+.05*min(1.,m["gain"]/max(.01,estimate(nxt,entries))),trial));break
    # New hypotheses: more data, Korean, hard mining and calibrated critic. Explicit separate candidates.
    variants=[]
    if trained:
        for st,r,c in sorted(trained,key=lambda a:-a[1]["best_validation_pesq_wb"])[:2]:
            if c.get("v6_depth") or r["v6"]["checkpoint_epoch"]==0 or r["v6"]["attempted_epochs"]>=args.max_epochs: continue
            p=parent_info(st,r);cap=min(args.pilot_epochs,args.max_epochs-p["attempted_epochs"])
            if cap<1: continue
            common=dict(epochs=cap,mix=c["v6_mix"],public_mix=c["v6_public_mix"],parent=p,lr=max(1e-8,float(c["lr"])*.5))
            if c["v6_mix"]<args.max_mix:
                q=config(c["architecture"],"train",code_sha,policies,cache,**{**common,"mix":min(args.max_mix,c["v6_mix"]*2)},korean=c["v6_korean"],hard=c["v6_hard"],critic=c["v6_critic"])
                if identity(q) not in tried:
                    trial=with_smoke(q)
                    if trial: variants.append(trial)
            if args.korean_limit and not c["v6_korean"]:
                q=config(c["architecture"],"train",code_sha,policies,cache,**common,korean=args.korean_limit,hard=c["v6_hard"],critic=c["v6_critic"])
                if identity(q) not in tried:
                    trial=with_smoke(q)
                    if trial: variants.append(trial)
            if not c["v6_hard"]:
                q=config(c["architecture"],"train",code_sha,policies,cache,**common,korean=c["v6_korean"],hard=True,critic=c["v6_critic"])
                if identity(q) not in tried:
                    trial=with_smoke(q)
                    if trial: variants.append(trial)
            if not c["v6_critic"] and args.enable_critic:
                q=config(c["architecture"],"train",code_sha,policies,cache,**common,korean=c["v6_korean"],hard=c["v6_hard"],critic=True)
                if identity(q) not in tried:
                    trial=with_smoke(q)
                    if trial: variants.append(trial)
            if args.enable_cascade:
                for depth in (1,2,3):
                    q=config("v6_cascade", "cascade",code_sha,policies,cache,**common,base=c["architecture"],depth=depth,coarse=True)
                    if identity(q) not in tried:
                        trial=with_smoke(q)
                        if trial: variants.append(trial)
    extensions.sort(key=lambda a:-a[0])
    if pilots and len(trained_families)<args.min_pilots: return pilots[0]
    # Alternate focused extensions with controlled experiments instead of exhausting all variants first.
    if extensions and (len(trained)%3!=0 or not variants): return extensions[0][1]
    if variants: return variants[0]
    if extensions: return extensions[0][1]
    return pilots[0] if pilots else None

def command(kaggle,*args,timeout=90):
    return subprocess.run([str(kaggle),*map(str,args)],cwd=ROOT,capture_output=True,text=True,encoding="utf-8",errors="replace",timeout=timeout,check=False)

def poll(kaggle,folder,state,deadline,interval=60,clock=time.monotonic,sleep=time.sleep):
    """Never pushes. Network loss is unknown, not terminal; retry within deadline."""
    ref=state["ref"]
    while clock()<deadline:
        try:
            p=command(kaggle,"kernels","status",ref)
            match=re.search(r"KernelWorkerStatus\.([A-Z_]+)",p.stdout+" "+p.stderr) if p.returncode==0 else None
            worker=match.group(1) if match else None
        except (OSError,subprocess.TimeoutExpired) as exc:
            worker=None
        if not worker:
            print("V6 NETWORK_WAIT "+ref,flush=True)
            core.dashboard_event(folder.parents[1],"training",remote_status="NETWORK_WAIT",ref=ref)
        elif worker=="COMPLETE":
            try:
                output=command(kaggle,"kernels","output",ref,"-p",folder,"-o",timeout=300)
                if output.returncode!=0: raise OSError(output.stderr[-2000:])
            except (OSError,subprocess.TimeoutExpired):
                print("V6 OUTPUT_RETRY "+ref,flush=True);sleep(interval);continue
            r,cfg=verify(folder,state)
            state["status"]="COMPLETE";core.json_write(folder/"remote_state.json",state)
            core.dashboard_event(folder.parents[1],"scoring",remote_status="COMPLETE",ref=ref)
            return r
        elif worker in TERMINAL:
            state["status"]="ERROR_TERMINAL";core.json_write(folder/"remote_state.json",state)
            print("V6 TERMINAL "+worker+" "+ref,flush=True)
            raise legacy.KaggleTerminalFailure(worker+" "+ref)
        else:
            print("V6 REMOTE "+worker+" "+ref,flush=True)
            core.dashboard_event(folder.parents[1],"training",remote_status=worker,ref=ref)
        sleep(interval)
    raise TimeoutError("V6 remote unresolved; resume same job, never resubmit: "+ref)

def submit(kaggle,code,tag,round_no,cfg,wait):
    kind="v4_smoke" if cfg["smoke"] else "v4_external"
    folder=core.OUTPUT/("run_"+tag)/f"round_{round_no:04d}"/kind
    job=ROOT/"kaggle_run"/"ralph_v4"/tag/f"job_{round_no:04d}"
    folder.mkdir(parents=True);job.mkdir(parents=True)
    title=f"Research Speech {tag} job {round_no}";ref=f"siyul1122/research-speech-{tag}-job-{round_no}"
    code,n=re.subn(r"^DATASET_SLUG = .+$",lambda _: "DATASET_SLUG = "+repr(v4.DATASET),code,count=1,flags=re.M)
    if n!=1: raise ValueError("Missing dataset config")
    code,n=re.subn(r"^EXPERIMENT_CONFIG = .+$",lambda _: "EXPERIMENT_CONFIG = "+repr(cfg),code,count=1,flags=re.M)
    if n!=1: raise ValueError("Missing experiment config")
    compile(code,"run.py","exec")
    (job/"run.py").write_text(code,encoding="utf-8");(folder/"source.py").write_text(code,encoding="utf-8")
    core.json_write(job/"kernel-metadata.json",{"id":ref,"title":title,"code_file":"run.py","language":"python","kernel_type":"script","is_private":"true","enable_gpu":"true","enable_internet":"true","dataset_sources":[v4.DATASET],"competition_sources":[],"kernel_sources":list(dict.fromkeys(cfg[k] for k in ("warm_ref","public_cache") if cfg.get(k)))})
    state={"ref":ref,"kind":kind,"status":"SUBMIT_PENDING","v6":True,**{k:cfg[k] for k in ("v56_key","v56_source_sha","v56_action","architecture","epochs","sepformer_scale_policies")}}
    core.json_write(folder/"remote_state.json",state)
    # Exactly one push even when connection breaks; then same-ref polling only.
    try:
        p=command(kaggle,"kernels","push","-p",job,"--accelerator","NvidiaTeslaT4",timeout=180)
        print(p.stdout,p.stderr,flush=True)
    except (OSError,subprocess.TimeoutExpired) as exc: print("V6 PUSH UNKNOWN",repr(exc),flush=True)
    r=poll(kaggle,folder,state,time.monotonic()+wait*60)
    return state,r

def resume(kaggle,wait):
    pending=list(recovery.pending_states())
    for path,st in pending:
        if not st.get("v6"): raise RuntimeError("Resolve legacy job separately before V6")
        try: poll(kaggle,path.parent,st,time.monotonic()+wait*60)
        except legacy.KaggleTerminalFailure: pass
    core.no_uncertain_prior_jobs()

def quota_retry(kaggle,wait_minutes=90):
    deadline=time.monotonic()+wait_minutes*60
    while True:
        try: return v56.quota_hours(kaggle)
        except (OSError,RuntimeError,subprocess.SubprocessError):
            if time.monotonic()>=deadline: raise
            print("V6 QUOTA_NETWORK_WAIT",flush=True);time.sleep(60)

def prepare_cache(kaggle,code,korean,dns,wait):
    # Separate private CPU kernel, never attaches challenge dataset or models.
    tag="v4"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    folder=core.OUTPUT/("run_"+tag)/"round_0000"/"v4_cache"
    job=ROOT/"kaggle_run"/"ralph_v4"/tag/"public_cache"
    folder.mkdir(parents=True);job.mkdir(parents=True)
    ref=f"siyul1122/ralph-public-cache-{tag}"
    cfg={"v6_cache_build":True,"v6_cache_korean":korean,"v6_cache_dns":dns}
    code,n=re.subn(r"^EXPERIMENT_CONFIG = .+$",lambda _: "EXPERIMENT_CONFIG = "+repr(cfg),code,count=1,flags=re.M)
    if n!=1: raise ValueError("Cache source contract")
    (job/"run.py").write_text(code,encoding="utf-8")
    core.json_write(job/"kernel-metadata.json",{"id":ref,"title":"Ralph Public Cache "+tag,"code_file":"run.py","language":"python","kernel_type":"script","is_private":"true","enable_gpu":"false","enable_internet":"true","dataset_sources":[],"kernel_sources":[],"competition_sources":[]})
    state={"ref":ref,"kind":"v4_cache","status":"SUBMIT_PENDING","v6_cache":True}
    core.json_write(folder/"remote_state.json",state)
    try:
        p=command(kaggle,"kernels","push","-p",job,timeout=180)
        print(p.stdout,p.stderr,flush=True)
    except (OSError,subprocess.TimeoutExpired) as exc: print("V6 CACHE PUSH UNKNOWN",repr(exc),flush=True)
    deadline=time.monotonic()+wait*60
    while time.monotonic()<deadline:
        try: worker,_=recovery.status(kaggle,ref)
        except (OSError,subprocess.SubprocessError): worker=None
        if worker=="COMPLETE":
            state["status"]="COMPLETE";core.json_write(folder/"remote_state.json",state)
            print("V6 CPU CACHE COMPLETE",ref,flush=True);return ref
        if worker in TERMINAL:
            state["status"]="ERROR_TERMINAL";core.json_write(folder/"remote_state.json",state)
            raise legacy.KaggleTerminalFailure("CPU cache "+worker)
        print("V6 CPU CACHE",worker or "NETWORK_WAIT",ref,flush=True)
        time.sleep(60)
    raise TimeoutError("CPU cache unresolved; use reconcile, no duplicate push")

def run(args):
    code=source();code_sha=hashlib.sha256(code.encode()).hexdigest()
    if not args.execute and not args.prepare_cache:
        print("V6 DRY RUN — no GPU or remote CPU submission",flush=True)
        print("Families:",args.families,"mix:",args.mix,"max mix:",args.max_mix,"epochs:",args.max_epochs)
        print("Source SHA256:",code_sha)
        return
    if not args.allow_public_assets: raise RuntimeError("Explicit instructor permission for public assets required")
    if (core.OUTPUT/"v57_audit_seal.json").exists(): raise RuntimeError("Final independent audit sealed")
    kaggle=shutil.which("kaggle")
    if not kaggle: raise RuntimeError("Kaggle CLI missing")
    lock=core.OUTPUT/"active.lock";lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("x",encoding="utf-8") as f: f.write(str(os.getpid()))
    try:
        if args.resume: resume(kaggle,args.max_job_minutes)
        else: core.no_uncertain_prior_jobs()
        if args.prepare_cache:
            prepare_cache(kaggle,code,args.cache_korean,args.cache_dns_noise,args.max_job_minutes)
            return
        from v571_preflight import require_report
        from v572_scale_audit import require_report as require_scale
        policies=require_scale(data_root=args.data_root)
        if require_report().get("sepformer_scale_policies")!=policies: raise ValueError("Stale public preflight")
        from v6_preflight import require
        require(code_sha,args.data_root,args.families)
        cache=args.public_cache_ref or v4.find_public_cache()
        tag="v4"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        started=time.monotonic();rounds=0;failures=0;quota=quota_retry(kaggle)
        budget=args.budget_hours or max(0.,quota-args.reserve_hours)
        while not args.max_rounds or rounds<args.max_rounds:
            core.no_uncertain_prior_jobs()
            if (core.OUTPUT/"v57_audit_seal.json").exists(): raise RuntimeError("Final independent audit sealed")
            remaining=min(quota_retry(kaggle)-args.reserve_hours,budget-(time.monotonic()-started)/3600)
            if remaining<=0: break
            entries=ledger()
            cfg=choose(entries,code_sha,policies,cache,remaining,args.max_job_minutes/60,args)
            if cfg is None: print("V6 STOP no affordable candidate",flush=True);break
            cost=estimate(cfg,entries)
            if cost>min(quota_retry(kaggle)-args.reserve_hours,budget-(time.monotonic()-started)/3600): break
            rounds+=1
            print("V6 PLAN "+json.dumps({"round":rounds,"estimated_hours":cost,"remaining":remaining,"config":cfg}),flush=True)
            try:
                st,r=submit(kaggle,code,tag,rounds,cfg,args.max_job_minutes)
                failures=0
                if not r["smoke"]: print("V6 SCORE",r["best_validation_pesq_wb"],flush=True)
            except legacy.KaggleTerminalFailure:
                failures+=1
                if failures>=3: break
                continue
            fresh=ledger()
            eligible=[(s,r) for s,r,c,_ in fresh.values() if r and not r["smoke"]]
            if eligible:
                best=max(eligible,key=lambda a:a[1]["best_validation_pesq_wb"])
                core.json_write(core.OUTPUT/("run_"+tag)/"v6_best.json",{"state":best[0],"result":best[1]})
    finally: lock.unlink(missing_ok=True)
    if args.execute and args.finalize_out:
        core.no_uncertain_prior_jobs()
        from v6_finalize import finalize
        with lock.open("x",encoding="utf-8") as f: f.write(str(os.getpid()))
        try:
            finalize(argparse.Namespace(data_root=args.data_root,out=args.finalize_out,top=args.finalize_top,gate_epochs=12,mp_root=None,cmgan_root=None,allow_public_download=args.allow_public_assets))
        finally: lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true");p.add_argument("--prepare-cache",action="store_true")
    p.add_argument("--allow-public-assets",action="store_true");p.add_argument("--resume",action="store_true")
    p.add_argument("--cache-korean",type=int,default=512);p.add_argument("--cache-dns-noise",action="store_true")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--families",nargs="+",choices=FAMILIES,default=list(FAMILIES))
    p.add_argument("--mix",type=int,default=2048);p.add_argument("--public-mix",type=int,default=512);p.add_argument("--max-mix",type=int,default=32768)
    p.add_argument("--korean-limit",type=int,default=256);p.add_argument("--pilot-epochs",type=int,default=3)
    p.add_argument("--chunk-epochs",type=int,default=10);p.add_argument("--max-epochs",type=int,default=200)
    p.add_argument("--min-pilots",type=int,default=3);p.add_argument("--max-rounds",type=int,default=0)
    p.add_argument("--max-job-minutes",type=int,default=360);p.add_argument("--budget-hours",type=float,default=0.)
    p.add_argument("--reserve-hours",type=float,default=2.);p.add_argument("--public-cache-ref")
    p.add_argument("--enable-critic",action="store_true");p.add_argument("--enable-cascade",action="store_true")
    p.add_argument("--finalize-out",type=Path,help="Optional offline selection after GPU loop stops cleanly")
    p.add_argument("--finalize-top",type=int,default=3,choices=range(1,6))
    args=p.parse_args()
    if args.execute and args.prepare_cache: p.error("Use separate CPU cache command")
    if not (1<=args.mix<=args.max_mix<=1000000 and 1<=args.public_mix<=32768 and 0<=args.korean_limit<=2000 and 0<=args.cache_korean<=2000 and
        1<=args.pilot_epochs<=args.max_epochs<=500 and 1<=args.chunk_epochs<=50 and 1<=args.min_pilots<=len(FAMILIES) and
        args.max_rounds>=0 and 15<=args.max_job_minutes<=720 and math.isfinite(args.reserve_hours) and args.reserve_hours>=2 and math.isfinite(args.budget_hours) and args.budget_hours>=0):
        p.error("Invalid V6 limits")
    run(args)
if __name__=="__main__": main()
