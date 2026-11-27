"""V5.6 unified Ralph scheduler: private serial GPU jobs, bounded cost estimates."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import ralph_v4 as v4
import ralph_v2 as core
import ralph_loop as legacy

def signature(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def source_hash(source):
    return hashlib.sha256(source.encode()).hexdigest()

def quota_hours(kaggle):
    output=legacy.command([kaggle,"quota"],capture=True)
    for line in output.splitlines():
        cells=line.strip().split()
        if len(cells)>=4 and cells[0]=="GPU":
            try: remaining=float(cells[2].rstrip("h"))
            except ValueError: continue
            if math.isfinite(remaining) and remaining>=0: return remaining
    raise RuntimeError("Cannot verify GPU quota; no submission")

def states():
    out={}
    for path in sorted(core.OUTPUT.glob("run_v4*/round_*/*/remote_state.json")):
        state=json.loads(path.read_text(encoding="utf-8"))
        if not state.get("v56_key"): continue
        if state["v56_key"] in out: raise RuntimeError("Duplicate V5.6 experiment identity")
        if state.get("status")=="COMPLETE":
            result=v4.verify(path.parent,state["kind"])
            v4.verify_v56_metadata(result,state)
            if not (path.parent/"source.py").is_file(): raise RuntimeError("Completed V5.6 archived source missing")
        out[state["v56_key"]]=state
    return out

def verified_smoke(key,ledger):
    s=ledger.get(key)
    return bool(s and s.get("status")=="COMPLETE" and s.get("kind")=="v4_smoke")

def stage_catalog(records,epochs):
    """Collect the existing typed strategies without modifying their comparisons."""
    result=[]
    for fn,field,rung in ((v4.v52_stage_plan,"v52_stage",5),(v4.v53_stage_plan,"v53_stage",12),(v4.v54_stage_plan,"v54_stage",8)):
        seen=[]
        while True:
            p=fn(records,seen,rung)
            if p is None: break
            seen.append({"plan":p})
            result.append((field+":"+p[field],p,rung,max(rung,epochs)))
        # Include earlier completed stage names too; planner above may have skipped them.
        while True:
            p=fn([],seen,rung)
            if p is None: break
            seen.append({"plan":p})
            result.append((field+":"+p[field],p,rung,max(rung,epochs)))
    # Preserve explicit pinned/frozen state fingerprints from verified history.
    for name,p,_,_ in result:
        arch=p["architecture"]
        if "v53_stage" in p:
            p["v53_expected_initial_hash"]=next((r.get("v53_initial_hash") for _,r in records if r.get("model_family")==arch and r.get("v53_initial_hash")),None)
        if arch.startswith("cascade_cmgan"):
            p["v54_expected_coarse_hash"]=next((r.get("cascade_coarse_hash") for _,r in records if r.get("cascade_pretrained") and r.get("cascade_coarse_hash")),None)
    priority=["v53_stage:cmgan-decoder","v52_stage:phase-control","v52_stage:complex-head","v54_stage:native-two",
              "v53_stage:metricgan-head","v52_stage:phase-loss","v54_stage:cmgan-two","v52_stage:adaptive-critic"]
    result.sort(key=lambda row:(priority.index(row[0]) if row[0] in priority else len(priority),row[0]))
    return result

def key_for(source_sha,action,name,epochs,seed,cache,parent=None,settings=None):
    return signature([source_sha,action,name,epochs,seed,cache,parent,settings])

def base_cfg(p,epochs,seed,cache):
    arch=p["architecture"]
    return {"id":"V3_RESIDUAL_PHASE","epochs":epochs,"batch_size":1 if arch.startswith("sepformer_") or arch.startswith("cascade_") or arch=="cmgan_transfer" else 4,
            "patience":7,"seed":seed,"warm_ref":None,"public_cache":cache,"amp":False,"selection_target":"challenge",**p}

def trial_cfg(p,name,epochs,seed,cache,sha,action,parent=None):
    cfg=base_cfg(p,epochs,seed,cache)
    # Native continuation is only allowed from an exact same-source/same-stage pilot.
    if parent and action=="promote" and "v52_stage" in p and not p.get("pesq_aware"):
        cfg["warm_ref"]=parent["ref"]
        cfg["lr"]=float(cfg["lr"])*.5
    identity={k:v for k,v in cfg.items() if k not in ("v53_expected_initial_hash","v54_expected_coarse_hash")}
    cfg.update(v56_key=key_for(sha,action,name,epochs,seed,cache,parent.get("ref") if parent else None,identity),
               v56_action=action,v56_source_sha=sha)
    return cfg

def smoke_cfg(arch,sha):
    p={"architecture":arch,"epochs":2,"batch_size":1 if arch.startswith(("cascade_","sepformer_")) or arch=="cmgan_transfer" else 4,
       "id":"V3_RESIDUAL_PHASE","patience":2,"seed":42,"smoke":True,"external_size":16,"public_cache":None,
       "amp":False,"pesq_aware":False,"data_profile":"base","selection_target":"challenge"}
    p.update(v56_key=key_for(sha,"smoke",arch,2,42,None),v56_action="smoke",v56_source_sha=sha)
    return p

def observed_epoch_hours(records,arch):
    values=[]
    for _,r in records:
        if r.get("model_family")!=arch or r.get("smoke") or r.get("evaluation_protocol")!="v4.2-fixed-dev-v2": continue
        t=r.get("epoch_seconds",[])
        values.extend(float(v)/3600 for v in t if isinstance(v,(int,float)) and math.isfinite(v) and v>0)
    return statistics.median(values) if values else None

def cost_hours(cfg,records):
    if cfg["v56_action"]=="smoke": return .15
    if cfg["v56_action"]=="public": return .45
    arch=cfg["architecture"]
    rate=observed_epoch_hours(records,arch)
    if rate is None:
        rate=.22 if arch.startswith("sepformer_") else .16 if arch.startswith("cascade_") or arch=="cmgan_transfer" else .08
    # Planning estimate (not billed GPU time), with headroom for evaluation/download.
    return .12+1.35*rate*cfg["epochs"]

def promotion_value(result,records,cfg):
    score=float(result["best_validation_pesq_wb"])
    baseline=float((result.get("zero_shot") or {}).get("challenge",1.6961209534144983))
    gain=score-baseline
    rate=gain/max(cost_hours(cfg,records),.01)
    # High PESQ still matters even if an excellent pretrained starting point gains little.
    return score+.30*max(-1.,min(1.,rate))

def next_trial(records,ledger,sha,cache,epochs,remaining,job_limit,seed=42):
    """One successive-halving-like promotion per two new pilots, then seed expansion.
    Complete verified results only. Terminal failures are not automatically retried.
    """
    catalog=stage_catalog(records,epochs)
    if not catalog: return None
    current=[(st,r) for st,r in records if r.get("v56_source_sha")==sha and r.get("seed")==seed and r.get("v56_action") in ("pilot","promote")]
    by_stage={}
    for st,r in current:
        field=next((k for k in ("v52_stage","v53_stage","v54_stage") if r.get(k)),None)
        if field and r["v56_action"]=="pilot" and st.get("ref"):
            by_stage.setdefault(field+":"+r[field],[]).append((st,r))
    pilots=[];promotions=[];matched={}
    for name,p,low,high in catalog:
        cfg=trial_cfg(p,name,low,seed,cache,sha,"pilot")
        exact=next(((st,r) for st,r in by_stage.get(name,[]) if r.get("v56_key")==cfg["v56_key"]),None)
        if exact: matched[name]=exact
        if cfg["v56_key"] not in ledger:
            pilots.append((name,cfg))
        elif exact and high>low:
            st,r=exact
            cfg=trial_cfg(p,name,high,seed,cache,sha,"promote",st)
            if cfg["v56_key"] not in ledger:
                promotions.append((promotion_value(r,records,cfg),name,cfg))
    # Promote only the upper half of successful pilots. Once a pilot has been
    # promoted, it still counts in the threshold, rather than promoting losers.
    rank=sorted(matched,key=lambda n:float(matched[n][1]["best_validation_pesq_wb"]),reverse=True)
    top=set(rank[:max(1,(len(rank)+1)//2)])
    promotions.sort(key=lambda a:(-a[0],a[1]))
    promotions=[(name,cfg) for _,name,cfg in promotions if name in top]
    completed_promotions=sum(r.get("v56_action")=="promote" for _,r in current)
    prefer_promotion=len(matched)>=4 and completed_promotions<len(matched)//2
    order=(promotions+pilots if prefer_promotion else pilots+promotions)
    for name,cfg in order:
        s=smoke_cfg(cfg["architecture"],sha)
        existing=ledger.get(s["v56_key"])
        if existing and not verified_smoke(s["v56_key"],ledger): continue
        estimate=cost_hours(cfg,records)
        if estimate>job_limit or estimate+(0 if existing else cost_hours(s,records))>remaining: continue
        if not existing: return s
        return cfg
    return None

def maybe_ensemble(root,records,output,preflight=None):
    """One local ensemble attempt per distinct checkpoint shortlist; no GPU jobs."""
    if root is None: return
    import v55_ensemble as ensemble
    import submit_v5 as submission
    public_families=tuple(preflight["families"]) if preflight else ()
    folders=ensemble.select_experts(records,3,public_families)
    if len(folders)<2: return
    key=signature([submission.sha256(p/"best_model.pt") for p in folders])
    dest=output/("ensemble_"+key[:16])
    if dest.exists(): return
    args=argparse.Namespace(root=root,out=dest,run=folders,max_experts=3,gate_epochs=12,device="cpu",validation=True,public_families=public_families,cmgan_root=Path(preflight["cmgan_root"]) if preflight else None)
    try:
        ensemble.run(args)
    except Exception as exc:
        dest.mkdir(parents=True,exist_ok=True)
        core.json_write(dest/"failure.json",{"error":repr(exc),"key":key})
        print("V56 ensemble unavailable:",str(exc),flush=True)

def run(args):
    external=(v4.ROOT/"src/kaggle_v4_external.py").read_text(encoding="utf-8")
    public=v4.public_with_external((v4.ROOT/"src/kaggle_v4_public.py").read_text(encoding="utf-8"),external)
    compile(external,"external.py","exec");compile(public,"public.py","exec")
    sha=source_hash(external);psha=source_hash(public)
    core.no_uncertain_prior_jobs()
    if not args.execute:
        print("DRY RUN: V5.6 unified Ralph, no GPU submission")
        print("Initial stage catalog:",json.dumps([n for n,_,_,_ in stage_catalog(v4.history(),args.epochs)]))
        return
    if (core.OUTPUT/"v57_audit_seal.json").exists(): raise RuntimeError("Independent audit sealed; no further V5.6 training/tuning")
    if not args.allow_public_assets: raise RuntimeError("Confirm instructor permission for licensed external corpora/pretrained models with --allow-public-assets")
    from v571_preflight import require_report,REPORT
    preflight=require_report(getattr(args,"preflight_report",None) or REPORT)
    if args.data_root:
        import v55_ensemble as ensemble
        from pesq import pesq  # fail before GPU if optional ensemble dependency missing
        _,rows=ensemble.load_train(args.data_root.resolve())
        ensemble.split_train(rows)
        if not (args.data_root/"validation"/"samples.csv").is_file(): raise ValueError("Missing local validation manifest")
    kaggle=shutil.which("kaggle")
    if not kaggle: raise RuntimeError("Kaggle CLI missing")
    lock=core.OUTPUT/"active.lock";lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("x",encoding="utf-8") as f: f.write(str(__import__("os").getpid()))
    tag="v4"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    run_dir=core.OUTPUT/("run_"+tag)
    started=time.monotonic()
    failures=0;rounds=0
    try:
        core.no_uncertain_prior_jobs()
        if (core.OUTPUT/"v57_audit_seal.json").exists(): raise RuntimeError("Independent audit sealed; no further V5.6 training/tuning")
        run_dir.mkdir(parents=True)
        cache=args.public_cache_ref or v4.find_public_cache()
        initial_quota=quota_hours(kaggle)
        # A local wall-clock budget and current quota are separate constraints.
        budget=args.budget_hours if args.budget_hours>0 else max(0.,initial_quota-args.reserve_hours)
        while not args.max_rounds or rounds<args.max_rounds:
            core.no_uncertain_prior_jobs()
            records=v4.history();ledger=states()
            remaining_quota=quota_hours(kaggle)
            spent=(time.monotonic()-started)/3600
            remaining=min(remaining_quota-args.reserve_hours,budget-spent)
            if remaining<=0: print("STOP budget/quota reserve",flush=True);break
            if args.data_root and rounds>0:
                maybe_ensemble(args.data_root,records,core.OUTPUT/"v56_ensembles",preflight)
                spent=(time.monotonic()-started)/3600
                remaining=min(quota_hours(kaggle)-args.reserve_hours,budget-spent)
                if remaining<=0: print("STOP budget after ensemble",flush=True);break
            cfg=None;source=external;kind="v4_external"
            public_key=key_for(psha,"public","public",2,42,cache)
            if public_key not in ledger and remaining>=.45 and args.max_job_minutes/60>=.45:
                cfg={"id":"V4_PUBLIC_ZERO_SHOT","architecture":"public","epochs":2,"seed":42,"public_cache":cache,
                     "v5_public":True,"v571_recipe":True,"selection_target":"challenge","v56_key":public_key,"v56_action":"public","v56_source_sha":psha}
                source=public;kind="v4_public"
            else:
                # Search current seed first; expand deterministically only when its
                # affordable candidate space is exhausted. No target-PESQ cutoff.
                seeds=sorted({42}|{int(r["seed"]) for _,r in records if r.get("v56_source_sha")==sha and r.get("v56_action") in ("pilot","promote")})
                for seed in seeds+[max(seeds)+1]:
                    cfg=next_trial(records,ledger,sha,cache,args.epochs,remaining,args.max_job_minutes/60,seed)
                    if cfg: break
                if cfg and cfg["v56_action"]=="smoke": kind="v4_smoke"
            if cfg is None: print("STOP no affordable eligible trial",flush=True);break
            estimate=cost_hours(cfg,records)
            # Reserve plus estimated cost must be available again immediately before push.
            current=quota_hours(kaggle)
            if current<args.reserve_hours+estimate or (time.monotonic()-started)/3600+estimate>budget:
                print("STOP budget changed before submission",flush=True);break
            cfg["v56_wait_minutes"]=args.max_job_minutes
            rounds+=1
            print("V56 PLAN",json.dumps({"round":rounds,"estimated_gpu_hours":estimate,"remaining_quota":current,"config":cfg}),flush=True)
            try:
                start=time.monotonic()
                st,r=v4.submit(kaggle,source,tag,rounds,kind,cfg)
                elapsed=(time.monotonic()-start)/3600
                after=quota_hours(kaggle)
                baseline=float((r.get("zero_shot") or {}).get("challenge",1.6961209534144983))
                gain=float(r["best_validation_pesq_wb"])-baseline if kind!="v4_smoke" else None
                core.append_history(run_dir/"v56_efficiency.jsonl",{"ref":st["ref"],"key":cfg["v56_key"],"action":cfg["v56_action"],"estimated_hours":estimate,"observed_wall_hours":elapsed,"quota_delta_hours":current-after,"development_gain":gain,"gain_per_wall_hour":gain/max(elapsed,1e-9) if gain is not None else None})
            except legacy.KaggleTerminalFailure as exc:
                failures+=1
                print("V56 terminal error",str(exc),flush=True)
                if failures>=3: print("STOP consecutive terminal errors",flush=True);break
                continue
            failures=0
            if kind!="v4_smoke":
                fresh=v4.history()
                eligible=[(a,b) for a,b in fresh if not b.get("transductive_noisy") and a.get("kind") in ("v4_external","v4_public")]
                if eligible:
                    best=max(eligible,key=lambda q:q[1]["best_validation_pesq_wb"])
                    core.json_write(run_dir/"v56_best_challenge.json",{"state":best[0],"result":best[1]})
                    print("V56 BEST DEVELOPMENT",best[1]["best_validation_pesq_wb"],flush=True)
            if args.data_root and kind=="v4_external":
                maybe_ensemble(args.data_root,v4.history(),core.OUTPUT/"v56_ensembles",preflight)
        core.json_write(run_dir/"v56_session.json",{"gpu_submissions":rounds,"elapsed_wall_hours":(time.monotonic()-started)/3600,"initial_quota":initial_quota,"budget_hours":budget,"reserve_hours":args.reserve_hours})
    finally:
        lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--allow-public-assets",action="store_true")
    p.add_argument("--epochs",type=int,default=25,choices=range(12,101),help="Maximum promoted trial epochs")
    p.add_argument("--max-rounds",type=int,default=0,help="0: no fixed GPU submission count")
    p.add_argument("--budget-hours",type=float,default=0.,help="0: initial quota minus reserve; local wall-time planning budget")
    p.add_argument("--reserve-hours",type=float,default=2.)
    p.add_argument("--max-job-minutes",type=int,default=180,help="Local wait limit; timeout leaves remote unresolved, not cancelled")
    p.add_argument("--public-cache-ref")
    p.add_argument("--preflight-report",type=Path,help="Successful V5.7.1 CPU report; defaults to runs/v57_public_preflight/report.json")
    p.add_argument("--data-root",type=Path,help="Optional local dataset for automatic CPU ensembles")
    args=p.parse_args()
    if args.max_rounds<0 or not math.isfinite(args.budget_hours) or args.budget_hours<0 or not math.isfinite(args.reserve_hours) or args.reserve_hours<2 or not 15<=args.max_job_minutes<=720: p.error("Invalid budget/round limits")
    run(args)

if __name__=="__main__": main()
