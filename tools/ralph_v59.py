"""V5.9 bounded progressive training on verified checkpoints, local controller."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import ralph_v56 as old
import ralph_v4 as v4
import ralph_v2 as core
import ralph_loop as legacy
from v59_train_source import build

MILESTONES=(25,50,100,200)
PRIORITY=("v53_stage:sepformer-dns-progressive","v53_stage:cmgan-progressive",
          "v52_stage:complex-head","v53_stage:metricgan-progressive","v52_stage:phase-loss",
          "v53_stage:sepformer-wham-progressive","v53_stage:sepformer-dns-decoder",
          "v53_stage:cmgan-decoder")

def sha_file(p):
    with p.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()

def archived_cfg(st):
    path=Path(st["source_path"])
    lines=[x for x in path.read_text(encoding="utf-8").splitlines() if x.startswith("EXPERIMENT_CONFIG = ")]
    if len(lines)!=1: raise ValueError("Missing unique archived experiment config")
    cfg=ast.literal_eval(lines[0].split("=",1)[1].strip())
    if not isinstance(cfg,dict): raise ValueError("Invalid archived experiment config")
    return cfg

def stage_name(cfg):
    for field in ("v52_stage","v53_stage"):
        if cfg.get(field): return field+":"+cfg[field]
    return None

def eligible_stage(cfg):
    return bool(stage_name(cfg) and not any(cfg.get(k) for k in ("v54_stage","pesq_aware","self_supervised","transductive","distill","v53_anchor"))
                and cfg.get("transfer_mode")!="lora")

def catalog(records,policies):
    stages=[]
    for name,p,low,_ in old.stage_catalog(records,25):
        if not eligible_stage(p): continue
        if p["architecture"] in policies: p={**p,"sepformer_scale_policies":{p["architecture"]:policies[p["architecture"]]}}
        stages.append((name,p,low))
    return sorted(stages,key=lambda row:(PRIORITY.index(row[0]) if row[0] in PRIORITY else len(PRIORITY),row[0]))

def domains(r):
    return [float(r["domain_scores"][k]) for k in ("external_normal","external_stress")]

def verify_lineage(st,r,cfg):
    m=r.get("v59")
    if not isinstance(m,dict) or m.get("protocol")!="v59-weights-continuation-v1": raise ValueError("Missing V5.9 metadata")
    p=cfg.get("v59_parent")
    if m.get("parent")!=p or r.get("warm_ref")!=(p["ref"] if p else None): raise ValueError("V5.9 parent mismatch")
    if not eligible_stage(cfg) or not cfg.get("v59_enabled") or cfg.get("v56_action")!=r.get("v56_action"): raise ValueError("V5.9 training contract")
    n=int(r["epochs_completed"]);best=int(r["best_epoch"])
    if not 0<=best<=n<=int(cfg["epochs"]) or n<1: raise ValueError("Invalid V5.9 epoch counts")
    if m.get("attempted_epochs")!=(p["attempted_epochs"] if p else 0)+n or m.get("checkpoint_epoch")!=(p["checkpoint_epoch"] if p else 0)+best: raise ValueError("V5.9 epoch lineage mismatch")
    cp=Path(st["source_path"]).parent/"best_model.pt"
    if m.get("checkpoint_sha256")!=sha_file(cp) or m.get("optimizer_restored") is not False: raise ValueError("V5.9 checkpoint integrity failure")
    root=m.get("root_external_domains")
    expected=p["root_external_domains"] if p else ((r.get("zero_shot") or {}).get("external_domains") or domains(r))
    if not isinstance(root,list) or len(root)!=2 or root!=expected or any(not math.isfinite(float(v)) for v in root): raise ValueError("V5.9 external lineage mismatch")
    if any(a<b-.15001 for a,b in zip(domains(r),root)): raise ValueError("V5.9 root domain regression")
    initial=(r.get("zero_shot") or {}).get("challenge")
    gain=m.get("gain")
    if m.get("start_pesq")!=initial or (initial is None and gain is not None) or (initial is not None and (not isinstance(gain,(int,float)) or not math.isfinite(gain) or abs(gain-(r["best_validation_pesq_wb"]-initial))>1e-6)): raise ValueError("V5.9 gain mismatch")
    if p and (initial is None or abs(initial-p["score"])>.02 or any(abs(a-b)>.02 for a,b in zip(r["zero_shot"]["external_domains"],p["external_domains"]))): raise ValueError("V5.9 parent reproduction mismatch")
    lrs=m.get("final_lrs")
    if not isinstance(lrs,list) or not lrs or any(not math.isfinite(float(v)) or not 0<float(v)<=float(cfg["lr"])+1e-12 for v in lrs): raise ValueError("Invalid V5.9 learning rates")
    return m

def parent_record(st,r,cfg):
    if r.get("v59"):
        m=verify_lineage(st,r,cfg)
        attempted=m["attempted_epochs"];epoch=m["checkpoint_epoch"];root=m["root_external_domains"];gain=m["gain"]
    else:
        attempted=int(r["epochs_completed"]);epoch=int(r["best_epoch"])
        root=(r.get("zero_shot") or {}).get("external_domains") or domains(r)
        gain=float(r["best_validation_pesq_wb"])-float(r["zero_shot"]["challenge"]) if r.get("zero_shot") else None
    if not 0<=epoch<=attempted or attempted<1: raise ValueError("Invalid parent epoch count")
    return {"ref":st["ref"],"sha256":sha_file(Path(st["source_path"]).parent/"best_model.pt"),
            "attempted_epochs":attempted,"checkpoint_epoch":epoch,"score":float(r["best_validation_pesq_wb"]),
            "external_domains":domains(r),"root_external_domains":root},gain

def compatible_records(records,stages,base_sha,sha):
    lookup={n:p for n,p,_ in stages}
    out=[]
    for st,r in records:
        if st.get("kind")!="v4_external" or r.get("smoke") or r.get("transductive_noisy") or r.get("seed")!=42: continue
        if r.get("v56_source_sha") not in (base_sha,sha) or not st.get("source_path"): continue
        cfg=archived_cfg(st);name=stage_name(cfg)
        if name not in lookup or not eligible_stage(cfg): continue
        expected=lookup[name]
        if any(cfg.get(k)!=v for k,v in expected.items() if k not in ("v53_expected_initial_hash","v54_expected_coarse_hash","lr")): continue
        if cfg.get("sepformer_scale_policies",{})!=expected.get("sepformer_scale_policies",{}): continue
        if r.get("v56_source_sha")==sha:
            if r.get("v56_action") not in ("v59_pilot","v59_extend"): continue
            verify_lineage(st,r,cfg)
        elif r.get("v56_action")!="pilot" or cfg.get("warm_ref"): continue
        out.append((name,st,r,cfg))
    return out

def consumed_parents():
    refs=set()
    for p in core.OUTPUT.glob("run_v4*/round_*/*/remote_state.json"):
        state=json.loads(p.read_text(encoding="utf-8"))
        if state.get("v56_action")!="v59_extend": continue
        cfg=archived_cfg({"source_path":str(p.parent/"source.py")})
        if cfg.get("v59_parent"): refs.add(cfg["v59_parent"]["ref"])
    return refs

def estimate(cfg,records):
    rate=old.observed_epoch_hours(records,cfg["architecture"])
    if rate is None: rate=.22 if cfg["architecture"].startswith("sepformer_") else .16 if cfg["architecture"]=="cmgan_transfer" else .08
    # Include baseline re-evaluation, final re-evaluation, downloads and packaging.
    return (.20 if cfg.get("public_cache") else .45)+1.35*rate*int(cfg["epochs"])

def next_chunk(total,minimum,cap,maximum):
    if cap<minimum or maximum-total<minimum: return 0
    for target in sorted(set([m for m in MILESTONES if m<=maximum]+[maximum])):
        gap=target-total
        if gap<minimum: continue
        n=min(cap,gap)
        if 0<gap-n<minimum and gap-minimum>=minimum: n=min(n,gap-minimum)
        return n if n>=minimum else 0
    return 0

def make_cfg(p,name,epochs,cache,sha,parent=None,parent_cfg=None,result=None):
    cfg=old.base_cfg(p,epochs,42,cache)
    cfg["v59_enabled"]=True
    cfg["v59_parent"]=parent
    action="v59_extend" if parent else "v59_pilot"
    if parent:
        # Match the parent's exact training recipe (apart from LR/epoch/seed-preserving warm start).
        cfg={k:v for k,v in parent_cfg.items() if not k.startswith("v56_") and not k.startswith("v59_")}
        cfg.update(epochs=epochs,warm_ref=parent["ref"],public_cache=cache,v59_enabled=True,v59_parent=parent)
        previous_lr=float(parent_cfg["lr"])
        last=(result.get("v59") or {}).get("final_lrs",[])
        cfg["lr"]=max(1e-8,min(previous_lr*.5,float(last[0]) if last else previous_lr))
        # The expected hash refers to the original pretrained weights, not the warm state.
        if cfg.get("v53_stage"): cfg["v53_expected_initial_hash"]=result["v53_initial_hash"]
    identity={k:v for k,v in cfg.items() if k not in ("v53_expected_initial_hash","v54_expected_coarse_hash")}
    cfg.update(v56_key=old.key_for(sha,action,name,epochs,42,cache,parent["ref"] if parent else None,identity),
               v56_action=action,v56_source_sha=sha)
    return cfg

def smoke(arch,sha,policies):
    cfg=old.smoke_cfg(arch,sha,policies)
    cfg["v59_enabled"]=True
    cfg["v56_key"]=old.key_for(sha,"smoke",arch,2,42,None,settings={"v59":True,"policies":cfg.get("sepformer_scale_policies",{})})
    return cfg

def choose(records,ledger,consumed,stages,base_sha,sha,cache,policies,remaining,job_limit,maximum=200,chunk=20,min_pilots=3):
    prior=compatible_records(records,stages,base_sha,sha)
    names={x[0] for x in prior}
    completed_extensions=sum(r.get("v56_action")=="v59_extend" for _,_,r,_ in prior)
    limit=min(remaining,job_limit)
    def with_smoke(cfg):
        s=smoke(cfg["architecture"],sha,policies)
        seen=ledger.get(s["v56_key"])
        if seen and not old.verified_smoke(s["v56_key"],ledger): return None
        if estimate(cfg,records)>limit or estimate(cfg,records)+(0 if seen else old.cost_hours(s,records))>remaining: return None
        return cfg if seen else s
    pilots=[]
    for name,p,low in stages:
        if name in names: continue
        cfg=make_cfg(p,name,low,cache,sha)
        if cfg["v56_key"] in ledger: continue
        trial=with_smoke(cfg)
        if trial: pilots.append(trial)
    extensions=[]
    for name,st,r,cfg in prior:
        if st["ref"] in consumed: continue
        parent,gain=parent_record(st,r,cfg)
        # Do not keep spending on a failed or flat lineage; keep the prior best.
        if parent["checkpoint_epoch"]==0 or (gain is not None and gain<.005): continue
        minimum=12 if cfg.get("v53_stage") else 5
        cap=min(chunk,maximum-parent["attempted_epochs"])
        candidate=None
        while cap>=minimum:
            n=next_chunk(parent["attempted_epochs"],minimum,cap,maximum)
            if not n: break
            proposal=make_cfg(cfg,name,n,cache,sha,parent,cfg,r)
            if proposal["v56_key"] not in ledger:
                trial=with_smoke(proposal)
                if trial:
                    candidate=(proposal,trial);break
            cap=n-1
        if candidate:
            proposal,trial=candidate
            score=parent["score"]
            rate=(gain or 0.)/max(estimate(proposal,records),.01)
            extensions.append((score+.05*max(0.,min(1.,rate)),score,name,trial))
    extensions.sort(key=lambda x:(-x[0],x[2]))
    if extensions:
        # Keep extension choice in the high-performing region of completed candidates.
        top=max(float(r["best_validation_pesq_wb"]) for _,_,r,_ in prior)
        extensions=[x for x in extensions if x[1]>=top-.20]
    want_pilot=len(names)<min_pilots or completed_extensions>=2*max(1,len(names)-min_pilots+1)
    if pilots and (want_pilot or not extensions): return pilots[0]
    if extensions: return extensions[0][3]
    return pilots[0] if pilots else None

def check_public(records,policies,public_sha):
    if not any(st.get("kind")=="v4_public" and r.get("v571_recipe") and r.get("sepformer_scale_policies")==policies and
               r.get("v56_source_sha")==public_sha for st,r in records):
        raise RuntimeError("Current public zero-shot benchmark missing; run V5.6 one round first")

def run(args):
    external=(v4.ROOT/"src/kaggle_v4_external.py").read_text(encoding="utf-8")
    public=v4.public_with_external((v4.ROOT/"src/kaggle_v4_public.py").read_text(encoding="utf-8"),external)
    source=build(external);base_sha=old.source_hash(external);sha=old.source_hash(source)
    core.no_uncertain_prior_jobs()
    if not args.execute:
        print("V59 DRY RUN — no GPU submission",flush=True)
        print("Eligible stages:",json.dumps([n for n,_,_ in catalog(v4.history(),{})]),flush=True)
        print("Milestones:",MILESTONES,"max attempted epochs:",args.max_epochs,flush=True)
        return
    if not args.allow_public_assets: raise RuntimeError("Confirm instructor permission with --allow-public-assets")
    if (core.OUTPUT/"v57_audit_seal.json").exists(): raise RuntimeError("Independent audit sealed")
    from v571_preflight import require_report,REPORT
    from v572_scale_audit import require_report as require_scale
    preflight=require_report(args.preflight_report or REPORT)
    policies=require_scale(data_root=args.data_root)
    if preflight.get("sepformer_scale_policies")!=policies: raise RuntimeError("Preflight policy mismatch")
    import v55_ensemble as ens
    _,rows=ens.load_train(args.data_root.resolve());ens.split_train(rows)
    if not (args.data_root/"validation/samples.csv").is_file(): raise ValueError("Missing validation manifest")
    kaggle=shutil.which("kaggle")
    if not kaggle: raise RuntimeError("Kaggle CLI missing")
    lock=core.OUTPUT/"active.lock";lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("x",encoding="utf-8") as f: f.write(str(os.getpid()))
    tag="v4"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    folder=core.OUTPUT/("run_"+tag)
    started=time.monotonic();rounds=0;failures=0
    try:
        core.no_uncertain_prior_jobs()
        if (core.OUTPUT/"v57_audit_seal.json").exists(): raise RuntimeError("Independent audit sealed")
        folder.mkdir(parents=True)
        cache=args.public_cache_ref or v4.find_public_cache()
        initial=old.quota_hours(kaggle)
        budget=args.budget_hours or max(0.,initial-args.reserve_hours)
        while not args.max_rounds or rounds<args.max_rounds:
            core.no_uncertain_prior_jobs()
            records=v4.history();ledger=old.states()
            check_public(records,policies,old.source_hash(public))
            quota=old.quota_hours(kaggle)
            remaining=min(quota-args.reserve_hours,budget-(time.monotonic()-started)/3600)
            if remaining<=0: print("V59 STOP budget/quota reserve",flush=True);break
            stages=catalog(records,policies)
            cfg=choose(records,ledger,consumed_parents(),stages,base_sha,sha,cache,policies,remaining,args.max_job_minutes/60,args.max_epochs,args.chunk_epochs,args.min_pilots)
            if cfg is None: print("V59 STOP no affordable improving candidate",flush=True);break
            kind="v4_smoke" if cfg["v56_action"]=="smoke" else "v4_external"
            cost=old.cost_hours(cfg,records) if kind=="v4_smoke" else estimate(cfg,records)
            quota=old.quota_hours(kaggle)
            if quota<args.reserve_hours+cost or (time.monotonic()-started)/3600+cost>budget:
                print("V59 STOP budget changed before submission",flush=True);break
            rounds+=1
            cfg["v56_wait_minutes"]=args.max_job_minutes
            print("V59 PLAN "+json.dumps({"round":rounds,"estimated_gpu_hours":cost,"remaining_quota":quota,"config":cfg}),flush=True)
            begin=time.monotonic()
            try:
                st,r=v4.submit(kaggle,source,tag,rounds,kind,cfg)
                if kind!="v4_smoke":
                    verify_lineage(st,r,cfg)
                    print("V59 RESULT "+json.dumps(r["v59"]),flush=True)
                failures=0
            except legacy.KaggleTerminalFailure as exc:
                failures+=1
                print("V59 terminal error",str(exc),flush=True)
                if failures>=3: break
                continue
            core.append_history(folder/"v59_efficiency.jsonl",{"ref":st["ref"],"action":cfg["v56_action"],"estimated_hours":cost,
                "observed_wall_hours":(time.monotonic()-begin)/3600,"score":r["best_validation_pesq_wb"],"lineage":r.get("v59")})
            if kind!="v4_smoke":
                records=v4.history()
                verified=compatible_records(records,stages,base_sha,sha)
                if verified:
                    _,best_st,best_r,_=max(verified,key=lambda row:row[2]["best_validation_pesq_wb"])
                    core.json_write(folder/"v59_best.json",{"state":best_st,"result":best_r})
                # CPU ensemble only when opted in; long training gets priority.
                if args.ensemble: old.maybe_ensemble(args.data_root,records,core.OUTPUT/"v56_ensembles",preflight)
        core.json_write(folder/"v59_session.json",{"gpu_submissions":rounds,"elapsed_wall_hours":(time.monotonic()-started)/3600,"initial_quota":initial,"budget_hours":budget,"reserve_hours":args.reserve_hours})
    finally:
        lock.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--allow-public-assets",action="store_true")
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--max-epochs",type=int,default=200,choices=range(25,201),help="Maximum attempted epochs per lineage")
    p.add_argument("--chunk-epochs",type=int,default=20,choices=range(12,51))
    p.add_argument("--min-pilots",type=int,default=3,choices=range(2,9))
    p.add_argument("--max-rounds",type=int,default=0)
    p.add_argument("--budget-hours",type=float,default=0.)
    p.add_argument("--reserve-hours",type=float,default=2.)
    p.add_argument("--max-job-minutes",type=int,default=360)
    p.add_argument("--public-cache-ref")
    p.add_argument("--preflight-report",type=Path)
    p.add_argument("--ensemble",action="store_true")
    args=p.parse_args()
    if args.max_rounds<0 or not math.isfinite(args.budget_hours) or args.budget_hours<0 or not math.isfinite(args.reserve_hours) or args.reserve_hours<2 or not 15<=args.max_job_minutes<=720: p.error("Invalid budget/round limits")
    run(args)

if __name__=="__main__": main()
