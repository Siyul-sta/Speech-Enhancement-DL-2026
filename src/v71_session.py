"""V7.1 remote session wrapper. All candidates use the frozen session parent.
Runs isolated child processes on the SAME allocated Kaggle GPU session.
Only completed stage outputs are eligible. No child kernel submissions.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

def atomic_json(path,value):
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,allow_nan=False,indent=2),encoding="utf-8");tmp.replace(path)

def canonical(v):
    return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False)

def digest(v):
    return hashlib.sha256(canonical(v).encode()).hexdigest()

def stage_source(base,cfg,work,dataset):
    import re
    for name,val in (("EXPERIMENT_CONFIG",cfg),("DATASET_SLUG",dataset)):
        base,n=re.subn(r"^"+name+r" = .+$",lambda _:name+" = "+repr(val),base,count=1,flags=re.M)
        if n!=1:raise ValueError("Missing "+name)
    needle='WORK = Path("/kaggle/working") if Path("/kaggle").exists() else Path("local_test_output")'
    if base.count(needle)!=1:raise ValueError("WORK source contract")
    base=base.replace(needle,"WORK = Path("+repr(str(work))+")",1)
    needle='mp_root=Path(mp_root) if mp_root else WORK/"v6_mp_assets"'
    if base.count(needle)!=1:raise ValueError("MP asset cache contract")
    base=base.replace(needle,'mp_root=Path(mp_root) if mp_root else Path("/tmp/v71_mp_assets")',1)
    compile(base,"v71_stage.py","exec")
    return base

def run_child(base,cfg,work,dataset,timeout):
    work.mkdir(parents=True,exist_ok=False)
    script=Path("/tmp")/("v71_"+work.parent.name+"_"+work.name+".py")
    script.write_text(stage_source(base,cfg,work,dataset),encoding="utf-8")
    p=subprocess.Popen([sys.executable,"-u",str(script)],start_new_session=True)
    try:
        code=p.wait(timeout=max(1,timeout))
    except (subprocess.TimeoutExpired,KeyboardInterrupt):
        os.killpg(p.pid,signal.SIGKILL);p.wait();raise
    finally:script.unlink(missing_ok=True)
    if code:raise RuntimeError("Stage exit "+str(code))
    r=json.loads((work/"result.json").read_text(encoding="utf-8"))
    cp=work/"best_model.pt"
    with cp.open("rb") as f:h=hashlib.file_digest(f,"sha256").hexdigest()
    if h!=r["v6"]["checkpoint_sha256"] or r["v56_key"]!=cfg["v56_key"] or r["epochs_completed"]<1 or r["v7"]["recipe"]!=cfg["v7_recipe"]:
        raise ValueError("Stage identity or checkpoint invalid")
    if cfg["smoke"] and r["epochs_completed"]!=cfg["epochs"]:raise ValueError("Incomplete smoke")
    # Exactly ONE best_model.pt in the remote output mount (the session winner).
    cp.replace(work/"checkpoint.pt")
    return r

def run_session(base,plan,root=Path("/kaggle/working"),runner=run_child,clock=time.monotonic):
    if hashlib.sha256(base.encode()).hexdigest()!=plan["source"]:raise ValueError("Base source mismatch")
    if not 1<=len(plan["candidates"])<=4 or not 0<plan["hours"]<=5:raise ValueError("Invalid bounded session")
    root.mkdir(parents=True,exist_ok=True);trials=root/"v71_trials";trials.mkdir(exist_ok=False)
    started=clock();deadline=started+plan["hours"]*3600
    records=[];winner=None;best=None;pending=list(enumerate(plan["candidates"]));observed={}
    def save(reason):
        atomic_json(root/"v71_report.json",{"protocol":"v71-session-v1","plan_sha":digest(plan),"records":records,"winner":winner,"reason":reason})
    save("running")
    while pending:
        # Reorder remaining same-parent challengers by actual family gain in this session.
        pending.sort(key=lambda pair:-(pair[1]["priority"]+observed.get(pair[1]["family"],0.)))
        i,item=pending.pop(0)
        if deadline-clock()<item["estimate_hours"]*3600+60:break
        folder=trials/f"trial_{i:02d}"
        record={"index":i,"smoke":"reused" if item["reuse_smoke"] else "not_started","train":"not_started"}
        records.append(record)
        try:
            if not item["reuse_smoke"]:
                print("V71_STAGE "+json.dumps({"index":i,"stage":"smoke","recipe":item["config"]["v7_recipe"]}),flush=True)
                runner(base,item["smoke"],folder/"smoke",plan["dataset"],deadline-clock()-60)
                record["smoke"]="complete";save("running")
            if deadline-clock()<item["train_hours"]*3600+60:break
            print("V71_STAGE "+json.dumps({"index":i,"stage":"train","recipe":item["config"]["v7_recipe"]}),flush=True)
            r=runner(base,item["config"],folder/"train",plan["dataset"],deadline-clock()-60)
            score=float(r["best_validation_pesq_wb"])
            if not math.isfinite(score):raise ValueError("Nonfinite score")
            record.update(train="complete",score=score)
            gain=max(0.,score-float(item["config"]["v6_parent"]["score"]))
            observed[item["family"]]=min(.1,gain/max(.1,item["train_hours"]))
            if best is None or score>best:
                winner=i;best=score
            print("V71_SCORE "+json.dumps({"index":i,"recipe":item["config"]["v7_recipe"],"pesq":score}),flush=True)
            save("running")
        except Exception as exc:
            # Preserve completed trials; stop rather than consume GPU after a failure.
            record["error"]=type(exc).__name__+": "+str(exc)[:500];save("stage_error");break
    reason="stage_error" if records and records[-1].get("error") else "finished" if not pending and all(r["train"]=="complete" for r in records) else "budget"
    # A failed child may have written a partial checkpoint; never expose it as a mountable parent.
    for cp in trials.rglob("best_model.pt"):
        cp.replace(cp.with_name("unverified_checkpoint.pt"))
    if winner is not None:
        shutil.copy2(trials/f"trial_{winner:02d}"/"train/checkpoint.pt",root/"best_model.pt")
    save(reason)
    print("V71_SESSION_END "+json.dumps({"winner":winner,"score":best,"reason":reason}),flush=True)
