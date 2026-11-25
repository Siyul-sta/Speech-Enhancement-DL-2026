"""V4 sequential research controller; private kernels, no score cutoff."""
import argparse
import ast
import csv
import itertools
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
import ralph_v2 as core
import ralph_loop as legacy

ROOT = Path(__file__).resolve().parents[1]
DATASET = "siyul1122/speech-data-private"

def verify_rows(path, count, expected):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    scores = [float(row["pesq_wb"]) for row in rows]
    if len(scores) != count or [int(row["index"]) for row in rows] != list(range(count)):
        raise ValueError("Incomplete validation")
    if not all(math.isfinite(v) for v in scores) or not math.isfinite(float(expected)) or abs(sum(scores)/count-float(expected)) > 1e-6:
        raise ValueError("Invalid score")

def verify(folder, kind="v4_public"):
    r = json.loads((folder/"result.json").read_text(encoding="utf-8"))
    if r["validation_files"] != 164:
        raise ValueError("Expected 164 validation clips")
    verify_rows(folder/"validation_per_file.csv",164,r["best_validation_pesq_wb"])
    if kind == "v4_external":
        if r["external_validation_files"] != 64 or not (folder/"best_model.pt").is_file():
            raise ValueError("Missing external evaluation or checkpoint")
        verify_rows(folder/"external_validation.csv",64,r["external_validation_pesq_wb"])
        joint=(r["best_validation_pesq_wb"]+r["external_validation_pesq_wb"])/2
        if not math.isfinite(float(r["joint_score"])) or abs(joint-r["joint_score"])>1e-6:
            raise ValueError("Invalid joint score")
    return r

def history():
    records=[]
    for path in sorted(core.OUTPUT.glob("run_v4*/round_*/*/remote_state.json")):
        state=json.loads(path.read_text(encoding="utf-8"))
        if state.get("status")=="COMPLETE":
            try:
                records.append((state,verify(path.parent,state["kind"])))
            except (OSError,ValueError,KeyError):
                continue
    return records

def submit(kaggle, source, tag, i, kind, config=None):
    folder=core.OUTPUT/("run_"+tag)/f"round_{i:04d}"/kind
    folder.mkdir(parents=True)
    job=ROOT/"kaggle_run"/"ralph_v4"/tag/f"job_{i:04d}"
    job.mkdir(parents=True)
    title=f"Research Speech {tag} job {i}"
    ref=f"siyul1122/research-speech-{tag}-job-{i}"
    source=source.replace('DATASET_SLUG = "PRIVATE_DATASET_SLUG"',f"DATASET_SLUG = {DATASET!r}")
    if config:
        source,count=re.subn(r"^EXPERIMENT_CONFIG = .+$",lambda _: "EXPERIMENT_CONFIG = "+repr(config),source,count=1,flags=re.M)
        if count!=1: raise ValueError("Missing config")
    compile(source,"run.py","exec")
    (job/"run.py").write_text(source,encoding="utf-8")
    (folder/"source.py").write_text(source,encoding="utf-8")
    core.json_write(job/"kernel-metadata.json",{"id":ref,"title":title,"code_file":"run.py","language":"python","kernel_type":"script","is_private":"true","enable_gpu":"true","enable_internet":"true","dataset_sources":[DATASET],"competition_sources":[],"kernel_sources":[config["warm_ref"]] if config and config.get("warm_ref") else []})
    state=folder/"remote_state.json"
    core.json_write(state,{"ref":ref,"status":"SUBMIT_PENDING","kind":kind})
    try:
        core.poll_adaptive_kernel(kaggle,job,ref,folder,180*60)
    except legacy.KaggleTerminalFailure:
        core.json_write(state,{"ref":ref,"status":"ERROR_TERMINAL","kind":kind})
        raise
    r=verify(folder,kind)
    core.json_write(state,{"ref":ref,"status":"COMPLETE","kind":kind})
    print(json.dumps(r,indent=2),flush=True)
    return {"ref":ref,"kind":kind,"status":"COMPLETE"},r

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--execute",action="store_true")
    p.add_argument("--loop",action="store_true",help="Continue mixed-corpus training and warm-start search until quota/safety guard")
    p.add_argument("--epochs",type=int,default=15,choices=range(2,101))
    args=p.parse_args()
    sources={k:(ROOT/"src"/name).read_text(encoding="utf-8") for k,name in (("v4_public","kaggle_v4_public.py"),("v4_external","kaggle_v4_external.py"))}
    for k,s in sources.items(): compile(s,k,"exec")
    core.no_uncertain_prior_jobs()
    print("V4: public benchmark; optional continuous external-data and warm-start research; no PESQ cutoff")
    if not args.execute:
        print("DRY RUN; no submission")
        return
    kaggle=shutil.which("kaggle")
    if not kaggle: raise RuntimeError("Kaggle CLI missing")
    lock=core.OUTPUT/"active.lock"
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("x",encoding="utf-8") as f: f.write(str(os.getpid()))
    try:
        core.no_uncertain_prior_jobs()
        tag="v4"+datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        records=history()
        i=0
        if not any(st["kind"]=="v4_public" for st,r in records) or not args.loop:
            legacy.require_quota(kaggle,2)
            i+=1
            records.append(submit(kaggle,sources["v4_public"],tag,i,"v4_public"))
        if not args.loop: return
        for n in itertools.count(sum(st["kind"]=="v4_external" for st,r in records)):
            try: legacy.require_quota(kaggle,2)
            except legacy.QuotaGuardReached as exc:
                print("STOP",exc,flush=True); break
            external=[(st,r) for st,r in records if st["kind"]=="v4_external"]
            best=max(external,key=lambda pair:pair[1]["joint_score"],default=None)
            # Alternate scratch baselines and warm-start learning rates/data ratios.
            warm=best[0]["ref"] if best and n%4!=0 else None
            cfg={"id":"V3_RESIDUAL_PHASE","epochs":args.epochs,"batch_size":4,"patience":7,"seed":42,"warm_ref":warm,
                 "lr":([0.0001,0.0002,0.00005][n%3] if warm else 0.00035),"external_size":[512,1024,2048][n%3]}
            i+=1
            records.append(submit(kaggle,sources["v4_external"],tag,i,"v4_external",cfg))
            ranked=max((r for st,r in records if st["kind"]=="v4_external"),key=lambda r:r["joint_score"])
            print("BEST JOINT",ranked["joint_score"],"challenge",ranked["best_validation_pesq_wb"],"external",ranked["external_validation_pesq_wb"],flush=True)
    finally:
        lock.unlink(missing_ok=True)

if __name__=="__main__":
    main()
