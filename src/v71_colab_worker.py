"""Run a frozen V7.1 plan on a Colab GPU, without Kaggle API credentials.
This file and v71_session.py are embedded in the private local handoff ZIP.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import v71_session as session

def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def replace(code,old,new):
    if code.count(old)!=1:raise ValueError("Source contract changed: "+old[:100])
    return code.replace(old,new,1)

def replace_function(code,name,body):
    import ast
    n=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=code.splitlines(keepends=True)
    return "".join(lines[:n.lineno-1])+body+"\n"+"".join(lines[n.end_lineno:])

def colab_source(base,cfg,work,dataset,data,parent,cache):
    code=session.stage_source(base,cfg,work,dataset)
    code=replace_function(code,"locate_data_root","def locate_data_root():\n    return Path("+repr(str(data))+")")
    code=replace_function(code,"v6_parent_path",
        "def v6_parent_path(parent):\n    p=Path("+repr(str(parent))+")\n    if v6_sha(p)!=parent['sha256']:raise ValueError('Colab parent checkpoint hash mismatch')\n    return p")
    code=replace(code,'if Path("/kaggle").exists() and device.type!="cuda": raise RuntimeError("GPU required")',
        'if device.type!="cuda": raise RuntimeError("Colab GPU required")')
    code=replace(code,'(Path("/tmp/ralph_v4_external") if Path("/kaggle").exists() else WORK / "external")',
        'Path('+repr(str(cache))+')')
    code=replace(code,'workers=min(2,max(0,(os.cpu_count() or 1)-1)) if Path("/kaggle").exists() else 0',
        'workers=min(2,max(0,(os.cpu_count() or 1)-1))')
    compile(code,"colab_stage.py","exec");return code

def run(bundle,out,hours=None):
    import torch
    if not torch.cuda.is_available():raise RuntimeError("Select a GPU runtime in Colab first")
    bundle=bundle.resolve();out=out.resolve()
    if out.exists() or out.is_relative_to(bundle):raise ValueError("Use a fresh Drive output directory outside bundle")
    manifest=json.loads((bundle/"handoff_manifest.json").read_text(encoding="utf-8"))
    for name,h in manifest["files"].items():
        p=(bundle/name).resolve()
        if not p.is_relative_to(bundle) or not p.is_file() or sha(p)!=h:raise ValueError("Handoff file mismatch: "+name)
    plan=json.loads((bundle/"v71_plan.json").read_text(encoding="utf-8"))
    base=(bundle/"v71_base.txt").read_text(encoding="utf-8")
    if session.digest(plan)!=manifest["plan_sha"]:raise ValueError("Plan mismatch")
    if any(i["config"].get("public_cache") or i["smoke"].get("public_cache") for i in plan["candidates"]):raise ValueError("Kaggle public cache cannot be mounted in Colab")
    if any(i["reuse_smoke"] for i in plan["candidates"]):raise ValueError("Export must request GPU smoke on Colab")
    if hours is not None:
        if not 0<hours<=5:raise ValueError("Session hours must be between 0 and 5")
        plan["hours"]=hours
    parent=bundle/"parent"/"best_model.pt";data=bundle/"data"
    if sha(parent)!=plan["parent"]["sha256"]:raise ValueError("Parent checkpoint mismatch")
    out.mkdir(parents=True)
    shutil.copy2(bundle/"handoff_manifest.json",out/"handoff_manifest.json")
    shutil.copy2(bundle/"v71_base.txt",out/"v71_base.txt")
    session.atomic_json(out/"v71_plan.json",plan)
    session.atomic_json(out/"colab_environment.json",{"device":torch.cuda.get_device_name(0),"torch":str(torch.__version__),"cuda":torch.version.cuda,
        "handoff_plan_sha":manifest["plan_sha"],"runtime_plan_sha":session.digest(plan),"data_root":str(data),"parent_path":str(parent),"output_root":str(out),"cache":"/content/v71_public_cache"})
    original=session.stage_source
    def adapted(code,cfg,work,dataset):
        # Avoid recursion: colab_source invokes original stage-source function.
        session.stage_source=original
        try:return colab_source(code,cfg,work,dataset,data,parent,Path("/content/v71_public_cache"))
        finally:session.stage_source=adapted
    def runner(code,cfg,work,dataset,timeout):
        # Save exact executed source after successful child; use original result schema.
        r=session.run_child(code,cfg,work,dataset,timeout)
        (work/"executed_source.py").write_text(adapted(code,cfg,work,dataset),encoding="utf-8")
        return r
    session.stage_source=adapted
    try:session.run_session(base,plan,root=out,runner=runner)
    finally:session.stage_source=original

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--bundle",required=True,type=Path);p.add_argument("--out",required=True,type=Path);p.add_argument("--hours",type=float)
    a=p.parse_args();run(a.bundle,a.out,a.hours)
