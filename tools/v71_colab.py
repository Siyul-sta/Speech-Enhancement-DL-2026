"""Private local handoff for the existing V7.1 plan. No Kaggle submissions.
Export only train/validation audio, verified parent and reviewed source.
Verify returned completed stages without importing a fake Kaggle reference.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import sys
import tempfile
import zipfile
import ralph_v71 as v
import v71_colab_worker as worker

def json_read(p):return json.loads(p.read_text(encoding="utf-8"))

def input_files(root):
    files=set()
    for split,count in (("train",1296),("validation",164)):
        p=root/split/"samples.csv";files.add(p)
        with p.open(encoding="utf-8-sig",newline="") as f:rows=list(csv.DictReader(f))
        if len(rows)!=count:raise ValueError("Unexpected "+split+" count")
        for row in rows:
            for col in ("noisy_path","clean_path"):
                q=(root/row[col]).resolve()
                if not q.is_relative_to(root) or not q.is_file():raise ValueError("Unsafe or missing audio "+str(q))
                if q.suffix.lower()!=".wav":raise ValueError("Expected WAV only")
                files.add(q)
    return sorted(files)

def find_plan(ref):
    hits=[]
    for p in v.core.OUTPUT.glob("run_v4*/round_*/*/remote_state.json"):
        st=json_read(p)
        if st.get("ref")==ref and st.get("v71_session"):hits.append(p.parent)
    if len(hits)!=1:raise ValueError("Expected exactly one matching V7.1 session")
    return hits[0]

def export(data,ref,out):
    if (v.core.OUTPUT/"v57_audit_seal.json").exists():raise ValueError("Independent audit sealed")
    data=data.resolve();out=out.resolve()
    if out.exists() or out.is_relative_to(data):raise ValueError("Use new ZIP outside data")
    folder=find_plan(ref);plan=json_read(folder/"v71_plan.json")
    base=(folder/"v71_base.txt").read_text(encoding="utf-8")
    if v.search.digest(base)!=plan["source"] or v.digest(plan)!=json_read(folder/"remote_state.json")["plan_sha"]:
        raise ValueError("Session plan/source mismatch")
    e=v.v7.verified_entries()
    parents=[x for x in e.values() if x[1] and x[0]["ref"]==plan["parent"]["ref"]]
    if len(parents)!=1 or v.ctl.parent_info(parents[0][0],parents[0][1])!=plan["parent"]:raise ValueError("Verified parent not found")
    parent=parents[0][3]
    for item in plan["candidates"]:
        if item["config"].get("public_cache") or item["smoke"].get("public_cache"):raise ValueError("Export currently requires no Kaggle cache")
        if item["config"]["v6_parent"]!=plan["parent"]:raise ValueError("Candidate parent mismatch")
        item["reuse_smoke"]=False
    files={p.relative_to(data).as_posix():p for p in input_files(data)}
    payload={"data/"+k:p for k,p in files.items()}
    for n in ("source.py","best_model.pt","result.json","remote_state.json","validation_per_file.csv","external_validation.csv","zero_shot_validation.csv","zero_shot_external.csv"):
        payload["parent/"+n]=parent/n
    for n in ("v71_session.py","v71_colab_worker.py"):payload[n]=v.ctl.ROOT/"src"/n
    with tempfile.TemporaryDirectory() as t:
        tmp=Path(t);(tmp/"v71_base.txt").write_text(base,encoding="utf-8")
        v.core.json_write(tmp/"v71_plan.json",plan)
        payload.update({"v71_base.txt":tmp/"v71_base.txt","v71_plan.json":tmp/"v71_plan.json"})
        manifest={"protocol":"v71-colab-handoff-v1","plan_sha":v.digest(plan),"original_ref":ref,"files":{k:v.ctl.sha(p) for k,p in payload.items()}}
        out.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(out,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for name,p in payload.items():z.write(p,name)
            z.writestr("handoff_manifest.json",json.dumps(manifest,indent=2))
    print("COLAB EXPORT "+json.dumps({"zip":str(out),"bytes":out.stat().st_size,"files":len(payload),"parent_pesq":plan["parent"]["score"],"candidates":[i["config"]["v7_recipe"] for i in plan["candidates"]]}),flush=True)

def extract(zpath,out):
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            q=(out/m.filename).resolve()
            if not q.is_relative_to(out.resolve()):raise ValueError("Unsafe ZIP path")
        z.extractall(out)

def verify(handoff,returned,out,data,package):
    data=data.resolve();out=out.resolve()
    if out.exists() or out.is_relative_to(data) or out.is_relative_to(v.core.OUTPUT.resolve()):raise ValueError("Use fresh output outside data/Kaggle ledger")
    with zipfile.ZipFile(handoff) as z:
        manifest=json.loads(z.read("handoff_manifest.json"));plan=json.loads(z.read("v71_plan.json"));base=z.read("v71_base.txt").decode().replace("\r\n","\n")
        if v.digest(plan)!=manifest["plan_sha"] or v.search.digest(base)!=plan["source"]:raise ValueError("Original handoff mismatch")
        for name,h in manifest["files"].items():
            if name.startswith("data/"):
                if v.ctl.sha(data/name[5:])!=h:raise ValueError("Local data changed")
    out.mkdir(parents=True)
    session=out/"session"
    if returned.is_dir():shutil.copytree(returned,session)
    else:session.mkdir();extract(returned,session)
    if json_read(session/"handoff_manifest.json")!=manifest:raise ValueError("Wrong Colab handoff")
    runtime=json_read(session/"v71_plan.json");env=json_read(session/"colab_environment.json")
    if env["handoff_plan_sha"]!=manifest["plan_sha"] or env["runtime_plan_sha"]!=v.digest(runtime):raise ValueError("Colab runtime plan mismatch")
    if {**runtime,"hours":plan["hours"]}!=plan or not 0<runtime["hours"]<=5 or (session/"v71_base.txt").read_text(encoding="utf-8")!=base:raise ValueError("Modified plan/source")
    report=json_read(session/"v71_report.json")
    if report["plan_sha"]!=v.digest(runtime):raise ValueError("Result plan mismatch")
    # On interruption only completed records with complete evidence can be recovered.
    for row in report["records"]:
        i=row["index"];item=runtime["candidates"][i]
        for stage in ("smoke","train"):
            if row[stage]!="complete":continue
            cfg=item["smoke" if stage=="smoke" else "config"]
            p=session/"v71_trials"/f"trial_{i:02d}"/stage
            remote=str(PurePosixPath(env["output_root"])/"v71_trials"/f"trial_{i:02d}"/stage)
            expected=worker.colab_source(base,cfg,remote,runtime["dataset"],env["data_root"],env["parent_path"],env["cache"])
            if (p/"executed_source.py").read_text(encoding="utf-8")!=expected:raise ValueError("Colab executed source mismatch")
    completed=[r for r in report["records"] if r["train"]=="complete"]
    if report["reason"]=="running":
        report["reason"]="budget";report["winner"]=max(completed,key=lambda r:r["score"])["index"] if completed else None
        if completed:shutil.copy2(session/"v71_trials"/f"trial_{report['winner']:02d}"/"train/checkpoint.pt",session/"best_model.pt")
        v.core.json_write(session/"v71_report.json",report)
    # This ref is local provenance only; never put it into the Kaggle ledger.
    ref="colab/"+manifest["plan_sha"][:24]
    rows,winner=v.verify_stages(session,ref)
    result={"protocol":"v71-colab-verified-v1","parent":plan["parent"],"verified_stages":len(rows),"reason":report["reason"],
            "candidates":[{"recipe":c["v7_recipe"],"pesq":r["best_validation_pesq_wb"]} for _,r,c,_ in rows if not c["smoke"]],
            "winner":None if winner is None else {"recipe":winner[3]["v7_recipe"],"pesq":winner[2]["best_validation_pesq_wb"]}}
    v.core.json_write(out/"verified.json",result);print("COLAB VERIFIED "+json.dumps(result),flush=True)
    if package:
        from v6_preflight import ASSETS
        import v6_finalize as fin
        import v6_refine_analysis as research
        # Replay the existing parent and the winning Colab candidate as PCM16.
        candidates=[]
        p=out/"parent";p.mkdir()
        with zipfile.ZipFile(handoff) as z:
            for n,h in manifest["files"].items():
                if n.startswith("parent/"):
                    q=p/Path(n).name;q.write_bytes(z.read(n))
                    if v.ctl.sha(q)!=h:raise ValueError("Parent evidence changed")
        st=json_read(p/"remote_state.json");r,c=v.ctl.verify(p,st)
        if v.ctl.parent_info(st,r)!=plan["parent"]:raise ValueError("Parent lineage mismatch")
        candidates.append(("parent",r,p))
        if winner:
            _,st,r,c,p=winner
            shutil.copy2(p/"checkpoint.pt",p/"best_model.pt")
            v.core.json_write(p/"remote_state.json",st);candidates.append(("colab",r,p))
        scores=[]
        for name,r,p in candidates:
            root=out/("package_"+name);root.mkdir()
            spec=fin.package_expert(root,0,(r["best_validation_pesq_wb"],"v6",r["model_family"],p),ASSETS,None,v.ctl.sha(data/"train/samples.csv"),True)
            fin.seal_package(root,{"protocol":"v6-offline-v1","experts":[spec],"alignment":"none","weights":[1.]})
            fin.invoke(root,"--probe");replay=out/("replay_"+name)
            fin.invoke(root,"--data-root",data,"--manifest",data/"validation/samples.csv","--count",164,"--output",replay)
            summary=json_read(replay/"summary.json")
            if summary["count"]!=164 or summary["manifest_sha256"]!=v.ctl.sha(data/"validation/samples.csv"):raise ValueError("Incomplete PCM16 replay")
            scores.append((float(summary["pesq_wb"]),name))
        best=max(scores,key=lambda x:x[0]);v.core.json_write(out/"package_selection.json",{"scores":dict((n,s) for s,n in scores),"winner":best[1],"pesq":best[0]})
        print("COLAB PCM16 WINNER "+json.dumps({"package":str(out/("package_"+best[1])),"pesq":best[0]}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="cmd",required=True)
    a=sub.add_parser("export");a.add_argument("--data-root",type=Path,required=True);a.add_argument("--ref",required=True);a.add_argument("--out",type=Path,required=True)
    a=sub.add_parser("verify");a.add_argument("--data-root",type=Path,required=True);a.add_argument("--handoff",type=Path,required=True);a.add_argument("--returned",type=Path,required=True);a.add_argument("--out",type=Path,required=True);a.add_argument("--package",action="store_true")
    a=p.parse_args()
    if a.cmd=="export":export(a.data_root,a.ref,a.out)
    else:verify(a.handoff,a.returned,a.out,a.data_root,a.package)
if __name__=="__main__":main()
