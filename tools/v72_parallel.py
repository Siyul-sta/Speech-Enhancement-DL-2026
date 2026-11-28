"""V7.2 parallel lane reservation and common offline PCM16 comparison.
Colab output stays separate from the Kaggle ledger; no invented remote refs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import zipfile
import ralph_v71 as v

def reservation(handoff,data,base):
    with zipfile.ZipFile(handoff) as z:
        m=json.loads(z.read("handoff_manifest.json"))
        p=json.loads(z.read("v71_plan.json"))
        archived=z.read("v71_base.txt").decode().replace("\r\n","\n")
    if m["protocol"]!="v71-colab-handoff-v1" or v.digest(p)!=m["plan_sha"] or v.search.digest(archived)!=p["source"] or archived!=base:
        raise ValueError("Colab handoff source/plan mismatch")
    if not 1<=len(p["candidates"])<=4:raise ValueError("Invalid reservation")
    for split in ("train","validation"):
        name=split+"/samples.csv"
        if v.ctl.sha(data/name)!=m["files"]["data/"+name]:raise ValueError("Different dataset manifest")
    recipes=set();keys=set();families={"continue"}
    for item in p["candidates"]:
        c=item["config"];sm=item["smoke"];name=c["v7_recipe"]
        if c["v6_parent"]!=p["parent"] or c["v56_source_sha"]!=p["source"]:raise ValueError("Invalid reserved candidate")
        if name not in v.search.BY_NAME:raise ValueError("Unknown reserved recipe")
        recipes.add(name);keys.update((c["v56_key"],sm["v56_key"]));families.add(v.search.BY_NAME[name][0])
    return {"plan_sha":m["plan_sha"],"recipes":sorted(recipes),"keys":sorted(keys),"excluded_families":sorted(families)}

def require_sub_pending(sub):
    # Never resume the old duplicate MAIN submission as part of the SUB lane.
    for path in v.core.OUTPUT.glob("run_*/round_*/**/remote_state.json"):
        st=json.loads(path.read_text(encoding="utf-8"))
        if st["status"] in ("COMPLETE","ERROR_TERMINAL"):continue
        if not st.get("v71_session"):raise RuntimeError("Resolve prior Kaggle job: "+st["ref"])
        p=json.loads((path.parent/"v71_plan.json").read_text(encoding="utf-8"))
        if p.get("parallel_sub")!=sub["plan_sha"] or v.digest(p)!=st["plan_sha"]:
            raise RuntimeError("Cancel/reconcile previous MAIN Kaggle job first: "+st["ref"])
        for item in p["candidates"]:
            c=item["config"]
            if c["v7_recipe"] in sub["recipes"] or v.search.BY_NAME[c["v7_recipe"]][0] in sub["excluded_families"] or c["v56_key"] in sub["keys"]:
                raise RuntimeError("Overlapping pending job: "+st["ref"])

def compare(data,handoff,returned,out,top):
    import v71_colab as colab
    import v6_finalize as fin
    from v6_preflight import ASSETS
    data=data.resolve();out=out.resolve()
    if out.exists() or out.is_relative_to(data) or out.is_relative_to(v.core.OUTPUT.resolve()):raise ValueError("Use fresh output outside data/history")
    if (v.core.OUTPUT/"v57_audit_seal.json").exists():raise ValueError("Independent audit sealed")
    out.mkdir(parents=True)
    # Verify Colab lineage and replay parent + Colab winner, even if the session ended early.
    colab.verify(handoff,returned,out/"colab",data,True)
    c=json.loads((out/"colab/package_selection.json").read_text(encoding="utf-8"))
    rows=[{"lane":"parent" if n=="parent" else "colab","pesq":s,"package":str(out/"colab"/("package_"+n))} for n,s in c["scores"].items()]
    # Verified canonical Kaggle candidates only. No remote submission and no ledger writes.
    entries=[e for e in v.v7.verified_entries().values() if e[1] and not e[2].get("smoke") and (e[2].get("v6_base_family") or e[2]["architecture"])=="mp_dns"]
    entries.sort(key=lambda e:-e[1]["best_validation_pesq_wb"])
    seen=set();n=0
    with zipfile.ZipFile(handoff) as z:seen.add(json.loads(z.read("v71_plan.json"))["parent"]["sha256"])
    for st,r,cfg,folder in entries:
        h=r["v6"]["checkpoint_sha256"]
        if h in seen:continue
        seen.add(h)
        if n>=top:break
        root=out/f"package_kaggle_{n:02d}";root.mkdir()
        spec=fin.package_expert(root,0,(r["best_validation_pesq_wb"],"v6",r["model_family"],folder),ASSETS,None,v.ctl.sha(data/"train/samples.csv"),True)
        fin.seal_package(root,{"protocol":"v6-offline-v1","experts":[spec],"alignment":"none","weights":[1.]})
        fin.invoke(root,"--probe")
        replay=out/f"replay_kaggle_{n:02d}"
        fin.invoke(root,"--data-root",data,"--manifest",data/"validation/samples.csv","--count",164,"--output",replay)
        s=json.loads((replay/"summary.json").read_text(encoding="utf-8"))
        if s["count"]!=164 or s["manifest_sha256"]!=v.ctl.sha(data/"validation/samples.csv"):raise ValueError("Incomplete replay")
        rows.append({"lane":"kaggle","ref":st["ref"],"pesq":float(s["pesq_wb"]),"package":str(root)});n+=1
    import math
    if not all(math.isfinite(float(r["pesq"])) for r in rows):raise ValueError("Nonfinite PESQ")
    winner=max(rows,key=lambda r:r["pesq"])
    result={"protocol":"v72-parallel-pcm16-v1","count":164,"validation_manifest_sha256":v.ctl.sha(data/"validation/samples.csv"),"candidates":rows,"winner":winner}
    v.core.json_write(out/"parallel_report.json",result);print("V72 WINNER "+json.dumps(winner),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root",type=Path,required=True);p.add_argument("--handoff",type=Path,required=True)
    p.add_argument("--returned",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--top",type=int,choices=range(1,6),default=3)
    a=p.parse_args();compare(a.data_root,a.handoff,a.returned,a.out,a.top)
