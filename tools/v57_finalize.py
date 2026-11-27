"""V5.7 select and freeze a portable champion using common PCM16 development scoring."""
from __future__ import annotations
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
import ralph_v4 as v4
import ralph_v2 as core
import submit_v5 as s
import v57_public_assets as public
import v57_offline as offline
from v51_data_audit import speaker

TOOLS=Path(__file__).resolve().parent

def write(path,obj):
    core.json_write(path,obj)

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def dependency_versions():
    out={}
    for name in ("torch","numpy","pandas","scipy","soundfile","einops","speechbrain","hyperpyyaml","torchaudio","pesq"):
        try: out[name]=version(name)
        except PackageNotFoundError: pass
    return out

def finalize_package(root):
    for name in ("submit_v5.py","v55_gating.py","v57_public_assets.py","v57_offline.py","v51_data_audit.py"):
        shutil.copy2(TOOLS/name,root/name)
    write(root/"environment.json",dependency_versions())
    (root/"README.txt").write_text(
        "Use trusted code only. Install compatible dependencies before disconnecting.\n"
        "Run: python v57_offline.py --input <noisy_directory> --output <new_output_directory>\n"
        "Input/output: mono 16 kHz, 48000 samples, output PCM16 WAV.\n"
        "No network used by supported inference adapters; Python guard is not OS isolation.\n",
        encoding="utf-8")
    write(root/"package_manifest.json",offline.package_manifest(root))
    offline.verify_package(root)

def package_single(folder,out,allow_download=False,cmgan_root=None):
    state=read(folder/"remote_state.json")
    r=v4.verify(folder,"v4_external")
    v4.verify_v56_metadata(r,state)
    family=r.get("model_family")
    if state.get("status")!="COMPLETE" or state.get("kind")!="v4_external" or r.get("evaluation_protocol")!="v4.2-fixed-dev-v2" or r.get("smoke") or r.get("transductive_noisy"):
        raise ValueError("Ineligible completed checkpoint")
    if family not in s.FAMILIES and family not in public.PUBLIC: raise ValueError("Unsupported family")
    out.mkdir(parents=True,exist_ok=False)
    m=out/"model_1";m.mkdir()
    for name in ("source.py","best_model.pt"):
        shutil.copy2(folder/name,m/name)
    spec={"weight":1.,"family":family,"source":"model_1/source.py","checkpoint":"model_1/best_model.pt",
          "source_sha256":s.sha256(m/"source.py"),"checkpoint_sha256":s.sha256(m/"best_model.pt")}
    if family in public.PUBLIC:
        meta=public.package_public(family,folder,(m/"source.py").read_text(encoding="utf-8"),m/"assets",allow_download,cmgan_root)
        spec["public_assets"]={**meta,"root":"model_1/assets"}
    write(out/"bundle.json",{"models":[spec]})
    finalize_package(out)
    return r

def package_ensemble(folder,out,train_sha):
    r=read(folder/"ensemble_report.json")
    if r.get("protocol")!="v55-train-fit-select-v1" or r.get("train_manifest_sha256")!=train_sha or r.get("bundle_sha256")!=s.sha256(folder/"bundle.json") or (folder/"failure.json").exists():
        raise ValueError("Unverified ensemble report")
    cfg=read(folder/"bundle.json")
    out.mkdir(parents=True,exist_ok=False)
    paths={}
    for m in cfg["models"]:
        for name in ("source","checkpoint"):
            paths[m[name]]=m[name+"_sha256"]
    if cfg.get("gating"): paths[cfg["gating"]["checkpoint"]]=cfg["gating"]["checkpoint_sha256"]
    for name,sha in paths.items():
        src=s.safe_path(folder,name)
        if s.sha256(src)!=sha: raise ValueError("Ensemble asset hash mismatch")
        dst=(out/name).resolve()
        if not dst.is_relative_to(out.resolve()): raise ValueError("Escaping bundle path")
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    write(out/"bundle.json",cfg)
    finalize_package(out)

def candidate_list(records,ensembles,limit):
    """Top verified development candidates; no audit data used here."""
    pool=[]
    for st,r in records:
        if st.get("kind")!="v4_external" or st.get("status")!="COMPLETE" or r.get("smoke") or r.get("transductive_noisy") or r.get("evaluation_protocol")!="v4.2-fixed-dev-v2" or not st.get("source_path"): continue
        family=r.get("model_family")
        if family not in s.FAMILIES and family not in public.PUBLIC: continue
        folder=Path(st["source_path"]).parent
        score=float(r["best_validation_pesq_wb"])
        if math.isfinite(score) and (folder/"best_model.pt").is_file(): pool.append((score,"single",folder))
    # Deduplicate checkpoints; preserve the highest prior score.
    pool.sort(key=lambda x:(-x[0],str(x[2])))
    used=set();chosen=[]
    for score,kind,folder in pool:
        key=s.sha256(folder/"best_model.pt")
        if key in used: continue
        used.add(key);chosen.append((score,kind,folder))
    # Include all eligible ensemble reports in the shortlist ranking, but
    # compare float and PCM16 prior scores only as a screening heuristic.
    for folder in ensembles:
        try:
            r=read(folder/"submission_validation.json")
            e=read(folder/"ensemble_report.json")
            score=float(r["pesq_wb"])
            if r["validation_files"]==164 and math.isfinite(score) and e.get("protocol")=="v55-train-fit-select-v1":
                chosen.append((score,"ensemble",folder))
        except (OSError,ValueError,KeyError,TypeError): continue
    chosen.sort(key=lambda x:(-x[0],str(x[2])))
    return chosen[:limit]

def offline_command(package,*args):
    subprocess.run([sys.executable,str(package/"v57_offline.py"),"--package",str(package),*map(str,args)],check=True,cwd=package)

def choose_best(results):
    ok=[r for r in results if r.get("status")=="COMPLETE" and math.isfinite(float(r["pcm16_pesq"]))]
    if not ok: raise RuntimeError("No verified offline candidate; inspect candidate report")
    return max(ok,key=lambda r:(r["pcm16_pesq"],-r["rank"]))

def finalize(args):
    core.no_uncertain_prior_jobs()
    if (core.OUTPUT/"active.lock").exists(): raise RuntimeError("Stop active Ralph before final selection")
    data=args.data_root.resolve();out=args.out.resolve()
    if out.exists() or out.is_relative_to(data): raise ValueError("Use new output outside dataset")
    val=data/"validation/samples.csv";train=data/"train/samples.csv"
    offline.read_manifest(val,164);offline.read_manifest(train,1296)
    train_sha=s.sha256(train)
    records=v4.history()
    ensemble_dirs=list((core.OUTPUT/"v56_ensembles").glob("ensemble_*"))
    candidates=candidate_list(records,ensemble_dirs,args.top)
    if not candidates: raise RuntimeError("No eligible verified V4/V5 checkpoints; run training first")
    out.mkdir(parents=True)
    results=[]
    for i,(score,kind,folder) in enumerate(candidates,1):
        slot=out/f"candidate_{i:02d}"
        package=slot/"package";report=slot/"evaluation"
        entry={"rank":i,"kind":kind,"source_run":str(folder),"prior_development_pesq":score}
        try:
            if kind=="single": package_single(folder,package,args.allow_public_download,args.cmgan_root)
            else: package_ensemble(folder,package,train_sha)
            offline_command(package,"--probe")
            offline_command(package,"--data-root",data,"--manifest",val,"--count",164,"--output",report)
            summary=read(report/"summary.json")
            if summary["count"]!=164 or summary["manifest_sha256"]!=s.sha256(val): raise ValueError("Development evaluation identity mismatch")
            entry.update(status="COMPLETE",pcm16_pesq=summary["pesq_wb"],summary=summary,package=str(package))
        except Exception as exc:
            entry.update(status="REJECTED",reason=repr(exc))
        results.append(entry)
        write(out/"candidate_report.json",results)
    winner=choose_best(results)
    shutil.copytree(Path(winner["package"]),out/"champion")
    offline.verify_package(out/"champion")
    report={"protocol":"v57-final-selection-v1","selected":winner,"candidates":results,"validation_manifest_sha256":s.sha256(val),
        "train_manifest_sha256":train_sha,"champion_manifest_sha256":s.sha256(out/"champion/package_manifest.json"),
        "selection":"highest actual PCM16 WAV PESQ among successfully evaluated shortlisted candidates",
        "limitations":["Development validation is not independent test; candidate selection biases this score.",
        "Only verified local V4/V5 checkpoint and V5.5 ensemble artifacts admitted. Older V1–V3 and public-recipe-only records are excluded.",
        "Rejected/unshortlisted candidates may be better; no hidden-test claim."]}
    write(out/"final_report.json",report)
    print("V57 CHAMPION",winner["pcm16_pesq"],out/"champion",flush=True)

def audit_disjoint(challenge,audit,manifest):
    """Exact-waveform/segment/known-speaker overlap guard, not proof of provenance."""
    rows=offline.read_manifest(manifest)
    known=set();hashes=set();ids=set()
    for split,count in (("train",1296),("validation",164)):
        for r in offline.read_manifest(challenge/split/"samples.csv",count):
            ids.add(r["segment_id"]);known.add(speaker(r))
            for k in ("noisy_path","clean_path"): hashes.add(offline.waveform_digest(s.read_wav(s.safe_path(challenge,r[k]))))
    groups=set()
    audit_hashes=set()
    for r in rows:
        group=speaker(r);groups.add(group)
        if r["segment_id"] in ids or (group!="unknown" and group in known): raise ValueError("Audit segment or known speaker overlap")
        for k in ("noisy_path","clean_path"):
            digest=offline.waveform_digest(s.read_wav(s.safe_path(audit,r[k])))
            if digest in hashes: raise ValueError("Audit waveform overlap with challenge")
            if digest in audit_hashes: raise ValueError("Duplicate audit waveform")
            audit_hashes.add(digest)
    return {"audit_files":len(rows),"known_audit_speakers":len(groups-{"unknown"}),"unknown_speaker_present":"unknown" in groups,
            "exact_overlap":0,"limitations":"Does not prove external-corpus independence or detect near duplicates; provenance must be independently verified."}

def audit(args):
    core.no_uncertain_prior_jobs()
    if (core.OUTPUT/"active.lock").exists(): raise RuntimeError("Stop active Ralph before audit")
    final=args.final.resolve();package=final/"champion";r=read(final/"final_report.json")
    offline.verify_package(package)
    if s.sha256(package/"package_manifest.json")!=r["champion_manifest_sha256"]: raise ValueError("Frozen champion identity changed")
    data=args.data_root.resolve();audit_root=args.audit_root.resolve()
    if data==audit_root or audit_root.is_relative_to(data) or data.is_relative_to(audit_root): raise ValueError("Audit root must be separate")
    if s.sha256(data/"train/samples.csv")!=r["train_manifest_sha256"] or s.sha256(data/"validation/samples.csv")!=r["validation_manifest_sha256"]: raise ValueError("Challenge manifests changed")
    manifest=args.manifest.resolve()
    if not manifest.is_relative_to(audit_root): raise ValueError("Audit manifest must belong to audit root")
    overlap=audit_disjoint(data,audit_root,manifest)
    if not args.confirm_heldout: raise ValueError("Confirm audit provenance was independently held out from all training/tuning")
    # Global one-shot guard; an interrupted audit remains sealed for manual review.
    marker=core.OUTPUT/"v57_audit_seal.json";marker.parent.mkdir(parents=True,exist_ok=True)
    payload={"status":"STARTED","champion":r["champion_manifest_sha256"],"manifest":s.sha256(manifest),"provenance":args.provenance,"overlap":overlap}
    with marker.open("x",encoding="utf-8") as f: json.dump(payload,f,indent=2)
    offline_command(package,"--data-root",audit_root,"--manifest",manifest,"--count",overlap["audit_files"],"--output",final/"independent_audit")
    payload["status"]="COMPLETE";write(marker,payload)
    print("Independent audit recorded; do not use it for model tuning",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="mode",required=True)
    a=sub.add_parser("select")
    a.add_argument("--data-root",type=Path,required=True);a.add_argument("--out",type=Path,required=True)
    a.add_argument("--top",type=int,default=5,choices=range(1,21))
    a.add_argument("--allow-public-download",action="store_true")
    a.add_argument("--cmgan-root",type=Path)
    b=sub.add_parser("audit")
    b.add_argument("--final",type=Path,required=True);b.add_argument("--data-root",type=Path,required=True)
    b.add_argument("--audit-root",type=Path,required=True);b.add_argument("--manifest",type=Path,required=True)
    b.add_argument("--confirm-heldout",action="store_true");b.add_argument("--provenance",required=True)
    args=p.parse_args()
    if args.mode=="select": finalize(args)
    else: audit(args)

if __name__=="__main__": main()
