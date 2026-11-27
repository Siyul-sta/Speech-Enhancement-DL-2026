"""Read-only artifact recovery from an already completed Kaggle kernel.
Never submits jobs, changes remote state, or promotes an unverified score.
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import ralph_v4
import reconcile_ralph_v2 as reconcile

ROOT=Path(__file__).resolve().parents[1]
def recover(ref,out,kaggle=None,source=None):
    if not re.fullmatch(r"[\w-]+/[\w-]+",ref): raise ValueError("Expected exact owner/notebook")
    out=Path(out).resolve()
    if out.exists(): raise FileExistsError("Choose a new recovery directory")
    kaggle=kaggle or shutil.which("kaggle")
    if not kaggle: raise RuntimeError("Kaggle CLI missing")
    status,_=reconcile.status(kaggle,ref)
    if status!="COMPLETE": raise RuntimeError("Remote state is not COMPLETE: "+str(status))
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as d:
        folder=Path(d)
        subprocess.run([kaggle,"kernels","output",ref,"-p",str(folder)],cwd=ROOT,check=True,timeout=600)
        r=json.loads((folder/"result.json").read_text(encoding="utf-8"))
        kind="v4_smoke" if r.get("smoke") else ("v4_external" if r.get("checkpoint") or (folder/"best_model.pt").exists() else "v4_public")
        r=ralph_v4.verify(folder,kind)
        if source is not None:
            source=Path(source)
            compile(source.read_text(encoding="utf-8"),str(source),"exec")
            shutil.copy2(source,folder/"source.py")
        ranked=kind!="v4_smoke" and not r.get("transductive_noisy") and r.get("evaluation_protocol")=="v4.2-fixed-dev-v2"
        meta={"ref":ref,"kind":kind,"verified":True,"eligible_protocol":ranked,
              "archived_source_supplied":source is not None,
              "challenge_pesq":r["best_validation_pesq_wb"],
              "note":"Recovered score only. Exact archived source must be independently matched to job before inference or promotion."}
        (folder/"recovery.json").write_text(json.dumps(meta,indent=2,allow_nan=False),encoding="utf-8")
        shutil.copytree(folder,out)
    print(json.dumps(meta,indent=2),flush=True)
    return meta

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref",required=True)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--source",type=Path,help="Exact previously archived job source (optional; not automatically trusted)")
    a=p.parse_args()
    recover(a.ref,a.out,source=a.source)

if __name__=="__main__": main()
