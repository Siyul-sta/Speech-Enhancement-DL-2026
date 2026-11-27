"""Prepare a portable native-checkpoint bundle from verified local V4/V5 runs."""
import argparse
import json
import shutil
from pathlib import Path
from submit_v5 import sha256,FAMILIES

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,nargs="+",required=True,help="Completed run folders with result.json, source.py, best_model.pt")
    p.add_argument("--weights",nargs="+",type=float,help="Fixed global weights; default uniform")
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if not 1<=len(args.run)<=5: p.error("1–5 native checkpoints required")
    weights=args.weights or [1/len(args.run)]*len(args.run)
    if len(weights)!=len(args.run) or any(not 0<w<=1 for w in weights) or abs(sum(weights)-1)>1e-6: p.error("Weights must be positive and sum to one")
    if args.out.exists(): p.error("Use a new output directory")
    entries=[]
    for folder,w in zip(args.run,weights):
        r=json.loads((folder/"result.json").read_text(encoding="utf-8"))
        family=r.get("model_family","residual")
        if family not in FAMILIES or r.get("smoke") or r.get("transductive_noisy") or r.get("validation_files")!=164:
            p.error("Not an eligible native checkpoint: "+str(folder))
        for file in ("source.py","best_model.pt"):
            if not (folder/file).is_file(): p.error("Missing "+file)
        entries.append((folder,w,r))
    args.out.mkdir(parents=True)
    config={"models":[],"selection":"fixed global weights; validate bundle before submission"}
    for i,(folder,w,r) in enumerate(entries):
        sub=args.out/f"model_{i+1}"
        sub.mkdir()
        for file in ("source.py","best_model.pt","result.json"):
            shutil.copy2(folder/file,sub/file)
        config["models"].append({"weight":w,"source":f"model_{i+1}/source.py","checkpoint":f"model_{i+1}/best_model.pt",
                                 "source_sha256":sha256(sub/"source.py"),"checkpoint_sha256":sha256(sub/"best_model.pt"),
                                 "individual_validation_pesq":r["best_validation_pesq_wb"]})
    (args.out/"bundle.json").write_text(json.dumps(config,indent=2,allow_nan=False),encoding="utf-8")
    print(args.out/"bundle.json")
if __name__=="__main__": main()
