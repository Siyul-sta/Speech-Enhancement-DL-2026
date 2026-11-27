"""Offline V5.7 package runner. Trusted code only; Python network/subprocess guard."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import submit_v5 as s

def block_network():
    # Not an OS sandbox: protects the trusted adapters from accidental Python downloads.
    def hook(event,args):
        if event.startswith("socket.") or event in ("subprocess.Popen","os.system","os.exec","os.posix_spawn"):
            raise RuntimeError("Offline inference forbids network/subprocess: "+event)
    sys.addaudithook(hook)

def package_manifest(root):
    files={}
    for p in sorted(root.rglob("*")):
        if p.is_symlink(): raise ValueError("Symlink in portable package")
        if p.is_file() and p.name!="package_manifest.json" and "__pycache__" not in p.parts:
            files[p.relative_to(root).as_posix()]=s.sha256(p)
    return {"format":"ralph-v57-offline-v1","files":files}

def verify_package(root):
    m=json.loads((root/"package_manifest.json").read_text(encoding="utf-8"))
    if m!=package_manifest(root): raise ValueError("Package manifest mismatch")
    return m

def read_manifest(path,count=None):
    with path.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    if (count is not None and len(rows)!=count) or not rows or len({r["segment_id"] for r in rows})!=len(rows): raise ValueError("Invalid manifest count or duplicate segment")
    return rows

def waveform_digest(x):
    return hashlib.sha256(np.ascontiguousarray(x,dtype="<f4").tobytes()).hexdigest()

def paired_bootstrap(values,groups=None,seed=57,n=5000):
    vals=np.asarray(values,dtype=np.float64)
    if vals.ndim!=1 or not len(vals) or not np.isfinite(vals).all(): raise ValueError("Invalid paired gains")
    rng=np.random.default_rng(seed)
    if groups is None:
        draws=rng.integers(len(vals),size=(n,len(vals)))
        scores=vals[draws].mean(1)
        method="clip bootstrap"
    else:
        if len(groups)!=len(vals): raise ValueError("Invalid cluster groups")
        labels=np.asarray(groups)
        unique=sorted(set(groups))
        if len(unique)<2: return {"mean":float(vals.mean()),"ci95":None,"method":"insufficient independent clusters","clusters":len(unique)}
        sums=np.array([vals[labels==k].sum() for k in unique])
        counts=np.array([sum(labels==k) for k in unique])
        draws=rng.integers(len(unique),size=(n,len(unique)))
        scores=sums[draws].sum(1)/counts[draws].sum(1)
        method="speaker-cluster bootstrap"
    return {"mean":float(vals.mean()),"ci95":[float(v) for v in np.quantile(scores,[.025,.975])],"method":method,"clusters":len(set(groups)) if groups is not None else None}

def summarize(rows):
    p=np.array([r["pesq_wb"] for r in rows])
    gain=np.array([r["gain"] for r in rows])
    def grouped(key):
        out={}
        for k in sorted({r[key] for r in rows}):
            a=[r for r in rows if r[key]==k]
            out[k]={"count":len(a),"noisy_pesq":float(np.mean([r["noisy_pesq"] for r in a])),"pesq_wb":float(np.mean([r["pesq_wb"] for r in a])),"gain":float(np.mean([r["gain"] for r in a]))}
        return out
    speakers=[r["speaker"] for r in rows]
    groups=speakers if "unknown" not in speakers else None
    return {"count":len(rows),"pesq_wb":float(p.mean()),"noisy_pesq":float(np.mean([r["noisy_pesq"] for r in rows])),
        "gain_vs_noisy":paired_bootstrap(gain,groups),"snr_db":float(np.mean([r["snr_db"] for r in rows])),
        "median_pesq":float(np.median(p)),"p05_pesq":float(np.quantile(p,.05)),"min_pesq":float(p.min()),
        "degraded_count":int((gain<0).sum()),"by_noise":grouped("noise"),"by_input_snr":grouped("input_snr_group"),
        "worst_indices":[r["index"] for r in sorted(rows,key=lambda r:r["pesq_wb"])[:10]],
        "worst_gain_indices":[r["index"] for r in sorted(rows,key=lambda r:r["gain"])[:10]]}

def evaluate(models,data,manifest,out,device,expected=None):
    from pesq import pesq
    from v51_data_audit import speaker,category
    data=data.resolve();rows=read_manifest(manifest,expected)
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    # Real WAV round-trip on disk; temporary audio is deleted on exit.
    with tempfile.TemporaryDirectory() as d:
        wav=Path(d)/"enhanced.wav"
        for i,r in enumerate(rows):
            x=s.read_wav(s.safe_path(data,r["noisy_path"]))
            c=s.read_wav(s.safe_path(data,r["clean_path"]))
            y=s.predict(models,x,device)
            sf.write(wav,y,s.SR,subtype="PCM_16")
            if sf.info(wav).subtype!="PCM_16": raise ValueError("Invalid WAV subtype")
            y=s.read_wav(wav)
            a=float(pesq(s.SR,c,y,"wb"));b=float(pesq(s.SR,c,np.clip(x,-1,1),"wb"))
            target=c.astype(np.float64)
            def snr(z): return float(10*np.log10((np.sum(target**2)+1e-12)/(np.sum((target-z.astype(np.float64))**2)+1e-12)))
            inp=snr(x);out_snr=snr(y)
            if not np.isfinite([a,b,inp,out_snr]).all(): raise ValueError("Nonfinite metric")
            nominal=round(inp,1)
            band=next((label for edge,label in ((0,"below 0"),(5,"0–<5"),(10,"5–<10"),(15,"10–<15"),(20,"15–<20")) if nominal<edge),"20+")
            records.append({"index":i,"pesq_wb":a,"noisy_pesq":b,"gain":a-b,"snr_db":out_snr,"input_snr_db":inp,"input_snr_group":band,
                "noise":category(r,("noise_type","noise_category","noise_label","noise_class")) or "unknown","speaker":speaker(r)})
    with (out/"scores.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    summary=summarize(records)
    summary.update(manifest_sha256=s.sha256(manifest),protocol="v57-pcm16-wav-paired-v1",
                   caveat="Confidence intervals are descriptive; development selection and repeated tuning can bias them.")
    (out/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print("V57 PCM16",summary["pesq_wb"],"count",summary["count"],flush=True)
    return summary

def verify_outputs(inp,out):
    expected={p.name for p in inp.iterdir() if p.is_file() and p.suffix.lower()==".wav"}
    actual={p.name for p in out.iterdir() if p.is_file() and p.suffix.lower()==".wav"}
    if not expected or actual!=expected or any(not p.is_file() or p.suffix.lower()!=".wav" or p.is_symlink() for p in out.iterdir()):
        raise ValueError("Unexpected submission files")
    for name in sorted(actual):
        s.read_wav(out/name)
        if sf.info(out/name).subtype!="PCM_16": raise ValueError("Output is not PCM16: "+name)
    return {name:s.sha256(out/name) for name in sorted(actual)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package",type=Path,default=Path(__file__).resolve().parent)
    p.add_argument("--device",choices=("cpu",),default="cpu")
    p.add_argument("--input",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--data-root",type=Path)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--count",type=int)
    p.add_argument("--probe",action="store_true")
    p.add_argument("--zip",action="store_true",help="Optional flat WAV-only ZIP after verification")
    args=p.parse_args()
    root=args.package.resolve()
    verify_package(root)
    torch.set_num_threads(2)
    block_network()
    models=s.load_bundle(root/"bundle.json",torch.device(args.device))
    if args.probe:
        x=np.sin(np.arange(s.LENGTH,dtype=np.float32)*.0864)*.03
        s.predict(models,x,torch.device("cpu"))
        print("V57 OFFLINE PROBE PASS",flush=True)
        return
    if args.output is None: p.error("--output required")
    out=args.output.resolve()
    if out.exists() or out.is_relative_to(root): p.error("New output directory outside package required")
    if args.data_root:
        data=args.data_root.resolve()
        if out.is_relative_to(data): p.error("Output cannot modify dataset")
        evaluate(models,data,args.manifest or data/"validation/samples.csv",out,torch.device("cpu"),args.count)
    else:
        if args.input is None: p.error("--input required")
        s.run_directory(models,args.input.resolve(),out,torch.device("cpu"))
        files=verify_outputs(args.input.resolve(),out)
        # Manifest outside WAV directory, suitable for a WAV-only submission ZIP.
        (out.parent/(out.name+".sha256.json")).write_text(json.dumps(files,indent=2),encoding="utf-8")
        if args.zip:
            import zipfile
            target=out.parent/(out.name+".zip")
            with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_DEFLATED) as z:
                for name in sorted(files): z.write(out/name,arcname=name)
            with zipfile.ZipFile(target) as z:
                if z.testzip() is not None or set(z.namelist())!=set(files) or any(hashlib.sha256(z.read(name)).hexdigest()!=digest for name,digest in files.items()): raise RuntimeError("ZIP verification failed")
        print("V57 verified submission",len(files),flush=True)

if __name__=="__main__": main()
