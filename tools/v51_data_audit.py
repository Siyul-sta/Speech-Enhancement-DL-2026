"""V5.1 local dataset diagnostics; no uploads, audio output or model training."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
import soundfile as sf

EXPECTED={"train":1296,"validation":164}
SR,LENGTH=16000,48000

def digest(wave):
    return hashlib.sha256(np.ascontiguousarray(wave,dtype="<f4").tobytes()).hexdigest()

def read_wave(root,name):
    p=(root/name).resolve()
    if not p.is_relative_to(root): raise ValueError("Manifest path escapes dataset root")
    x,sr=sf.read(p,dtype="float32",always_2d=True)
    if sr!=SR or x.shape!=(LENGTH,1) or not np.isfinite(x).all():
        raise ValueError("Invalid sample rate, shape or values")
    return x[:,0].astype(np.float64)

def category(row,names):
    for k in names:
        if (row.get(k) or "").strip(): return row[k].strip()
    return None

def speaker(row):
    name=category(row,("speaker_id","speaker","speaker_name"))
    if name is None:
        m=re.match(r"(speaker[0-9]+)(?:_|$)",row["segment_id"],re.I)
        name=m.group(1).lower() if m else None
    return hashlib.sha256(name.encode()).hexdigest()[:12] if name else "unknown"

def band_fractions(x):
    # Hann-weighted 3-second residual spectrum; descriptive, not noise class.
    z=np.abs(np.fft.rfft(x*np.hanning(len(x))))**2
    f=np.fft.rfftfreq(len(x),1/SR)
    total=z.sum()
    if total<=1e-18: return [0.,0.,0.]
    return [float(z[m].sum()/total) for m in (f<300,(f>=300)&(f<3400),f>=3400)]

def describe_pair(noisy,clean):
    n=noisy-clean
    cp=float(np.mean(clean**2)); npow=float(np.mean(n**2))
    corr=float(np.mean(clean*n)/max(math.sqrt(cp*npow),1e-12))
    snr=float(10*np.log10((cp+1e-12)/(npow+1e-12)))
    low,mid,high=band_fractions(n)
    return {"input_snr_db":snr,"clean_rms":math.sqrt(cp),"residual_rms":math.sqrt(npow),
            "abs_residual_correlation":abs(corr),"clean_silent":cp<1e-8,
            "noisy_clip_fraction":float(np.mean(np.abs(noisy)>=.999)),
            "residual_low_fraction":low,
            "residual_mid_fraction":mid,
            "residual_high_fraction":high}

def snr_group(snr):
    for upper,label in ((0,"<0"),(5,"0–5"),(10,"5–10"),(15,"10–15"),(20,"15–20")):
        if snr<upper: return label
    return "20+"

def score_rows(path,count):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        r=list(csv.DictReader(f))
    if len(r)!=count or [int(x["index"]) for x in r]!=list(range(count)):
        raise ValueError("PESQ CSV must use complete manifest order (index 0..n-1)")
    vals=[float(x["pesq_wb"]) for x in r]
    if not all(math.isfinite(x) for x in vals): raise ValueError("Non-finite PESQ")
    return vals

def groups(rows,key):
    result={}
    for value in sorted({r[key] for r in rows}):
        subset=[r for r in rows if r[key]==value]
        item={"count":len(subset),"mean_snr_db":float(np.mean([r["input_snr_db"] for r in subset]))}
        for name in ("noisy_pesq","model_pesq","pesq_gain"):
            if name in subset[0]:
                item["mean_"+name]=float(np.mean([r[name] for r in subset]))
        result[value]=item
    return result

def audit(root,out,with_pesq=False,model_scores=None):
    root=Path(root).resolve(); out=Path(out).resolve()
    if out==root or out.is_relative_to(root): raise ValueError("Keep reports outside dataset root")
    if out.exists(): raise ValueError("Use a new report directory")
    if with_pesq:
        from pesq import pesq
    else: pesq=None
    if model_scores and not with_pesq: raise ValueError("--model-scores requires --pesq")
    all_rows={}; fingerprints={}; identifiers={}; manifest_hashes={}; repeated_ids={}
    for split,count in EXPECTED.items():
        with (root/split/"samples.csv").open(encoding="utf-8-sig",newline="") as f:
            manifest=list(csv.DictReader(f))
        if len(manifest)!=count: raise ValueError(f"{split}: expected {count} rows, found {len(manifest)}")
        ids=[r["segment_id"] for r in manifest]
        if not all(ids): raise ValueError("Missing segment IDs within "+split)
        repeated_ids[split]=len(ids)-len(set(ids))
        manifest_hashes[split]=hashlib.sha256((root/split/"samples.csv").read_bytes()).hexdigest()
        identifiers[split]=set(ids)
        hashes={"clean":set(),"noisy":set()}; rows=[]
        scores=score_rows(model_scores,count) if model_scores and split=="validation" else None
        for i,r in enumerate(manifest):
            x,c=read_wave(root,r["noisy_path"]),read_wave(root,r["clean_path"])
            hashes["clean"].add(digest(c));hashes["noisy"].add(digest(x))
            item={"index":i,"speaker_group":speaker(r),
                  "noise_label":category(r,("noise_type","noise_category","noise_label","noise_class")) or "unknown",
                  **describe_pair(x,c)}
            item["snr_group"]=snr_group(item["input_snr_db"])
            if pesq and split=="validation":
                value=float(pesq(SR,c.astype(np.float32).clip(-1,1),x.astype(np.float32).clip(-1,1),"wb"))
                if not math.isfinite(value): raise ValueError("Invalid noisy PESQ")
                item["noisy_pesq"]=value
                if scores is not None:
                    item["model_pesq"]=scores[i];item["pesq_gain"]=scores[i]-value
            rows.append(item)
            if (i+1)%100==0 or i+1==count: print(split,i+1,"/",count,flush=True)
        all_rows[split]=rows; fingerprints[split]=hashes
    overlap={k:len(fingerprints["train"][k]&fingerprints["validation"][k]) for k in ("clean","noisy")}
    overlap["segment_id"]=len(identifiers["train"]&identifiers["validation"])
    a,b=({r["speaker_group"] for r in all_rows[s]}-{"unknown"} for s in EXPECTED)
    overlap["known_speaker_groups"]=len(a&b)
    summary={"diagnostic_only":True,"cross_split_exact_overlap":overlap,"splits":{},
             "manifest_sha256":manifest_hashes,
             "model_scores_sha256":hashlib.sha256(Path(model_scores).read_bytes()).hexdigest() if model_scores else None,
             "notes":["Exact waveform hash overlap is not a near-duplicate or alignment test.",
                      "Speaker IDs are metadata or segment-name heuristics; unknown is not a speaker.",
                      "Noise labels are read from manifest only, never inferred from spectrum.",
                      "Residual is noisy minus clean; may include speech, reverberation or alignment error.",
                      "Supplied model score CSV is matched by row index; use the exact dataset and manifest version."]}
    for split,rows in all_rows.items():
        summary["splits"][split]={"count":len(rows),"unique_clean":len(fingerprints[split]["clean"]),
             "unique_noisy":len(fingerprints[split]["noisy"]),
             "repeated_segment_ids":repeated_ids[split],
             "known_speaker_groups":len({r["speaker_group"] for r in rows}-{"unknown"}),
             "silence_count":sum(r["clean_silent"] for r in rows),
             "residual_corr_gt_0p15_count":sum(r["abs_residual_correlation"]>.15 for r in rows),
             "snr_quantiles_db":dict(zip(("p05","p50","p95"),map(float,np.quantile([r["input_snr_db"] for r in rows],[.05,.5,.95])))),
             "by_snr":groups(rows,"snr_group"),"by_noise_label":groups(rows,"noise_label"),"by_speaker":groups(rows,"speaker_group")}
    out.mkdir(parents=True)
    for split,rows in all_rows.items():
        with (out/(split+"_diagnostics.csv")).open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/"dataset_audit.json").write_text(json.dumps(summary,indent=2,allow_nan=False),encoding="utf-8")
    print("EXACT OVERLAP",overlap,flush=True)
    print("Local report:",out/"dataset_audit.json",flush=True)
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--pesq",action="store_true",help="Score original noisy validation only")
    p.add_argument("--model-scores",type=Path,help="Complete validation_per_file.csv from exact matching manifest")
    args=p.parse_args()
    audit(args.root,args.out,args.pesq,args.model_scores)

if __name__=="__main__": main()
