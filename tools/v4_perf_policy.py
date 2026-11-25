"""Quality-first paired performance benchmark; never promotes unverified variants."""
import hashlib
import json
import math
import statistics

SEEDS=(42,43,44)
PROTOCOL="v4.2-fixed-dev-v2"

def profile(epochs):
    return {"id":"V3_RESIDUAL_PHASE","architecture":"temporal","epochs":epochs,"batch_size":4,
            "patience":7,"external_size":1024,"lr":0.0001,"loss_mode":"base","data_profile":"base",
            "augmentation":"base","self_supervised":False,"distill":False,"transductive":False,
            "warm_ref":None,"perf_reproducible":True}

def signature(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()

def quality(result):
    if result.get("evaluation_protocol")!=PROTOCOL or result.get("smoke"):
        raise ValueError("Only fixed development evaluation is eligible")
    d=result["domain_scores"]; m=result["metrics"]
    values={k:float(d[k]) for k in ("challenge","external_normal","external_stress")}
    values["robust"]=float(result["robust_score"])
    for domain in ("challenge","external"):
        for key in ("si_sdr","stoi"):
            values[domain+"_"+key]=float(m[domain][key])
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError("Nonfinite quality")
    return values

def compare(pairs,epochs,amp=False):
    """Require all three paired seeds, unchanged fingerprints, no drop, >=5% speed gain."""
    reasons=[]; measurements=[]
    if len(pairs)!=len(SEEDS): return {"accepted":False,"reasons":["Incomplete seed pairs"],"pairs":[]}
    try:
        for seed,(base,candidate) in zip(SEEDS,pairs):
            for r in (base,candidate):
                if r["seed"]!=seed or r["epochs_completed"]!=epochs or not r.get("perf_reproducible"):
                    raise ValueError("Incomplete or mismatched training schedule")
                if r.get("transductive_noisy") or r.get("warm_ref") or r.get("model_family")!="temporal":
                    raise ValueError("Incompatible benchmark model")
                if not r.get("data_fingerprint") or not r.get("training_fingerprint"):
                    raise ValueError("Missing data fingerprints")
            if base["hardware"]!=candidate["hardware"] or base["data_fingerprint"]!=candidate["data_fingerprint"] or base["training_fingerprint"]!=candidate["training_fingerprint"]:
                raise ValueError("Mismatched data or hardware")
            if base["hardware"].get("device") in ("cpu",None) or base["loader_workers"]!=0 or candidate["loader_workers"]!=2:
                raise ValueError("Expected CUDA with serial versus two-worker loader")
            if base["precision"]!="fp32" or candidate["precision"]!=("amp_fp16_forward_fp32_loss" if amp else "fp32"):
                raise ValueError("Precision mismatch")
            q0,q1=quality(base),quality(candidate)
            delta={k:q1[k]-v for k,v in q0.items()}
            for k,d in delta.items():
                if d<0: reasons.append(f"seed {seed}: {k} decreased by {-d:.8f}")
            t0,t1=base["epoch_seconds"],candidate["epoch_seconds"]
            if len(t0)!=epochs or len(t1)!=epochs or not all(math.isfinite(t) and t>0 for t in t0+t1):
                raise ValueError("Missing valid epoch timings")
            ratio=sum(t1)/sum(t0)
            measurements.append({"seed":seed,"baseline_seconds":sum(t0),"candidate_seconds":sum(t1),"ratio":ratio,"delta":delta})
        ratio=statistics.median(m["ratio"] for m in measurements)
        if ratio>0.95: reasons.append("Median speed gain below 5%")
        return {"accepted":not reasons,"reasons":reasons,"pairs":measurements,"median_ratio":ratio}
    except (KeyError,TypeError,ValueError,OverflowError) as exc:
        return {"accepted":False,"reasons":reasons+[str(exc)],"pairs":measurements}

def authorized(report,source,cache,epochs,amp):
    return bool(report.get("accepted") is True and report.get("source_sha256")==signature(source)
                and report.get("cache")==cache and report.get("profile")==profile(epochs)
                and report.get("amp") is bool(amp) and report.get("seeds")==list(SEEDS))
