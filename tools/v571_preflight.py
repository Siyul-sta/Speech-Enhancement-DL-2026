"""V5.7.1 real CPU public-model preflight. No Kaggle jobs or private uploads."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
REPORT=ROOT/"runs/v57_public_preflight/report.json"
FAMILIES=("cmgan_transfer","cascade_cmgan_two","cascade_cmgan_three","cascade_cmgan","metricgan","sepformer_dns_transfer","sepformer_wham_transfer")
FILES=("src/kaggle_v4_external.py","src/kaggle_v4_public.py","tools/ralph_v4.py","tools/ralph_v56.py","tools/v57_public_assets.py","tools/submit_v5.py","tools/v55_gating.py","tools/v55_ensemble.py","tools/v57_offline.py","tools/v57_finalize.py","tools/v571_preflight.py","tools/v572_scale_audit.py")
DEPS=("torch","torchaudio","speechbrain","hyperpyyaml","ruamel.yaml","huggingface_hub","numpy","pandas","scipy","soundfile","einops","pesq")

def digest(p):
    with p.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()

def identity():
    return {"files":{p:digest(ROOT/p) for p in FILES},"dependencies":{p:version(p) for p in DEPS},"python":sys.version.split()[0]}

def require_report(path=REPORT):
    if not path.is_file(): raise RuntimeError("Run py -u tools/v571_preflight.py before GPU execution")
    r=json.loads(path.read_text(encoding="utf-8"))
    if r.get("protocol")!="v571-public-cpu-v1" or r.get("identity")!=identity() or r.get("status")!="PASS" or set(r.get("families",{}))!=set(FAMILIES):
        raise RuntimeError("Public CPU preflight missing, stale or incomplete")
    if not all(x.get("status")=="PASS" for x in r["families"].values()) or r.get("mixed_ensemble",{}).get("status")!="PASS":
        raise RuntimeError("Public CPU preflight did not pass")
    cmgan=Path(r["cmgan_root"])
    from v57_public_assets import CMGAN_BLOBS,git_blob
    for name,blob in CMGAN_BLOBS.items():
        if not (cmgan/name).is_file() or git_blob((cmgan/name).read_bytes())!=blob: raise RuntimeError("Preflight CMGAN cache unavailable")
    return r

def fixture():
    import numpy as np
    local=os.environ.get("V571_LOCAL_FIXTURE")
    if local:
        with np.load(local,allow_pickle=False) as data: return data["noisy"],data["clean"]
    rng=np.random.default_rng(571)
    t=np.arange(48000,dtype=np.float32)/16000
    env=(.3+.7*np.sin(2*np.pi*2*t)**2)
    clean=(.12*np.sin(2*np.pi*190*t)+.06*np.sin(2*np.pi*380*t)+.03*np.sin(2*np.pi*760*t))*env
    noisy=clean+.02*rng.normal(size=len(t))
    return noisy.astype("float32"),clean.astype("float32")

def write(p,obj):
    tmp=p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")
    tmp.replace(p)

def cpu_numeric_settings():
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)


def waveform_equivalence(actual,reference,allow_sepformer_roundoff=False):
    """Strict original comparison, or bounded FP32 roundoff with PCM16 checks."""
    import numpy as np
    from v55_ensemble import pcm16
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(reference,dtype=np.float64)
    if a.shape!=(48000,) or b.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid equivalence waveform")
    diff=a-b
    peak=max(1.,float(np.max(np.abs(b))))
    rms=max(1.,float(np.sqrt(np.mean(b*b))))
    pcm=(pcm16(a)-pcm16(b)).astype(np.float64)*32768
    stats={"max_abs_error":float(np.max(np.abs(diff))),
           "peak_relative_error":float(np.max(np.abs(diff))/peak),
           "normalized_rmse":float(np.sqrt(np.mean(diff*diff))/rms),
           "pcm16_max_lsb_error":float(np.max(np.abs(pcm))),
           "pcm16_rms_lsb_error":float(np.sqrt(np.mean(pcm*pcm))),
           "reference_peak":float(np.max(np.abs(b))),
           "reference_clipped_fraction":float(np.mean(np.abs(b)>1)),
           "actual_clipped_fraction":float(np.mean(np.abs(a)>1))}
    strict=bool(np.allclose(a,b,atol=2e-5,rtol=2e-4))
    bounded=(stats["peak_relative_error"]<=2e-5 and stats["normalized_rmse"]<=1e-5 and
             stats["pcm16_max_lsb_error"]<=4 and stats["pcm16_rms_lsb_error"]<=1)
    if not strict and not (allow_sepformer_roundoff and bounded):
        raise RuntimeError("Waveform equivalence failed: "+json.dumps(stats))
    stats["numeric_comparison"]="strict" if strict else "bounded_sepformer_fp32_roundoff"
    return stats


def child(package,output):
    import numpy as np
    import torch
    import submit_v5 as s
    import v57_offline as off
    off.verify_package(package)
    cpu_numeric_settings()
    off.block_network()
    models=s.load_bundle(package/"bundle.json",torch.device("cpu"))
    cfg=json.loads((package/"bundle.json").read_text(encoding="utf-8"))
    if len(cfg["models"])==1 and not cfg.get("gating"):
        expected=torch.load(s.safe_path(package,cfg["models"][0]["checkpoint"]),map_location="cpu",weights_only=True)
        restored=models[0][1].state_dict()
        if set(expected)!=set(restored) or any(expected[k].dtype!=restored[k].dtype or not torch.equal(expected[k],restored[k]) for k in expected):
            raise RuntimeError("Offline state differs from exact checkpoint")
        del expected,restored
    x,_=fixture()
    z=torch.from_numpy(x)[None]
    with torch.no_grad():
        raw=sum(w*m(z) for w,m in models)
    if raw.shape!=z.shape or not torch.isfinite(raw).all(): raise RuntimeError("Invalid offline raw output")
    np.save(output,raw.numpy()[0],allow_pickle=False)

def checked_child(package,output,reference,family=None):
    import numpy as np
    from pesq import pesq
    from v55_ensemble import pcm16
    subprocess.run([sys.executable,str(Path(__file__).resolve()),"--child",str(package),str(output)],check=True)
    actual=np.load(output,allow_pickle=False)
    stats=waveform_equivalence(actual,reference,family in ("sepformer_dns_transfer","sepformer_wham_transfer"))
    _,clean=fixture()
    p=float(pesq(16000,clean,pcm16(reference),"wb"))
    q=float(pesq(16000,clean,pcm16(actual),"wb"))
    if not np.isfinite([p,q]).all() or abs(p-q)>.01: raise RuntimeError("Offline PESQ mismatch")
    return {"status":"PASS",**stats,"fixture_pesq_delta":q-p,"samples":len(actual),"exact_single_state_verified":family is not None}

def spec_for(folder,source,checkpoint,family,assets=None):
    d={"weight":1.,"family":family,"source":source.relative_to(folder).as_posix(),"checkpoint":checkpoint.relative_to(folder).as_posix(),"source_sha256":digest(source),"checkpoint_sha256":digest(checkpoint)}
    if assets: d["public_assets"]=assets
    return d

def single(family,work,cache,package,policies=None):
    import re
    import numpy as np
    import torch
    import kaggle_v4_external as t
    import v57_public_assets as a
    import v57_finalize as f
    cpu_numeric_settings()
    t.WORK=cache
    t.EXPERIMENT_CONFIG={"id":"V3_RESIDUAL_PHASE","architecture":family,"seed":571,"sepformer_scale_policies":policies or {}}
    # Public hparams capture belongs to this family; remove stale captures.
    for name in ("offline_hyperparams.yaml","offline_hyperparams.json"):
        (cache/name).unlink(missing_ok=True)
    torch.manual_seed(571)
    model=getattr(t,a.PUBLIC[family])().eval()
    x,_=fixture();z=torch.from_numpy(x)[None]
    with torch.no_grad():
        original=model(z)
        if original.shape!=z.shape or not torch.isfinite(original).all(): raise RuntimeError("Online output invalid")
        # Emulate altered fine-tuned state (not real training). This ensures
        # the offline path cannot accidentally return only original weights.
        if hasattr(model,"residual"): p=model.residual.out.bias
        else: p=next(p for p in reversed(list(model.parameters())) if p.requires_grad)
        p.reshape(-1)[0].add_(.01)
        expected=model(z).numpy()[0].copy()
    package.mkdir()
    source=package/"source.py"
    code=(ROOT/"src/kaggle_v4_external.py").read_text(encoding="utf-8")
    source.write_text(re.sub(r"^EXPERIMENT_CONFIG = .+$",lambda _: "EXPERIMENT_CONFIG = "+repr(t.EXPERIMENT_CONFIG),code,count=1,flags=re.M),encoding="utf-8")
    checkpoint=package/"model.pt";torch.save(model.state_dict(),checkpoint)
    meta=a.package_public(family,cache,code,package/"assets",False,cache/"cmgan_assets")
    write(package/"bundle.json",{"models":[spec_for(package,source,checkpoint,family,{**meta,"root":"assets"})]})
    f.finalize_package(package)
    del model
    r=checked_child(package,work/(family+".npy"),expected,family)
    r["checkpoint_sha256"]=digest(checkpoint)
    r["initial_fixture_max_change"]=float(np.max(np.abs(expected-original.numpy()[0])))
    if r["initial_fixture_max_change"]<=1e-9: raise RuntimeError("Modified checkpoint did not change fixture output")
    if family.startswith("sepformer_") and (policies or {}).get(family,"raw")!="raw" and (r["reference_peak"]>.98001 or r["actual_clipped_fraction"]>0): raise RuntimeError("Bounded SepFormer policy failed amplitude check")
    return r

def mixed(work,public_package):
    import numpy as np
    import torch
    import kaggle_v4_external as t
    import submit_v5 as s
    import v57_finalize as f
    from v55_gating import EnsembleGate
    pkg=work/"mixed";pkg.mkdir()
    sub=pkg/"public";shutil.copytree(public_package,sub)
    public=json.loads((sub/"bundle.json").read_text(encoding="utf-8"))["models"][0]
    for k in ("source","checkpoint"): public[k]="public/"+public[k]
    public["public_assets"]["root"]="public/"+public["public_assets"]["root"]
    public["weight"]=.5
    source=pkg/"native.py"
    import re
    code=(ROOT/"src/kaggle_v4_external.py").read_text(encoding="utf-8")
    source.write_text(re.sub(r"^EXPERIMENT_CONFIG = .+$","EXPERIMENT_CONFIG = {'architecture':'complex_residual'}",code,count=1,flags=re.M),encoding="utf-8")
    cp=pkg/"native.pt";torch.save(t.ComplexResidualUNet().state_dict(),cp)
    native=spec_for(pkg,source,cp,"complex_residual");native["weight"]=.5
    x,_=fixture()
    results={}
    for mode in ("fixed","clip","frequency"):
        cfg={"models":[public,native]}
        if mode!="fixed":
            gate=EnsembleGate(2,mode)
            with torch.no_grad(): gate.net[-1].bias[0]=.3
            target=pkg/"gate.pt";torch.save(gate.state_dict(),target)
            cfg["gating"]={"mode":mode,"experts":2,"checkpoint":"gate.pt","checkpoint_sha256":digest(target)}
        write(pkg/"bundle.json",cfg)
        f.finalize_package(pkg)
        models=s.load_bundle(pkg/"bundle.json",torch.device("cpu"))
        z=torch.from_numpy(x)[None]
        with torch.no_grad(): expected=sum(w*m(z) for w,m in models).numpy()[0]
        del models
        results[mode]=checked_child(pkg,work/(mode+".npy"),expected)
    return {"status":"PASS","modes":results}

def run(out,data_root=None):
    import torch
    import torchaudio
    if torch.cuda.is_available(): print("CUDA available but CPU preflight uses CPU only",flush=True)
    if str(torch.__version__).split("+")[0]!=str(torchaudio.__version__).split("+")[0]: raise RuntimeError("Torch/TorchAudio version mismatch")
    if version("ruamel.yaml")!="0.18.16": raise RuntimeError("Install ruamel.yaml==0.18.16 before preflight")
    out.mkdir(parents=True,exist_ok=True)
    cache=out/"assets_cache";cache.mkdir(exist_ok=True)
    policies={}
    if data_root:
        from v572_scale_audit import require_report as require_scale
        policies=require_scale(data_root=data_root)
    report={"protocol":"v571-public-cpu-v1","status":"RUNNING","identity":identity(),"families":{},"cmgan_root":str(cache/"cmgan_assets"),"sepformer_scale_policies":policies}
    path=out/"report.json";write(path,report)
    with tempfile.TemporaryDirectory(dir=out) as d:
        work=Path(d);keep=None
        if data_root:
            import numpy as np
            import submit_v5 as s
            import v57_offline as off
            data_root=data_root.resolve();manifest=data_root/"train/samples.csv"
            rows=off.read_manifest(manifest,1296);row=rows[0]
            x=s.read_wav(s.safe_path(data_root,row["noisy_path"]));y=s.read_wav(s.safe_path(data_root,row["clean_path"]))
            fixture_path=work/"fixture.npz";np.savez(fixture_path,noisy=x,clean=y)
            os.environ["V571_LOCAL_FIXTURE"]=str(fixture_path)
            report["fixture"]={"kind":"local train pair","index":0,"manifest_sha256":digest(manifest)}
        else:
            os.environ.pop("V571_LOCAL_FIXTURE",None)
            report["fixture"]={"kind":"synthetic engineering fixture"}
        for family in FAMILIES:
            pkg=work/family
            print("V571 CHECK",family,flush=True)
            try:
                r=single(family,work,cache,pkg,policies)
                if family=="cmgan_transfer": keep=pkg
            except Exception as exc:
                r={"status":"FAIL","error":repr(exc)}
            report["families"][family]=r
            print(family,r,flush=True);write(path,report)
            if pkg!=keep: shutil.rmtree(pkg,ignore_errors=True)
        try:
            if keep is None: raise RuntimeError("CMGAN preflight failed")
            report["mixed_ensemble"]=mixed(work,keep)
        except Exception as exc: report["mixed_ensemble"]={"status":"FAIL","error":repr(exc)}
    print("V571 MIXED ENSEMBLE",report["mixed_ensemble"],flush=True)
    os.environ.pop("V571_LOCAL_FIXTURE",None)
    passed=all(r["status"]=="PASS" for r in report["families"].values()) and report["mixed_ensemble"]["status"]=="PASS"
    report["status"]="PASS" if passed else "FAIL"
    write(path,report)
    print("V571 PUBLIC PREFLIGHT",report["status"],path,flush=True)
    if not passed: raise SystemExit(1)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=REPORT.parent)
    p.add_argument("--data-root",type=Path,help="Optional real local train-only PCM16/PESQ equivalence fixture")
    p.add_argument("--child",type=Path,nargs=2,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.child: child(*args.child)
    else:
        import ralph_v2 as core
        core.no_uncertain_prior_jobs()
        lock=core.OUTPUT/"active.lock";lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open("x",encoding="utf-8") as f: f.write(str(os.getpid()))
        try: run(args.out.resolve(),args.data_root)
        finally: lock.unlink(missing_ok=True)

if __name__=="__main__": main()
