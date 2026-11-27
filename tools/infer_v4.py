"""Offline inference for trusted V4 checkpoint bundles, arbitrary audio lengths/rates."""
import argparse
import importlib.util
import math
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

def resample(x, source, target):
    g=math.gcd(source,target)
    return resample_poly(x,target//g,source//g).astype(np.float32) if source!=target else x.astype(np.float32)

@torch.inference_mode()
def enhance(model,x,device,window=48000,hop=24000):
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty mono audio")
    if not 0<hop<window: raise ValueError("Invalid overlap")
    output=np.zeros(len(x),np.float64)
    weights=np.zeros(len(x),np.float64)
    taper=np.maximum(np.hanning(window),1e-3)
    for start in range(0,len(x),hop):
        segment=x[start:start+window]
        padded=np.pad(segment,(0,window-len(segment)))
        y=model(torch.from_numpy(padded.astype(np.float32))[None].to(device))
        if y.shape!=(1,window) or not torch.isfinite(y).all():
            raise RuntimeError("Model output contract failed")
        n=len(segment)
        output[start:start+n]+=y.cpu().numpy()[0,:n]*taper[:n]
        weights[start:start+n]+=taper[:n]
        if start+n==len(x): break
    return (output/weights).astype(np.float32)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--source",type=Path,required=True,help="Trusted archived experiment source.py; executable Python")
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint",type=Path)
    group.add_argument("--recipe",help="Archived public winner recipe, e.g. metricgan:1.0")
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=p.parse_args()
    if args.input.resolve()==args.output.resolve(): raise ValueError("Refusing to overwrite input")
    spec=importlib.util.spec_from_file_location("trusted_v4_bundle",args.source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    family=module.EXPERIMENT_CONFIG.get("architecture","residual")
    names={"residual":"ResidualPhaseUNet","residual_relaxed":"RelaxedPhaseUNet","complex_residual":"ComplexResidualUNet","dual_axis":"DualAxisEnhancer","cmgan_transfer":"CMGANTransfer","temporal":"TemporalEnhancer","metricgan":"MetricGANTransfer","sepformer_dns_transfer":"SepformerTransfer","sepformer_wham_transfer":"SepformerTransfer","moe":"AdaptiveMixture","composed":"ComposedEnhancer","generated":"GeneratedEnhancer"}
    if args.recipe:
        fn=module.public_recipe(args.recipe,torch.device(args.device))
        model=lambda x: torch.cat([fn(v) for v in x.split(1)])
    else:
        model=getattr(module,names[family])().to(args.device).eval()
        model.load_state_dict(torch.load(args.checkpoint,map_location=args.device,weights_only=True),strict=True)
    audio,sr=sf.read(args.input,dtype="float32",always_2d=True)
    mono=resample(audio.mean(axis=1),sr,16000)
    y=resample(enhance(model,mono,args.device),16000,sr)
    y=np.pad(y,(0,max(0,len(audio)-len(y))))[:len(audio)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    sf.write(args.output,np.clip(y,-1,1),sr)
    print(f"Enhanced {len(audio)/sr:.2f}s at {sr} Hz; output mono")

if __name__=="__main__":
    main()
