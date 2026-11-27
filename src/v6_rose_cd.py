"""ROSE-CD official CT one-step differentiable inference adapter.
Pinned MIT source (no Lightning, training dataset, or CUDA extension execution).
Original checkpoint is loaded with weights_only=True; EMA folded into the network.
"""
import ast
import hashlib
import importlib
import json
import math
import sys
import types
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

V6_ROSE_REV="8270abbf48b016a1a0be370d6c40e67ca8b85fab"
V6_ROSE_FILES={
 "LICENSE":("f44353ef1a5483c9269efb0dfaf1ca39d139d076",1104),
 "sgmse/backbones/ncsnpp_v2.py":("99cebc89c9b3a4c8976b42fe2930d5f9b7124d4f",15820),
 "sgmse/backbones/shared.py":("c6f8a9d202a5748c86386be17c7fc6e475c45160",4879),
 "sgmse/backbones/ncsnpp_utils/layers.py":("b0d6e87f7ca55177d207102234d3416dc9df88da",22579),
 "sgmse/backbones/ncsnpp_utils/layerspp.py":("948b06884f72ffc4ec2bae0fa91fae9f5c863412",9001),
 "sgmse/backbones/ncsnpp_utils/normalization.py":("9a232043b1e13ad24b2eccba7ac4dfa02969a51d",7657),
 "sgmse/backbones/ncsnpp_utils/up_or_down_sampling.py":("cf7cd443fe7b6cf4f5d99afce4ed7a5f5a0fccf5",8901),
 "sgmse/backbones/ncsnpp_utils/utils.py":("ed4c05bb8dfa9f35dfb53f51f95bac21219e1f78",5608),
 "sgmse/backbones/ncsnpp_utils/op/upfirdn2d_native.py":("f2b1acc8e1cfddf4678af609f5061de60b478a36",1663),
 "sgmse/util/registry.py":("fe18f501258b29e5c5416064546012c47db74eda",1247),
 "sgmse/util/tensors.py":("c890f4ee479eaa837b378da5fc8f9c1675f72fc9",652),
 "sgmse/sdes.py":("61a44558d48ef44b1d772df98e36c378b7272566",13109),
 "sgmse/model.py":("2541355e8f954240d84a187e3cbcd55034750129",32728),
}
V6_ROSE_NET_KEYS=("nf","ch_mult","num_res_blocks","attn_resolutions","nonlinearity","resamp_with_conv","fir","fir_kernel",
 "skip_rescale","resblock_type","progressive","progressive_input","progressive_combine","init_scale","fourier_scale","image_size","embedding_type","dropout")
V6_ROSE_CFG_KEYS=("backbone","sde","theta","sigma_min","sigma_max","t_eps","network_scaling","c_in","c_out","c_skip","sigma_data",
 "loss_type","sr","n_fft","hop_length","window","spec_factor","spec_abs_exponent","transform_type")

def v6_rose_assets(root,allow_download=False,include_weights=True,expected=None):
    root=v6_public_git_assets("LiangXu123/ROSE-CD",V6_ROSE_REV,V6_ROSE_FILES,root,allow_download)
    if include_weights: v6_hf_weights("rose_cd",root,allow_download,expected)
    return root

def v6_rose_modules(root):
    # Unique namespaced package; never import upstream __init__ that compiles CUDA ops.
    root=Path(root).resolve()
    prefix="_v6rose_"+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    def pkg(name,path):
        if name not in sys.modules:
            mod=types.ModuleType(name);mod.__path__=[str(path)];mod.__package__=name
            sys.modules[name]=mod
        return sys.modules[name]
    base=root/"sgmse"
    for suffix in ("","util","backbones","backbones.ncsnpp_utils","backbones.ncsnpp_utils.op"):
        pkg(prefix+("."+suffix if suffix else ""),base.joinpath(*suffix.split(".")) if suffix else base)
    def load(suffix,relative):
        name=prefix+"."+suffix
        if name in sys.modules: return sys.modules[name]
        p=root/relative
        m=types.ModuleType(name);m.__file__=str(p);m.__package__=name.rsplit(".",1)[0]
        sys.modules[name]=m
        src=p.read_text(encoding="utf-8").replace("from sgmse.","from "+prefix+".")
        try: exec(compile(src,str(p),"exec"),m.__dict__)
        except Exception: sys.modules.pop(name,None);raise
        return m
    load("util.registry","sgmse/util/registry.py")
    load("util.tensors","sgmse/util/tensors.py")
    op=load("backbones.ncsnpp_utils.op.upfirdn2d_native","sgmse/backbones/ncsnpp_utils/op/upfirdn2d_native.py")
    sys.modules[prefix+".backbones.ncsnpp_utils.op"].upfirdn2d=op.upfirdn2d
    load("backbones.shared","sgmse/backbones/shared.py")
    for suffix in ("normalization","layers","up_or_down_sampling","layerspp"):
        load("backbones.ncsnpp_utils."+suffix,"sgmse/backbones/ncsnpp_utils/"+suffix+".py")
    net=load("backbones.ncsnpp_v2","sgmse/backbones/ncsnpp_v2.py").NCSNpp_v2
    sde=load("sdes","sgmse/sdes.py").OUVESDE
    return net,sde

class V6RoseDataMarker:
    """Safe unpickling marker for known training-data class; never constructed or used."""

def v6_rose_read_checkpoint(path):
    # Official checkpoint may contain the class reference in hyper_parameters.
    # Permit only that inert marker; never fall back to arbitrary pickle execution.
    with torch.serialization.safe_globals([(V6RoseDataMarker,"sgmse.data_module.SpecsDataModule")]):
        cp=torch.load(path,map_location="cpu",weights_only=True)
    if not isinstance(cp,dict) or not isinstance(cp.get("hyper_parameters"),dict) or not isinstance(cp.get("state_dict"),dict):
        raise ValueError("Invalid ROSE Lightning checkpoint")
    return cp

def v6_rose_config(h):
    # Keep architecture and inference hyperparameters only; no dataset paths or classes.
    defaults={"t_eps":.03,"network_scaling":None,"c_in":"1","c_out":"1","c_skip":"0","sigma_data":.1,
              "sr":16000,"n_fft":510,"hop_length":128,"window":"hann","spec_factor":.15,"spec_abs_exponent":.5,"transform_type":"exponent"}
    cfg={**defaults,**{k:h[k] for k in (*V6_ROSE_CFG_KEYS,*V6_ROSE_NET_KEYS) if k in h}}
    if cfg.get("backbone")!="ncsnpp_v2" or cfg.get("sde")!="ouve" or cfg.get("loss_type")!="precond_denoiser":
        raise ValueError("Expected official VoiceBank CT ncsnpp_v2/OUVE/preconditioned denoiser")
    if cfg["sr"]!=16000 or cfg["n_fft"]!=510 or cfg["hop_length"]<=0 or cfg["window"] not in ("hann","sqrthann"):
        raise ValueError("Unexpected ROSE speech transform")
    if cfg["transform_type"] not in ("exponent","log","none") or cfg["spec_factor"]<=0 or cfg["spec_abs_exponent"]<=0:
        raise ValueError("Invalid ROSE compression")
    if any(k not in cfg or not math.isfinite(float(cfg[k])) or float(cfg[k])<=0 for k in ("theta","sigma_min","sigma_max","sigma_data")) or cfg["sigma_max"]<=cfg["sigma_min"]:
        raise ValueError("Invalid OUVE parameters")
    # Bound model dimensions and JSON primitives; never use arbitrary serialized objects.
    if not 1<=int(cfg.get("nf",128))<=256 or len(cfg.get("ch_mult",[1,1,2,2,2,2,2]))>8:
        raise ValueError("Unexpected ROSE architecture")
    return json.loads(json.dumps(cfg,allow_nan=False))

class V6RoseCD(nn.Module):
    def __init__(self,asset_root,allow_download=False,pretrained=True,expected=None,config=None):
        super().__init__()
        root=v6_rose_assets(asset_root,allow_download,pretrained,expected)
        cp=None
        if pretrained:
            path,_=v6_hf_weights("rose_cd",root,False,expected)
            cp=v6_rose_read_checkpoint(path)
            cfg=v6_rose_config(cp["hyper_parameters"])
        elif config is not None: cfg=v6_rose_config(config)
        else:
            cfg=v6_rose_config(json.loads((root/"rose_config.json").read_text(encoding="utf-8")))
        self.cfg=cfg
        net,sde=v6_rose_modules(root)
        self.dnn=net(**{k:cfg[k] for k in V6_ROSE_NET_KEYS if k in cfg})
        self.sde=sde(cfg["theta"],cfg["sigma_min"],cfg["sigma_max"],N=1)
        # Reuse the exact pinned preconditioning methods, without importing training code.
        tree=ast.parse((root/"sgmse/model.py").read_text(encoding="utf-8"))
        score_cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ScoreModel")
        methods=[n for n in score_cls.body if isinstance(n,ast.FunctionDef) and n.name in ("forward","_c_in","_c_out","_c_skip")]
        if len(methods)!=4: raise ValueError("ROSE method contract")
        ns={"torch":torch}
        exec(compile(ast.Module(body=methods,type_ignores=[]),"rose_score_methods","exec"),ns)
        for name in ("_c_in","_c_out","_c_skip"): setattr(self,name,types.MethodType(ns[name],self))
        self._score=types.MethodType(ns["forward"],self)
        for name in ("backbone","loss_type","network_scaling","c_in","c_out","c_skip","sigma_data"): setattr(self,name,cfg[name])
        if cp is not None:
            states=cp["state_dict"]
            dnn={k[4:]:v for k,v in states.items() if k.startswith("dnn.")}
            if any(not k.startswith(("dnn.","pesq_loss.","target_score_model.","teacher_score_model.")) for k in states):
                raise ValueError("Unexpected ROSE state keys")
            self.dnn.load_state_dict(dnn,strict=True)
            ema=cp.get("ema")
            params=list(self.dnn.parameters())
            shadows=ema.get("shadow_params") if isinstance(ema,dict) else None
            if not isinstance(shadows,(list,tuple)) or len(shadows)!=len(params): raise ValueError("Missing/mismatched ROSE EMA")
            with torch.no_grad():
                for p,s in zip(params,shadows):
                    if p.shape!=s.shape or not torch.isfinite(s).all(): raise ValueError("Invalid EMA tensor")
                    p.copy_(s)
            # Portable config, not the original pickle. Archive alongside final state dict.
            (root/"rose_config.json").write_text(json.dumps(cfg,indent=2,allow_nan=False),encoding="utf-8")
    def _transform(self,z,inverse=False):
        mode=self.cfg["transform_type"];factor=self.cfg["spec_factor"];ex=self.cfg["spec_abs_exponent"]
        if mode=="none": return z
        if inverse:
            z=z/factor
            a=z.abs().pow(1/ex) if mode=="exponent" else z.abs().expm1()
            return torch.polar(a,z.angle())
        a=z.abs().pow(ex) if mode=="exponent" else z.abs().log1p()
        return factor*torch.polar(a,z.angle())
    def _one(self,x):
        n=x.shape[-1];peak=x.abs().amax(-1,keepdim=True).clamp_min(1e-8)
        w=torch.hann_window(self.cfg["n_fft"],device=x.device,dtype=x.dtype)
        if self.cfg["window"]=="sqrthann": w=w.sqrt()
        z=torch.stft(x/peak,self.cfg["n_fft"],self.cfg["hop_length"],window=w,center=True,return_complex=True)
        z=self._transform(z).unsqueeze(1)
        pad=(-z.shape[-1])%64
        # Complex reflect padding uses real channels for portable CPU/CUDA.
        if pad:
            mode="reflect" if z.shape[-1]>pad else "replicate"
            z=torch.complex(F.pad(z.real,(0,pad,0,0),mode=mode),F.pad(z.imag,(0,pad,0,0),mode=mode))
        # Exact one-step CT prior at T=1; fixed independent draw per utterance
        # for repeatable evaluation, not chosen using reference audio.
        gen=torch.Generator(device="cpu").manual_seed(6026)
        noise=torch.complex(torch.randn(z.shape,generator=gen),torch.randn(z.shape,generator=gen)).to(z.device)/math.sqrt(2.)
        t=torch.ones(len(x),device=x.device)
        xt=z+noise*self.sde._std(t)[:,None,None,None]
        enhanced=self._score(xt,z,t,return_X=True).squeeze(1)[...,:z.shape[-1]-pad if pad else z.shape[-1]]
        spec=self._transform(enhanced,inverse=True)
        return torch.istft(spec,self.cfg["n_fft"],self.cfg["hop_length"],window=w,center=True,length=n)*peak
    def forward(self,x):
        if x.ndim!=2 or x.shape[-1]<=self.cfg["n_fft"]: raise ValueError("ROSE expects B x samples")
        return torch.cat([self._one(v) for v in x.split(1)],0)

def v6_rose_groups(model):
    return [("head",[model.dnn.output_layer]),("backbone",[model.dnn])]
