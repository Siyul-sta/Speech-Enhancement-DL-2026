"""Pinned official MP-SENet inference/training adapter. Also embedded in Kaggle source.
Upstream: https://github.com/yxlu-0102/MP-SENet, MIT (Yexin Lu, 2023).
Only reviewed model classes are executed, not upstream training or inference scripts.
"""
import ast
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, GRU, Linear, LayerNorm, Dropout

V6_MP_REV="89932cfe90d1dacb8e170e4a331d762462c21792"
V6_MP_FILES={
    "models/model.py":("a289f7a5907de7b973110c9844b5bc0d190694c9",6913),
    "models/transformer.py":("f311cd4a0a69e2c896c4263d315573cfde623b4d",1983),
    "utils.py":("3183dd28bf9a0a5bea9304034604e350f2c8e62a",None),
    "LICENSE":("d1f1d55f4f342d6152467c80b305905da9e758ef",None),
    "best_ckpt/config.json":("20dc683a2f9f0d24ff19b4d65fdb97075d6d1d34",525),
    "best_ckpt/g_best_dns":("44bd3cc8ada147d47a07e36e25e9d6dd2ee7e72a",9138054),
    "best_ckpt/g_best_vb":("cb01711c8e930352b843f692dadae49a7f5718f7",9142950),
}
V6_MP_FAMILIES={"mp_dns":"dns","mp_vb":"vb"}

def v6_git_blob(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()

def v6_mp_assets(root,variant,allow_download=False,include_weights=True):
    root=Path(root)
    if variant not in ("dns","vb"): raise ValueError("Unknown MP-SENet checkpoint")
    names=["models/model.py","models/transformer.py","utils.py","LICENSE","best_ckpt/config.json"]
    if include_weights: names.append("best_ckpt/g_best_"+variant)
    for name in names:
        sha,size=V6_MP_FILES[name];p=root/name
        if not p.is_file():
            if not allow_download: raise FileNotFoundError(p)
            p.parent.mkdir(parents=True,exist_ok=True)
            url=f"https://raw.githubusercontent.com/yxlu-0102/MP-SENet/{V6_MP_REV}/{name}"
            with urllib.request.urlopen(url,timeout=90) as response:
                data=response.read((size or 1000000)+1)
            if (size is not None and len(data)!=size) or v6_git_blob(data)!=sha: raise ValueError("Upstream blob mismatch: "+name)
            tmp=p.with_suffix(p.suffix+".partial");tmp.write_bytes(data);tmp.replace(p)
        data=p.read_bytes()
        if (size is not None and len(data)!=size) or v6_git_blob(data)!=sha: raise ValueError("Cached MP-SENet blob mismatch: "+name)
    return root

def v6_mp_net(root):
    # Exact class bodies from pinned upstream; omit unused PESQ/joblib/plot imports.
    ns={"torch":torch,"nn":nn,"F":F,"MultiheadAttention":MultiheadAttention,
        "GRU":GRU,"Linear":Linear,"LayerNorm":LayerNorm,"Dropout":Dropout,"__name__":"v6_pinned_mp"}
    specs=[("utils.py",["LearnableSigmoid2d"]),
           ("models/transformer.py",["FFN","TransformerBlock"]),
           ("models/model.py",["SPConvTranspose2d","DenseBlock","DenseEncoder","MaskDecoder","PhaseDecoder","TSTransformerBlock","MPNet"])]
    for name,classes in specs:
        tree=ast.parse((Path(root)/name).read_text(encoding="utf-8"))
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in classes]
        if [n.name for n in nodes]!=classes: raise ValueError("Pinned model class contract changed")
        exec(compile(ast.Module(body=nodes,type_ignores=[]),name,"exec"),ns)
    cfg=json.loads((Path(root)/"best_ckpt/config.json").read_text(encoding="utf-8"))
    if (cfg["sampling_rate"],cfg["n_fft"],cfg["hop_size"],cfg["win_size"],cfg["compress_factor"])!=(16000,400,100,400,.3):
        raise ValueError("Unexpected official MP-SENet STFT config")
    return ns["MPNet"](SimpleNamespace(**cfg))

class V6MPNet(nn.Module):
    def __init__(self,family,asset_root,allow_download=False,pretrained=True):
        super().__init__()
        self.family=family
        self.variant=V6_MP_FAMILIES[family]
        root=v6_mp_assets(asset_root,self.variant,allow_download,include_weights=pretrained)
        self.net=v6_mp_net(root)
        if pretrained:
            state=torch.load(root/"best_ckpt"/("g_best_"+self.variant),map_location="cpu",weights_only=True)
            self.net.load_state_dict(state["generator"],strict=True)
    def _one(self,x):
        # Preserve upstream normalization, STFT and learned attention axes.
        # Run each utterance separately: upstream attention is not batch_first.
        rms=x.square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()
        y=x/rms
        w=torch.hann_window(400,device=y.device,dtype=y.dtype)
        z=torch.view_as_real(torch.stft(y,400,100,400,window=w,center=True,pad_mode="reflect",normalized=False,return_complex=True))
        mag=(z.square().sum(-1)+1e-9).sqrt().pow(.3)
        pha=torch.atan2(z[...,1]+1e-10,z[...,0]+1e-5)
        a,p,_=self.net(mag,pha)
        amp=a.clamp_min(0).pow(1/.3)
        out=torch.istft(torch.complex(amp*p.cos(),amp*p.sin()),400,100,400,window=w,center=True,length=x.shape[-1])*rms
        return out
    def forward(self,x):
        if x.ndim!=2 or x.shape[-1]<401: raise ValueError("MP-SENet expects B x samples")
        return torch.cat([self._one(y) for y in x.split(1)],dim=0)

def v6_mp_groups(model):
    n=model.net
    return [("decoder",[n.mask_decoder,n.phase_decoder]),("late",[n.TSTransformer[2],n.TSTransformer[3]]),
            ("early",[n.TSTransformer[0],n.TSTransformer[1]]),("encoder",[n.dense_encoder])]
