"""Official SEMamba++ adapter with portable Mamba1 and immutable HF asset lock.
The original repository moved from infected4098 to Eddieyjlee (same repository ID).
Source is fetched locally from pinned upstream; do not redistribute without checking rights.
"""
import ast
import json
import hashlib
import urllib.request
from pathlib import Path
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F

V6_PP_REV="cd369c89a8106072c077b468417b765b227afd74"
V6_PP_FILES={
 "model/FAN.py":("1c93e40b6724da49bf8b3328c6901c5949b3b534",6453),
 "model/frequencyglp.py":("8ff579ed06da22ba557c0e59a501db92646d33c0",3063),
 "model/encdec.py":("4dfcc8a38daa3c04f9d02e968993ac0782b1bf80",7351),
 "model/semambapp.py":("bafb802120944c5465c342db2e25e98800bba2c3",14563),
 "config.yaml":("5b200bc21783893c746dbd09874b5a696c0dee8e",None),
}
V6_HF_MODELS={
 "semambapp":("yongjoonlee/semambapp","semambapp.pth",{"gpl-2.0","gpl-2.0-only"}),
 "rose_cd":("LIANGXU123/ROSE-CD","logs/CT_pesq5e-4_L2/l6wdouw9/last.ckpt",{"mit"}),
}

def v6_public_git_assets(repo,rev,files,root,allow_download):
    root=Path(root)
    for name,(digest,size) in files.items():
        p=root/name
        if not p.is_file():
            if not allow_download: raise FileNotFoundError(p)
            p.parent.mkdir(parents=True,exist_ok=True)
            with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{rev}/{name}",timeout=90) as response:
                data=response.read((size or 2000000)+1)
            if (size is not None and len(data)!=size) or hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()!=digest: raise ValueError("Pinned source hash mismatch "+name)
            tmp=p.with_suffix(p.suffix+".partial");tmp.write_bytes(data);tmp.replace(p)
        data=p.read_bytes()
        if (size is not None and len(data)!=size) or hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()!=digest: raise ValueError("Local source hash mismatch "+name)
    return root

def v6_hf_lock(family,root,allow_download=False,expected=None):
    """Resolve official HF LFS SHA256 and immutable revision once. Never trust floating main at inference."""
    import re
    root=Path(root);repo,name,licenses=V6_HF_MODELS[family]
    p=root/"hf_lock.json"
    if p.is_file():
        lock=json.loads(p.read_text(encoding="utf-8"))
    elif expected:
        lock=dict(expected)
    else:
        if not allow_download: raise FileNotFoundError(p)
        from huggingface_hub import HfApi
        info=HfApi().model_info(repo,files_metadata=True,token=False)
        license=(getattr(info.card_data,"license",None) or "").lower()
        if license not in licenses: raise ValueError("Unknown or incompatible checkpoint license: "+license)
        files=[s for s in info.siblings if s.rfilename==name]
        if len(files)!=1 and family=="rose_cd":
            # HF release directories may omit the experiment run ID used in the
            # original eval script. Only the named CT loss variant, not teacher/other datasets.
            files=[s for s in info.siblings if "CT_pesq5e-4_L2" in s.rfilename.split("/") and s.rfilename.endswith(".ckpt")]
        if len(files)!=1:
            raise ValueError("Official checkpoint path unavailable/ambiguous: "+name+" candidates="+str([s.rfilename for s in info.siblings if s.rfilename.lower().endswith((".ckpt",".pth",".pt",".safetensors"))])[:5000])
        s=files[0];lfs=s.lfs
        if lfs:
            digest,size=lfs.sha256,int(lfs.size)
        else:
            from huggingface_hub import hf_hub_url,get_hf_file_metadata
            md=get_hf_file_metadata(hf_hub_url(repo,s.rfilename,revision=info.sha),token=False)
            digest,size=md.etag.strip('"'),int(md.size or 0)
        lock={"repo":repo,"revision":info.sha,"path":s.rfilename,"sha256":digest,"size":size,"license":license}
    if expected is not None and lock!=expected: raise ValueError("HF lock differs from accepted preflight")
    if (lock.get("repo")!=repo or lock.get("license") not in licenses or
        not re.fullmatch(r"[a-f0-9]{40}",lock.get("revision","")) or
        not re.fullmatch(r"[a-f0-9]{64}",lock.get("sha256","")) or not 0<int(lock.get("size",0))<=2*1024**3):
        raise ValueError("Invalid model artifact lock")
    path=lock.get("path","")
    if Path(path).is_absolute() or ".." in Path(path).parts or "\\" in path: raise ValueError("Invalid locked checkpoint path")
    if family=="rose_cd":
        if "CT_pesq5e-4_L2" not in path.split("/") or not path.endswith(".ckpt"): raise ValueError("Wrong CT checkpoint variant")
    elif path!=name: raise ValueError("Wrong official model filename")
    root.mkdir(parents=True,exist_ok=True)
    if not p.exists(): p.write_text(json.dumps(lock,indent=2,sort_keys=True),encoding="utf-8")
    return lock

def v6_hf_weights(family,root,allow_download=False,expected=None):
    root=Path(root);lock=v6_hf_lock(family,root,allow_download,expected)
    p=root/"official_weights.pt"
    if not p.is_file():
        if not allow_download: raise FileNotFoundError(p)
        from huggingface_hub import hf_hub_download
        import shutil
        downloaded=Path(hf_hub_download(lock["repo"],lock["path"],revision=lock["revision"],token=False))
        tmp=p.with_suffix(".partial")
        shutil.copyfile(downloaded,tmp)
        with tmp.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
        if tmp.stat().st_size!=lock["size"] or digest!=lock["sha256"]: raise ValueError("Downloaded HF checkpoint mismatch")
        tmp.replace(p)
    with p.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
    if p.stat().st_size!=lock["size"] or digest!=lock["sha256"]: raise ValueError("Local HF checkpoint mismatch")
    return p,lock

def v6_pp_assets(root,allow_download=False,include_weights=True,expected=None):
    root=v6_public_git_assets("Eddieyjlee/SEMambapp",V6_PP_REV,V6_PP_FILES,root,allow_download)
    if include_weights: v6_hf_weights("semambapp",root,allow_download,expected)
    return root

def v6_pp_net(root):
    import yaml
    from einops import rearrange
    ns={"torch":torch,"nn":nn,"F":F,"rearrange":rearrange,"partial":partial,
        "Mamba":V6Mamba1,"Block":V6MambaBlock,"RMSNorm":V6RMSNorm,
        "_init_weights":lambda module,**kw:None,"__name__":"v6_pinned_pp"}
    specs=[("model/FAN.py",["FANLayer","FANFFN_gate_freq","FANFFN_gate_channel"]),
        ("model/frequencyglp.py",["LocalFrequencyMix","FrequencyGLP"]),
        ("model/encdec.py",["LearnableSoftplus","get_padding","get_padding_2d","DenseBlock","DenseEncoder","MagDecoder","PhaseDecoder"]),
        ("model/semambapp.py",["LayerNorm","compute_auto_padding_2d","create_block","MambaBlock","SEMambapp_bottleneck","SEMambapp"])]
    for name,expected in specs:
        tree=ast.parse((Path(root)/name).read_text(encoding="utf-8"))
        nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in expected]
        if [n.name for n in nodes]!=expected: raise ValueError("SEMamba++ source structure mismatch")
        exec(compile(ast.Module(body=nodes,type_ignores=[]),name,"exec"),ns)
    cfg=yaml.safe_load((Path(root)/"config.yaml").read_text(encoding="utf-8"))
    st=cfg["stft_cfg"]
    if tuple(st[k] for k in ("sampling_rate","n_fft","hop_size","win_size"))!=(16000,400,100,400) or cfg["model_cfg"]["compress_factor"]!=.3: raise ValueError("SEMamba++ STFT mismatch")
    return ns["SEMambapp"](cfg)

class V6SEMambaPP(nn.Module):
    def __init__(self,asset_root,allow_download=False,pretrained=True,expected=None,use_fused=False):
        super().__init__()
        root=v6_pp_assets(asset_root,allow_download,pretrained,expected)
        self.net=v6_pp_net(root)
        if pretrained:
            p,_=v6_hf_weights("semambapp",root,False,expected)
            state=torch.load(p,map_location="cpu",weights_only=True)
            self.net.load_state_dict(state["generator"],strict=True)
        for m in self.net.modules():
            if isinstance(m,V6Mamba1): m.use_fused=bool(use_fused)
    def forward(self,x):
        if x.ndim!=2 or x.shape[-1]<=400: raise ValueError("SEMamba++ expects B x samples")
        # The official model uses peak, not RMS, normalization and predicts absolute magnitude.
        peak=x.abs().amax(-1,keepdim=True).clamp_min(1e-9)
        w=torch.hann_window(400,device=x.device,dtype=x.dtype)
        z=torch.stft(x/(peak+1e-9),400,100,400,window=w,center=True,pad_mode="reflect",return_complex=True)
        a,p,_=self.net(z.abs().clamp_min(1e-10).pow(.3),z.angle())
        amp=a.clamp_min(0).pow(1/.3)
        return torch.istft(torch.complex(amp*p.cos(),amp*p.sin()),400,100,400,window=w,center=True,length=x.shape[-1])*peak
