"""Portable SEMamba official-checkpoint adapter; no compiled Mamba dependency.
Upstream SEMamba: RoyChao19477/SEMamba, MIT with MP-SENet/CMGAN attribution.
Pinned upstream architecture classes + independently implemented Mamba1 recurrence.
"""
import ast
import math
import hashlib
import urllib.request
from pathlib import Path
from functools import partial
import torch
from torch import nn
import torch.nn.functional as F

V6_SEM_REV="750ff5d8f0e112557ca039c268cdcc51e267f342"
V6_SEM_FILES={
 "models/codec_module.py":("314f2ea60fe8ad67b34d2975e9032189a8ce4352",6296),
 "models/lsigmoid.py":("effb420bb63aa4b44b4165cee9d601df43f30f14",2139),
 "models/mamba_block.py":("439828bea5f9f1598d11b310a3df3fd5637f39b3",4306),
 "models/generator.py":("2fd354ac5f67ca582c7b835572fc2c66f0cd90ea",2666),
 "ckpts/config.yaml":("9098eff8c9fe68fcc09b9a5b2f6ee147344f78b1",2986),
 "ckpts/SEMamba_advanced.pth":("c5806e24f83514cf99d2711a1a99c0414e2ac390",9127253),
 "LICENSE":("d5baffcced8687f23f6275de558ac7dee74be447",1611),
}

def v6_sem_assets(root,allow_download=False,include_weights=True):
    root=Path(root)
    for name,(digest,size) in V6_SEM_FILES.items():
        if name.endswith(".pth") and not include_weights: continue
        p=root/name
        if not p.is_file():
            if not allow_download: raise FileNotFoundError(p)
            p.parent.mkdir(parents=True,exist_ok=True)
            url=f"https://raw.githubusercontent.com/RoyChao19477/SEMamba/{V6_SEM_REV}/{name}"
            with urllib.request.urlopen(url,timeout=120) as response: data=response.read(size+1)
            if len(data)!=size or hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()!=digest: raise ValueError("SEMamba blob mismatch "+name)
            tmp=p.with_suffix(p.suffix+".partial");tmp.write_bytes(data);tmp.replace(p)
        data=p.read_bytes()
        if len(data)!=size or hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()!=digest: raise ValueError("SEMamba cached asset mismatch "+name)
    return root

def v6_affine_scan(a,b,h):
    """Inclusive scan of h_t=a_t*h_(t-1)+b_t, tensors B,L,D,N.
    Parallel doubling avoids unstable division/cumprod and is differentiable.
    """
    if a.shape!=b.shape or a.ndim!=4 or h.shape!=a[:,0].shape: raise ValueError("Scan shape")
    k=1
    while k<a.shape[1]:
        b=torch.cat((b[:,:k],b[:,k:]+a[:,k:]*b[:,:-k]),1)
        a=torch.cat((a[:,:k],a[:,k:]*a[:,:-k]),1)
        k*=2
    return b+a*h[:,None]

class V6RMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-5):
        super().__init__();self.weight=nn.Parameter(torch.ones(dim));self.eps=eps
    def forward(self,x):
        return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)).to(x.dtype)*self.weight

class V6Mamba1(nn.Module):
    """Exact real-valued Mamba1 parameter layout; portable recurrence or optional fused scan."""
    def __init__(self,d_model,d_state=16,d_conv=4,expand=2,layer_idx=None,**kwargs):
        super().__init__()
        d=int(d_model*expand);r=math.ceil(d_model/16)
        self.d_inner=d;self.d_state=d_state;self.dt_rank=r;self.layer_idx=layer_idx
        self.in_proj=nn.Linear(d_model,2*d,bias=False)
        self.conv1d=nn.Conv1d(d,d,d_conv,groups=d,padding=d_conv-1)
        self.x_proj=nn.Linear(d,r+2*d_state,bias=False)
        self.dt_proj=nn.Linear(r,d,bias=True)
        self.A_log=nn.Parameter(torch.arange(1,d_state+1,dtype=torch.float32).log().repeat(d,1))
        self.D=nn.Parameter(torch.ones(d))
        self.out_proj=nn.Linear(d,d_model,bias=False)
    def _scan(self,u,dt,b,c):
        A=-self.A_log.float().exp()
        # Chunk sequences to bound memory; checkpoint whole sub-batches during training.
        h=u.new_zeros((len(u),self.d_inner,self.d_state),dtype=torch.float32);out=[]
        for i in range(0,u.shape[1],32):
            v=u[:,i:i+32].float();t=dt[:,i:i+32].float()
            a=torch.exp(t[...,None]*A)
            bu=(v*t)[...,None]*b[:,i:i+32,None,:].float()
            states=v6_affine_scan(a,bu,h)
            out.append((states*c[:,i:i+32,None,:].float()).sum(-1))
            h=states[:,-1]
        return torch.cat(out,1).to(u.dtype)
    def _part(self,x):
        u,z=self.in_proj(x).chunk(2,-1)
        u=F.silu(self.conv1d(u.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
        d,b,c=self.x_proj(u).split([self.dt_rank,self.d_state,self.d_state],-1)
        dt=F.softplus(self.dt_proj(d).float())
        # Avoid mandatory CUDA build: use official fused op only if installed and explicitly enabled.
        if x.is_cuda and getattr(self,"use_fused",False):
            from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
            y=selective_scan_fn(u.transpose(1,2).contiguous(),dt.transpose(1,2).contiguous(),-self.A_log.float().exp(),
                b.transpose(1,2).contiguous(),c.transpose(1,2).contiguous(),self.D.float(),z=z.transpose(1,2).contiguous(),
                delta_softplus=False).transpose(1,2)
            return self.out_proj(y)
        y=self._scan(u,dt,b,c)+u*self.D
        return self.out_proj(y*F.silu(z))
    def forward(self,x,inference_params=None):
        if inference_params is not None: raise ValueError("Streaming cache not supported")
        from torch.utils.checkpoint import checkpoint
        ys=[]
        for part in x.split(8 if x.device.type=="cpu" else 16,0):
            if torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()):
                ys.append(checkpoint(self._part,part,use_reentrant=False))
            else: ys.append(self._part(part))
        return torch.cat(ys,0)

class V6MambaBlock(nn.Module):
    def __init__(self,dim,mixer_cls,norm_cls=nn.LayerNorm,fused_add_norm=False,residual_in_fp32=False,mlp_cls=nn.Identity):
        super().__init__()
        if fused_add_norm or mlp_cls is not nn.Identity: raise ValueError("Only unfused Mamba1 identity-MLP blocks supported")
        self.mixer=mixer_cls(dim);self.norm=norm_cls(dim);self.residual_in_fp32=residual_in_fp32
    def forward(self,hidden_states,residual=None,inference_params=None):
        residual=hidden_states if residual is None else hidden_states+residual
        y=self.norm(residual.to(self.norm.weight.dtype))
        return self.mixer(y,inference_params),residual.float() if self.residual_in_fp32 else residual

def v6_sem_net(root):
    import yaml
    from einops import rearrange
    ns={"torch":torch,"nn":nn,"F":F,"rearrange":rearrange,"partial":partial,
        "Mamba":V6Mamba1,"Block":V6MambaBlock,"RMSNorm":V6RMSNorm,
        "_init_weights":lambda module,**kw:None,"__name__":"v6_pinned_sem"}
    specs=[("models/lsigmoid.py",["LearnableSigmoid2D"]),
        ("models/codec_module.py",["get_padding","get_padding_2d","DenseBlock","DenseEncoder","MagDecoder","PhaseDecoder"]),
        ("models/mamba_block.py",["create_block","MambaBlock","TFMambaBlock"]),
        ("models/generator.py",["SEMamba"])]
    for name,expected in specs:
        tree=ast.parse((Path(root)/name).read_text(encoding="utf-8"))
        nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in expected]
        if [n.name for n in nodes]!=expected: raise ValueError("SEMamba source structure mismatch")
        exec(compile(ast.Module(body=nodes,type_ignores=[]),name,"exec"),ns)
    cfg=yaml.safe_load((Path(root)/"ckpts/config.yaml").read_text(encoding="utf-8"))
    st=cfg["stft_cfg"]
    if tuple(st[k] for k in ("sampling_rate","n_fft","hop_size","win_size"))!=(16000,400,100,400) or cfg["model_cfg"]["compress_factor"]!=.3: raise ValueError("SEMamba STFT mismatch")
    return ns["SEMamba"](cfg)

class V6SEMamba(nn.Module):
    def __init__(self,asset_root,allow_download=False,pretrained=True,use_fused=False):
        super().__init__()
        root=v6_sem_assets(asset_root,allow_download,include_weights=pretrained)
        self.net=v6_sem_net(root)
        if pretrained:
            state=torch.load(root/"ckpts/SEMamba_advanced.pth",map_location="cpu",weights_only=True)
            self.net.load_state_dict(state["generator"],strict=True)
        for m in self.net.modules():
            if isinstance(m,V6Mamba1): m.use_fused=bool(use_fused)
    def forward(self,x):
        if x.ndim!=2 or x.shape[-1]<=400: raise ValueError("SEMamba expects B x samples")
        rms=x.square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()
        y=x/rms;w=torch.hann_window(400,device=x.device,dtype=x.dtype)
        z=torch.stft(y,400,100,400,window=w,center=True,pad_mode="reflect",return_complex=True)
        # Official inference uses abs/angle without epsilon; floor prevents undefined gradients.
        mag=z.abs().clamp_min(1e-10).pow(.3);phase=z.angle()
        a,p,_=self.net(mag,phase);amp=a.clamp_min(0).pow(1/.3)
        return torch.istft(torch.complex(amp*p.cos(),amp*p.sin()),400,100,400,window=w,center=True,length=x.shape[-1])*rms

def v6_sem_groups(model):
    n=model.net
    return [("decoder",[n.mask_decoder,n.phase_decoder]),("late",list(n.TSMamba[2:])),
            ("early",list(n.TSMamba[:2])),("encoder",[n.dense_encoder])]
