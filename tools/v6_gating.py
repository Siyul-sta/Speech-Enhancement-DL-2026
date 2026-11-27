"""V6.5 noisy-only bounded alignment, complex time/frequency gating and train-only selection."""
import torch
from torch import nn

NFFT=512
HOP=128

def v6_align(x,p,mode="none"):
    """B,M,T predictions; choose alignment using noisy input and expert outputs only."""
    if mode=="none": return p
    if mode not in ("rms","lag_rms"): raise ValueError("Unknown alignment")
    if p.ndim!=3 or p.shape[0]!=len(x) or p.shape[-1]!=x.shape[-1]: raise ValueError("Prediction shape")
    if mode=="lag_rms":
        n=p.shape[-1];maxlag=min(32,n//8)
        # FFT cross correlation. Only shift if normalized correlation improves by >=.02.
        size=1<<(2*n-1).bit_length()
        a=torch.fft.rfft(x, n=size)[:,None]
        b=torch.fft.rfft(p,n=size)
        c=torch.fft.irfft(a*b.conj(),n=size)
        corr=torch.cat((c[...,-maxlag:],c[...,:maxlag+1]),-1)
        norm=(x.square().sum(-1)[:,None]*p.square().sum(-1)).clamp_min(1e-12).sqrt()
        corr=corr/norm[...,None]
        idx=corr.argmax(-1);best=corr.gather(-1,idx[...,None]).squeeze(-1)
        lag=torch.where(best-corr[...,maxlag]>=.02,idx-maxlag,torch.zeros_like(idx))
        t=torch.arange(n,device=p.device)[None,None,:]-lag[...,None]
        shifted=p.gather(-1,t.clamp(0,n-1).expand_as(p))
        p=shifted*((t>=0)&(t<n))
    rms=p.square().mean(-1,keepdim=True).clamp_min(1e-12).sqrt()
    # Align levels to median expert, not to clean or noisy energy; bounded gain.
    target=rms.median(1,keepdim=True).values
    return p*(target/rms).clamp(.25,4.)

def v6_features(x):
    # Local noisy-only frequency summaries on same STFT grid.
    win=torch.hann_window(NFFT,device=x.device,dtype=x.dtype)
    z=torch.stft(x,NFFT,HOP,window=win,return_complex=True)
    power=z.abs().square()
    bands=torch.bucketize(torch.arange(NFFT//2+1,device=x.device)*16000/NFFT,torch.tensor([500.,1500.,3500.],device=x.device))
    total=power.sum(1).clamp_min(1e-10)
    frac=torch.stack([power[:,bands==i].sum(1)/total for i in range(4)],1)
    energy=total.log().clamp(-20,20)[:,None]
    flat=((power+1e-8).log().mean(1).exp()/(power.mean(1)+1e-8)).clamp(0,1)[:,None]
    return torch.cat((energy,flat,frac),1),bands

class V6Gate(nn.Module):
    def __init__(self,experts,mode):
        super().__init__()
        if not 2<=experts<=5 or mode not in ("clip","frequency","timefreq"): raise ValueError("Invalid gate")
        self.experts=experts;self.mode=mode
        self.bands=1 if mode=="clip" else 4
        self.net=nn.Sequential(nn.Conv1d(6,24,5,padding=2),nn.SiLU(),nn.Conv1d(24,experts*self.bands,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        feat,_=v6_features(x)
        y=self.net(feat).reshape(len(x),self.experts,self.bands,-1)
        if self.mode!="timefreq": y=y.mean(-1,keepdim=True)
        return y.softmax(1)

def v6_combine(x,p,w):
    if p.ndim!=3 or w.ndim!=4 or w.shape[:2]!=p.shape[:2] or p.shape[0]!=len(x): raise ValueError("Bad gate dimensions")
    if w.shape[2:]==(1,1): return (p*w.squeeze(-1)).sum(1)
    b,m,n=p.shape;win=torch.hann_window(NFFT,device=x.device,dtype=x.dtype)
    z=torch.stft(p.reshape(b*m,n),NFFT,HOP,window=win,return_complex=True).reshape(b,m,NFFT//2+1,-1)
    _,bands=v6_features(x)
    if w.shape[2]!=4 or w.shape[-1] not in (1,z.shape[-1]): raise ValueError("Bad frequency/time gate")
    return torch.istft((z*w[:,:,bands,:]).sum(1),NFFT,HOP,window=win,length=n)

def v6_fit_gate(x,p,c,mode,epochs=12):
    torch.manual_seed(6026);gate=V6Gate(p.shape[1],mode)
    opt=torch.optim.AdamW(gate.parameters(),lr=.003,weight_decay=1e-4)
    for epoch in range(epochs):
        for ids in torch.randperm(len(x),generator=torch.Generator().manual_seed(6026+epoch)).split(4):
            a,y,s=x[ids],p[ids],c[ids]
            pred=v6_combine(a,y,gate(a))
            loss=((pred-s).abs().mean(-1)/s.square().mean(-1).sqrt().clamp_min(.001)).mean()
            if not torch.isfinite(loss): raise ValueError("Gate loss nonfinite")
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(),5.,error_if_nonfinite=True);opt.step()
    return gate.eval()

class V6Ensemble(nn.Module):
    def __init__(self,models,alignment="none",weights=None,gate=None):
        super().__init__()
        self.experts=nn.ModuleList(models).eval().requires_grad_(False);self.alignment=alignment;self.gate=gate
        if (weights is None)==(gate is None): raise ValueError("Exactly one gate or fixed weight")
        if weights is not None:
            w=torch.as_tensor(weights,dtype=torch.float32)
            if w.ndim!=1 or len(w)!=len(models) or not torch.isfinite(w).all() or (w<0).any() or abs(float(w.sum())-1)>1e-6: raise ValueError("Invalid ensemble weights")
            self.register_buffer("weights",w)
    def forward(self,x):
        self.experts.eval()
        with torch.no_grad():
            p=torch.stack([m(x) for m in self.experts],1)
        if p.shape!=(len(x),len(self.experts),x.shape[-1]) or not torch.isfinite(p).all(): raise ValueError("Expert contract")
        p=v6_align(x,p,self.alignment)
        w=self.gate(x) if self.gate is not None else self.weights[None,:,None,None].expand(len(x),-1,-1,-1)
        return v6_combine(x,p,w)
