"""V5.5 noisy-only ensemble primitives; no downloads or reference at inference."""
import torch
from torch import nn

NFFT,HOP=512,128
BANDS=4

def band_index(device):
    hz=torch.arange(NFFT//2+1,device=device)*16000/NFFT
    return torch.bucketize(hz,torch.tensor([500.,1500.,3500.],device=device))

def noisy_features(x):
    # Scale-independent spectral summaries plus overall level, no metadata.
    x=x.float()
    rms=x.square().mean(-1).sqrt().clamp_min(1e-6)
    zcr=(x[:,1:]*x[:,:-1]<0).float().mean(-1)
    crest=(x.abs().amax(-1)/rms).clamp_max(50).log1p()
    window=torch.hann_window(NFFT,device=x.device,dtype=x.dtype)
    mag=torch.stft(x,NFFT,HOP,window=window,return_complex=True).abs()
    power=mag.square().mean(-1)
    p=power.sum(-1).clamp_min(1e-8)
    idx=band_index(x.device)
    ratios=torch.stack([power[:,idx==i].sum(-1)/p for i in range(BANDS)],-1)
    flat=((power+1e-8).log().mean(-1).exp()/(power.mean(-1)+1e-8)).clamp(0,1)
    return torch.cat((rms.log().clamp(-14,2)[:,None],zcr[:,None],crest[:,None],flat[:,None],ratios),-1)

class EnsembleGate(nn.Module):
    """One simplex per clip or four coarse frequency bands."""
    def __init__(self,experts,mode):
        super().__init__()
        if not 2<=experts<=5 or mode not in ("clip","frequency"): raise ValueError("Invalid ensemble gate")
        self.experts=experts
        self.mode=mode
        self.bands=1 if mode=="clip" else BANDS
        self.net=nn.Sequential(nn.Linear(8,24),nn.SiLU(),nn.Linear(24,experts*self.bands))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,noisy):
        return self.net(noisy_features(noisy)).reshape(-1,self.experts,self.bands).softmax(dim=1)

def mix_outputs(noisy,predictions,weights):
    # predictions B,M,T; weights B,M,1 or B,M,4. Same synthesis for fit and inference.
    if predictions.ndim!=3 or predictions.shape[0]!=noisy.shape[0] or predictions.shape[-1]!=noisy.shape[-1] or weights.shape[:2]!=predictions.shape[:2]:
        raise ValueError("Invalid ensemble tensor shapes")
    if weights.shape[-1]==1: return (predictions*weights).sum(1)
    if weights.shape[-1]!=BANDS: raise ValueError("Invalid frequency gate")
    b,m,n=predictions.shape
    win=torch.hann_window(NFFT,device=predictions.device,dtype=predictions.dtype)
    z=torch.stft(predictions.reshape(b*m,n),NFFT,HOP,window=win,return_complex=True)
    z=z.reshape(b,m,*z.shape[-2:])
    w=weights[:,:,band_index(z.device)]
    y=(z*w[...,None]).sum(1)
    return torch.istft(y,NFFT,HOP,window=win,length=n)

class GatedEnsemble(nn.Module):
    def __init__(self,models,gate):
        super().__init__()
        if len(models)!=gate.experts: raise ValueError("Expert count mismatch")
        self.experts=nn.ModuleList(models).eval().requires_grad_(False)
        self.gate=gate
    def forward(self,noisy):
        self.experts.eval()
        with torch.no_grad():
            predictions=torch.stack([m(noisy) for m in self.experts],1)
        if not torch.isfinite(predictions).all(): raise RuntimeError("Invalid expert output")
        return mix_outputs(noisy,predictions,self.gate(noisy))

def fit_gate(x,pred,clean,mode,epochs=12,seed=55,batch=4):
    # Only caller-provided train-fit predictions; experts remain frozen.
    if not 1<=epochs<=100 or len(x)!=len(pred) or len(x)!=len(clean) or not len(x): raise ValueError("Invalid fit inputs")
    torch.manual_seed(seed)
    gate=EnsembleGate(pred.shape[1],mode).to(x.device)
    opt=torch.optim.AdamW(gate.parameters(),lr=.003,weight_decay=1e-4)
    gen=torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        gate.train()
        for ids in torch.randperm(len(x),generator=gen).split(batch):
            ids=ids.to(x.device)
            a,p,c=x[ids],pred[ids],clean[ids]
            w=gate(a)
            y=mix_outputs(a,p,w)
            # Normalize by clean RMS to avoid louder clips dominating.
            loss=((y-c).abs().mean(-1)/c.square().mean(-1).sqrt().clamp_min(.001)).mean()
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite gate loss")
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(),5.,error_if_nonfinite=True)
            opt.step()
    return gate.eval()
