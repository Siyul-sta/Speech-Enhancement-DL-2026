"""V7 training-only, bounded research operators. No validation data or inference changes."""
import math
import numpy as np
import torch

def v7_spectral_balance(pred,clean):
    """Relative compressed magnitude error, equal weight across speech bands."""
    total=pred.new_zeros(())
    for n,h in ((256,64),(512,128),(1024,256)):
        w=torch.hann_window(n,device=pred.device,dtype=pred.dtype)
        a=torch.stft(pred,n,h,window=w,return_complex=True).abs()
        b=torch.stft(clean,n,h,window=w,return_complex=True).abs()
        # Smooth power compression; finite derivative at zero.
        a=(a+.001).pow(.3);b=(b+.001).pow(.3)
        edges=[0,n//32,n//8,n//4,n//2+1]
        for lo,hi in zip(edges,edges[1:]):
            aa,bb=a[:,lo:hi],b[:,lo:hi]
            scale=bb.mean((-2,-1)).detach().clamp_min(.15)
            total=total+((aa-bb).abs().mean((-2,-1))/scale).mean()
    return total/12

def v7_residual_trust(pred,coarse,noisy):
    """Discourage large corrections to a good frozen parent; noisy scale only."""
    scale=noisy.square().mean(-1).detach().clamp_min(1e-8).sqrt()
    delta=(pred-coarse.detach()).abs().mean(-1)/scale
    return torch.nn.functional.smooth_l1_loss(delta,torch.zeros_like(delta),beta=.1)

def v7_critic_gate(calibration,updates,max_weight=.025):
    if not calibration or updates<16:return 0.
    p=calibration.get("pearson");m=calibration.get("mae")
    if p is None or m is None or not math.isfinite(p) or not math.isfinite(m) or p<.6 or m>.5:return 0.
    if max_weight not in (.015,.025,.035):raise ValueError("Unapproved critic weight")
    return max_weight

def v7_critic_warmup(model,critic,opt,replay,train,cal_ids,cal,device,pesq_fn,seed,count):
    """Train fit-only warmup, then independent Train calibration. No validation target."""
    if train.split!="train" or count not in (24,48):raise ValueError("Invalid critic warmup")
    fit,expected,gate_fit,gate_select=v6_split(train)
    if set(cal_ids)&set(fit) or not set(cal_ids)<=set(expected):raise ValueError("Critic pool overlap")
    rng=np.random.default_rng(seed+701)
    ids=rng.choice(fit,count,replace=False).tolist()
    model.eval()
    for i in ids:
        x,s=train[i];x=x[None].to(device);s=s[None].to(device)
        with torch.no_grad():y=model(x)
        critic_replay_step(critic,opt,replay,y,x,s,pesq_fn)
    calibration=critic_calibration(model,critic,cal,device,pesq_fn)
    return count,calibration
