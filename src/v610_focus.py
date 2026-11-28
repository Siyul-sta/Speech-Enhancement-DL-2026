"""V6.10 optional training-only objectives; inserted into archived V6 runner.
No model topology, forward path, evaluator or validation split changes.
"""
import torch

def v610_phase_derivatives(pred,clean):
    """Energy-weighted circular errors of phase, group delay and instantaneous frequency."""
    total=pred.new_zeros(())
    eps=1e-4
    for n,h in ((256,64),(512,128),(1024,256)):
        win=torch.hann_window(n,device=pred.device,dtype=pred.dtype)
        a=torch.stft(pred,n,h,window=win,return_complex=True)
        b=torch.stft(clean,n,h,window=win,return_complex=True)
        am,bm=a.abs(),b.abs()
        u=a/(am.square()+eps**2).sqrt()
        v=b/(bm.square()+eps**2).sqrt()
        def circular(x,y,w):
            w=w.detach()
            d=(1-(x*y.conj()).real).clamp(0,2)
            return ((d*w).sum((-2,-1))/w.sum((-2,-1)).clamp_min(eps)).mean()
        ip=circular(u,v,bm)
        # Phase differences expressed as products of unit complex values.
        gd=circular(u[:,1:]*u[:,:-1].conj(),v[:,1:]*v[:,:-1].conj(),torch.minimum(bm[:,1:],bm[:,:-1]))
        inst=circular(u[:,:,1:]*u[:,:,:-1].conj(),v[:,:,1:]*v[:,:,:-1].conj(),torch.minimum(bm[:,:,1:],bm[:,:,:-1]))
        total=total+(.4*ip+.3*gd+.3*inst)
    return total/3

def v610_noise_weights(mix,train,cal_ids,rows):
    """Initialize group hard weights BEFORE the first training epoch, from Train calibration."""
    if train.split!="train":raise ValueError("V6.10 calibration must be train")
    if not mix.hard:raise ValueError("V6.10 hard weighting disabled")
    ids=list(map(int,cal_ids))
    if len(ids)!=len(rows) or set(ids)&set(map(int,mix.fit)):raise ValueError("Invalid disjoint train calibration")
    key=next((k for k in ("noise_type","noise_category","noise_label","noise_class") if k in train.manifest.columns),None)
    groups={}
    for expected,(i,v) in enumerate(rows):
        if int(i)!=expected or not math.isfinite(float(v)):raise ValueError("Invalid train calibration score")
        label=str(train.manifest.iloc[ids[expected]][key]) if key else "unknown"
        groups.setdefault(label,[]).append(float(v))
    # Shrink tiny groups toward overall calibration mean, so 1 clip cannot dominate.
    all_scores=[v for vals in groups.values() for v in vals]
    center=float(np.mean(all_scores))
    means={k:(sum(v)+3*center)/(len(v)+3) for k,v in groups.items()}
    mix.update_hard(means)
    return {"counts":{k:len(v) for k,v in groups.items()},"shrunk_scores":means,
        "probabilities":dict(zip(mix.group_names,map(float,mix.probs)))}
