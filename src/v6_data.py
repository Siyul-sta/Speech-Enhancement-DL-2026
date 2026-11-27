"""V6 train-only noise bank, cross-corpus mixing and deterministic curriculum.
Embedded into the Kaggle trainer; no external/private audio upload.
"""
import math
import re
import numpy as np
import torch
from torch.utils.data import Dataset
import soundfile as sf

def v6_speaker(row):
    for k in ("speaker_id","speaker","speaker_name"):
        v=str(row.get(k,"")).strip().lower()
        if v and v!="nan": return v
    m=re.match(r"(speaker[0-9]+)(?:_|$)",str(row.get("segment_id","")),re.I)
    return m.group(1).lower() if m else None

def v6_split(train,seed=6026):
    if train.split!="train" or len(train)!=1296: raise ValueError("Original train split required")
    groups={}
    for i,row in enumerate(train.manifest.to_dict("records")):
        name=v6_speaker(row)
        if name is None: raise ValueError("V6 requires known train speaker IDs")
        groups.setdefault(name,[]).append(i)
    if len(groups)<8: raise ValueError("Too few train speakers")
    names=sorted(groups)
    rng=np.random.default_rng(seed);rng.shuffle(names)
    # Two speaker-disjoint calibration groups, one ensemble fit, one select.
    pools=[]
    for i in range(3):
        ids=[]
        while len(ids)<32 and names: ids+=groups[names.pop()]
        if len(ids)<32: raise ValueError("Insufficient calibration speakers")
        pools.append(sorted(ids))
    fit=sorted(i for n in names for i in groups[n])
    if len(fit)<512: raise ValueError("Insufficient remaining training clips")
    return fit,*pools

def v6_crop(paths,rng,length=48000):
    from scipy.signal import resample_poly
    for _ in range(16):
        p=paths[int(rng.integers(len(paths)))]
        with sf.SoundFile(str(p)) as f:
            sr=f.samplerate;need=math.ceil(length*sr/16000)
            if len(f)<need: continue
            f.seek(int(rng.integers(len(f)-need+1)))
            x=f.read(need,dtype="float32",always_2d=True).mean(axis=1)
        if sr!=16000:
            g=math.gcd(sr,16000);x=resample_poly(x,16000//g,sr//g)
        x=np.asarray(x[:length],np.float32)
        if len(x)==length and np.isfinite(x).all() and float(np.mean(x*x))>1e-8: return x
    raise RuntimeError("No usable external audio crop after 16 attempts")

def v6_mix(clean,noise,snr):
    s=np.asarray(clean,np.float32);n=np.asarray(noise,np.float32)
    if s.shape!=n.shape or not np.isfinite(s).all() or not np.isfinite(n).all(): raise ValueError("Invalid mix inputs")
    sp=float(np.mean(s*s,dtype=np.float64));npow=float(np.mean(n*n,dtype=np.float64))
    if min(sp,npow)<1e-12: raise ValueError("Silent source")
    n=n*math.sqrt(sp/npow)*10**(-float(snr)/20)
    x=s+n;scale=max(1.,float(np.max(np.abs(x))),float(np.max(np.abs(s))))
    return torch.from_numpy(np.asarray(x/scale,np.float32)),torch.from_numpy(np.asarray(s/scale,np.float32))

class V6CrossMix(Dataset):
    def __init__(self,train,fit,external,public_noise,size,seed=42,hard=False,curriculum=True):
        if train.split!="train": raise ValueError("No evaluation data in noise bank")
        self.train=train;self.fit=np.asarray(fit,dtype=int);self.external=list(external);self.public_noise=list(public_noise)
        if len(set(self.fit))!=len(self.fit) or self.fit.min()<0 or self.fit.max()>=len(train): raise ValueError("Invalid fit IDs")
        self.noise=train.noisy[self.fit]-train.clean[self.fit]
        clean=train.clean[self.fit]
        sp=np.mean(clean*clean,axis=1,dtype=np.float64);npow=np.mean(self.noise*self.noise,axis=1,dtype=np.float64)
        corr=np.abs(np.mean(clean*self.noise,axis=1,dtype=np.float64))/np.maximum(np.sqrt(sp*npow),1e-12)
        pool=np.flatnonzero((sp>1e-8)&(npow>1e-10)&(corr<.15)&np.isfinite(corr))
        if len(pool)<32: raise ValueError("Too few valid additive residuals")
        self.pool=pool;self.snr=np.clip(10*np.log10(sp[pool]/npow[pool]),-10,30)
        key=next((k for k in ("noise_type","noise_category","noise_label","noise_class") if k in train.manifest.columns),None)
        labels=train.manifest.iloc[self.fit][key].fillna("unknown").astype(str).to_numpy() if key else np.full(len(self.fit),"unknown")
        self.labels=labels
        self.groups=[pool[labels[pool]==label] for label in sorted(set(labels[pool]))]
        self.group_names=[str(labels[g[0]]) for g in self.groups]
        self.base=np.sqrt(np.array([len(g) for g in self.groups],dtype=float))
        self.probs=self.base/self.base.sum();self.size=int(size);self.seed=int(seed);self.epoch=0
        self.hard=bool(hard);self.curriculum=bool(curriculum)
        if not 1<=self.size<=1000000 or not self.external: raise ValueError("Invalid V6 mix size/corpus")
    def update_hard(self,group_scores):
        # Actual train-only holdout PESQ; high weight for low scores, bounded 2x.
        if not self.hard: return
        factors=np.array([np.clip((3.5-float(group_scores.get(k,2.5))),.5,2.) for k in self.group_names])
        w=self.base*factors;self.probs=w/w.sum()
    def __len__(self): return self.size
    def __getitem__(self,i):
        if not 0<=i<self.size: raise IndexError(i)
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,int(i)]))
        s=v6_crop(self.external,rng) if rng.random()<.80 else self.train.clean[int(rng.choice(self.fit))].copy()
        g=self.groups[int(rng.choice(len(self.groups),p=self.probs))]
        j=int(rng.choice(g));n=np.roll(self.noise[j],int(rng.integers(len(s))))
        if self.public_noise and rng.random()<.20: n=v6_crop(self.public_noise,rng)
        snr=float(rng.choice(self.snr))
        if self.curriculum: snr+=max(0.,8.-.6*self.epoch)
        snr=float(np.clip(snr+rng.uniform(-2,2),-10,30))
        return v6_mix(s,n,snr)
