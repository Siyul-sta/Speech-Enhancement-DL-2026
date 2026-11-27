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
    def __init__(self,train,fit,external,public_noise,size,seed=42,hard=False,curriculum=True,korean=()):
        if train.split!="train": raise ValueError("No evaluation data in noise bank")
        self.train=train;self.fit=np.asarray(fit,dtype=int);self.external=list(external);self.korean=list(korean);self.public_noise=list(public_noise)
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
        corpus=self.korean if self.korean and rng.random()<.5 else self.external
        s=v6_crop(corpus,rng) if rng.random()<.80 else self.train.clean[int(rng.choice(self.fit))].copy()
        g=self.groups[int(rng.choice(len(self.groups),p=self.probs))]
        j=int(rng.choice(g));n=np.roll(self.noise[j],int(rng.integers(len(s))))
        if self.public_noise and rng.random()<.20: n=v6_crop(self.public_noise,rng)
        snr=float(rng.choice(self.snr))
        if self.curriculum: snr+=max(0.,8.-.6*self.epoch)
        snr=float(np.clip(snr+rng.uniform(-2,2),-10,30))
        return v6_mix(s,n,snr)

# Public training-only Korean parquet; pinned dataset revision, no remote code.
V6_ZEROTH_REV="1fe937899f828af822293d05e086200946088bdf"
V6_ZEROTH_LICENSE="CC BY 4.0; Zeroth-Korean, Atlas Guide/Gridspace, OpenSLR SLR40"
V6_DNS_NOISE_URL="https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2"

def v6_zeroth(root,limit):
    from contextlib import closing
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if not 1<=limit<=2000: raise ValueError("Invalid Zeroth subset limit")
    paths=[]
    for shard in range(6):
        quota=limit//6+int(shard<limit%6)
        if not quota: continue
        url=f"https://huggingface.co/datasets/kresnik/zeroth_korean/resolve/{V6_ZEROTH_REV}/data/train-{shard:05d}-of-00006.parquet"
        got=0
        with closing(parquet_audio_rows(url,columns=["audio"])) as rows:
            for i,row in enumerate(rows):
                x=decode_audit_audio(row["audio"])
                if len(x)<48000 or float(np.mean(x*x))<1e-8: continue
                # Save a deterministic 3-second crop; no private data is involved.
                k=(len(x)-48000)//2;x=x[k:k+48000]
                if float(np.mean(x*x))<1e-8: continue
                p=root/f"zeroth_{shard:02d}_{i:06d}.wav"
                sf.write(p,x.clip(-1,1),16000,subtype="PCM_16")
                paths.append(p);got+=1
                if got>=quota: break
        if got!=quota: raise RuntimeError("Incomplete Zeroth train shard")
    return paths

def v6_dns_noise(root,limit=1024):
    """Optional licensed AudioSet noise shard; bounded retained subset, CPU preparation only."""
    import hashlib,tarfile,urllib.request
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    archive=root/"dns_audioset_000.tar.bz2"
    if not archive.exists():
        tmp=archive.with_suffix(".partial")
        h=hashlib.sha256();n=0
        with urllib.request.urlopen(V6_DNS_NOISE_URL,timeout=90) as r,tmp.open("wb") as out:
            while True:
                chunk=r.read(1024*1024)
                if not chunk: break
                n+=len(chunk)
                if n>12*1024**3: raise ValueError("DNS shard exceeds download limit")
                h.update(chunk);out.write(chunk)
        expected=r.headers.get("Content-Length")
        if expected and expected.isdigit() and int(expected)!=n: raise ValueError("Incomplete DNS shard")
        tmp.replace(archive)
    with archive.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
    paths=[]
    with tarfile.open(archive,"r:bz2") as tf:
        for member in tf:
            if not member.isfile() or Path(member.name).suffix.lower()!=".wav" or not 0<member.size<32*1024*1024: continue
            if int(hashlib.sha256(member.name.encode()).hexdigest()[:8],16)%4: continue
            stream=tf.extractfile(member)
            if stream is None: continue
            # Never extract archive paths; use generated safe names.
            import io,scipy.signal
            try: x,sr=sf.read(io.BytesIO(stream.read()),dtype="float32",always_2d=True)
            except (OSError,RuntimeError,ValueError): continue
            x=x.mean(1)
            if sr!=16000:
                g=math.gcd(sr,16000);x=scipy.signal.resample_poly(x,16000//g,sr//g)
            if len(x)<48000 or not np.isfinite(x).all(): continue
            x=x[:48000]
            if float(np.mean(x*x))<1e-8: continue
            p=root/f"dns_{len(paths):05d}.wav";sf.write(p,x.clip(-1,1),16000,subtype="PCM_16");paths.append(p)
            if len(paths)>=limit: break
    if len(paths)<32: raise RuntimeError("DNS noise shard did not yield usable clips")
    archive.unlink(missing_ok=True)
    return paths,digest

def v6_cache_extra(root,korean_limit,dns):
    """Public-only CPU cache alongside V4 cache manifest."""
    import hashlib
    root=Path(root);out=root/"v6_extra";out.mkdir(parents=True,exist_ok=True)
    files={};meta={"protocol":"v6-extra-cache-v1","zeroth_revision":V6_ZEROTH_REV,"zeroth_license":V6_ZEROTH_LICENSE}
    if korean_limit:
        paths=v6_zeroth(out/"korean",korean_limit)
        for p in paths: files[p.relative_to(out).as_posix()]=v6_sha(p)
    if dns:
        paths,digest=v6_dns_noise(out/"noise")
        meta.update(dns_url=V6_DNS_NOISE_URL,dns_archive_sha256=digest,dns_license="AudioSet CC BY 4.0; original source terms apply")
        for p in paths: files[p.relative_to(out).as_posix()]=v6_sha(p)
    meta["files"]=files
    (out/"manifest.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return meta

def v6_read_extra_cache(ref):
    if not ref: return [],[]
    slug=ref.rsplit("/",1)[-1]
    hits=[p for p in Path("/kaggle/input").rglob("manifest.json") if p.parent.name=="v6_extra" and slug in p.parts]
    if not hits: return [],[]
    if len(hits)!=1: raise ValueError("Ambiguous V6 external cache")
    p=hits[0];meta=json.loads(p.read_text(encoding="utf-8"))
    if meta.get("protocol")!="v6-extra-cache-v1" or meta.get("zeroth_revision")!=V6_ZEROTH_REV: raise ValueError("V6 cache identity")
    korean=[];noise=[]
    for name,h in sorted(meta["files"].items()):
        f=(p.parent/name).resolve()
        if not f.is_relative_to(p.parent.resolve()) or not f.is_file() or v6_sha(f)!=h: raise ValueError("V6 cache integrity")
        if name.startswith("korean/"): korean.append(f)
        elif name.startswith("noise/"): noise.append(f)
        else: raise ValueError("Unrecognized extra cache member")
    return korean,noise
