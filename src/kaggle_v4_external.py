"""V3: two-head magnitude attenuation plus residual phase correction.

Hypothesis: separate magnitude reduction from a bounded phase-only residual
so the phase head cannot directly amplify noisy magnitudes.
Input: noisy mono 16 kHz WAV only, output: same-length enhanced waveform.
Train paired noisy/clean, validation clean for evaluation ONLY (164 files).
Standalone script: model, dataset loader, training/evaluation and artifacts.
No private WAVs are copied into outputs.
"""
from __future__ import annotations

import csv
import math
import importlib
import json
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

EXPERIMENT_CONFIG = {"id": "V3_RESIDUAL_PHASE", "epochs": 25, "batch_size": 4, "patience": 7, "seed": 42}
DATASET_SLUG = "PRIVATE_DATASET_SLUG"
SAMPLE_RATE = 16000
N_FFT = 512
HOP = 128
WORK = Path("/kaggle/working") if Path("/kaggle").exists() else Path("local_test_output")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def locate_data_root() -> Path:
    """Private Kaggle dataset contains one speech-data.zip or unpacked train/val."""
    if DATASET_SLUG == "PRIVATE_DATASET_SLUG":
        raise RuntimeError("Configure DATASET_SLUG using tools/kaggle_loop.py first.")
    slug = DATASET_SLUG.rsplit("/", 1)[-1]
    mount = Path("/kaggle/input") / slug
    if not mount.exists():
        # On some Kaggle runs datasets may be mounted under another parent.
        # Print a bounded diagnostic before failing rather than silently blaming CUDA.
        input_root = Path("/kaggle/input")
        top_levels = sorted(p.name for p in input_root.iterdir()) if input_root.exists() else []
        print(f"DATASET_MOUNT_DIAGNOSTIC expected={mount} mounted_roots={top_levels[:30]}", flush=True)
        if input_root.exists():
            for manifest in input_root.rglob("train/samples.csv"):
                candidate = manifest.parent.parent
                if (candidate / "validation/samples.csv").is_file():
                    print(f"DATASET_MOUNT_DIAGNOSTIC found_alternative_root={candidate}", flush=True)
                    return candidate
        raise FileNotFoundError(
            f"Private dataset not mounted in Kaggle run: {mount}. "
            "Check notebook Inputs/Add Input and kernel-metadata.json dataset_sources."
        )
    if (mount / "train/samples.csv").is_file() and (mount / "validation/samples.csv").is_file():
        return mount
    matches = list(mount.rglob("train/samples.csv"))
    for path in matches:
        root = path.parent.parent
        if (root / "validation/samples.csv").is_file():
            return root

    archives = sorted(mount.glob("*.zip"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one .zip containing train/ and validation/ in {mount}; found {len(archives)}")
    target = Path("/tmp/ralph_v4_private_speech") if Path("/kaggle").exists() else WORK / "speech_data"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives[0]) as zf:
        for member in zf.infolist():
            resolved = (target / member.filename).resolve()
            if not resolved.is_relative_to(target.resolve()):
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        zf.extractall(target)
    for p in [target] + [x.parent.parent for x in target.rglob("train/samples.csv")]:
        if (p / "train/samples.csv").is_file() and (p / "validation/samples.csv").is_file():
            return p
    raise FileNotFoundError("ZIP must contain train/samples.csv and validation/samples.csv")


class SpeechPairs(Dataset):
    def __init__(self, root: Path, split: str):
        manifest = pd.read_csv(root / split / "samples.csv")
        for col in ("noisy_path", "clean_path", "segment_id"):
            if col not in manifest.columns:
                raise ValueError(f"Missing {col} from {split}/samples.csv")
        self.manifest = manifest
        self.split=split
        self.noisy = np.empty((len(manifest), 48000), np.float32)
        self.clean = np.empty((len(manifest), 48000), np.float32)
        for idx, row in enumerate(manifest.itertuples(index=False)):
            for col, dest in (("noisy_path", self.noisy), ("clean_path", self.clean)):
                path = (root / getattr(row, col)).resolve()
                if not path.is_relative_to(root.resolve()):
                    raise ValueError("Manifest references file outside the dataset root")
                y, sr = sf.read(path, dtype="float32", always_2d=True)
                if sr != SAMPLE_RATE or y.shape != (48000, 1):
                    raise ValueError(f"Expected 16 kHz, mono, 48000 samples: {path}")
                dest[idx] = y[:, 0]
        print(f"{split}: {len(manifest)} pairs loaded", flush=True)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, i):
        return torch.from_numpy(self.noisy[i]), torch.from_numpy(self.clean[i])


class ChallengeRemix(Dataset):
    """Train-only empirical residual remix; reject suspicious correlated residuals."""
    def __init__(self, train, size, seed=42, *, empirical=False, balanced=False, exclude=()):
        if not isinstance(train,SpeechPairs) or train.split!="train" or len(train)!=1296:
            raise ValueError("Challenge remix requires original training split")
        self.clean=train.clean
        self.noise=train.noisy-train.clean
        rms=lambda a: np.sqrt(np.mean(a*a,axis=1,dtype=np.float64))
        cr,nr=rms(self.clean),rms(self.noise)
        cross=np.abs(np.mean(self.clean*self.noise,axis=1,dtype=np.float64))
        corr=cross/np.maximum(cr*nr,1e-12)
        self.pool=np.flatnonzero((cr>1e-4)&(nr>1e-5)&(corr<0.15)&np.isfinite(corr))
        self.pool=np.setdiff1d(self.pool,np.asarray(exclude,dtype=int))
        if len(self.pool)<32: raise RuntimeError("Insufficient additive-like train residuals")
        self.empirical=bool(empirical)
        self.noise_groups=[]
        if balanced:
            key=next((k for k in ("noise_type","noise_category","noise_label","noise_class") if k in train.manifest.columns),None)
            if key is None: raise ValueError("Noise balancing needs actual train metadata")
            labels=train.manifest[key].fillna("unknown").astype(str).to_numpy()
            self.noise_groups=[self.pool[labels[self.pool]==label] for label in sorted(set(labels[self.pool]))]
            # Mild sqrt-frequency balancing, not uniform oversampling of three-clip classes.
            counts=np.sqrt([len(g) for g in self.noise_groups]); self.noise_group_probs=counts/counts.sum()
        self.snr=20*np.log10(cr[self.pool]/nr[self.pool])
        self.snr_low,self.snr_high=np.clip(np.quantile(self.snr,[.05,.95]),-15,35)
        if not np.isfinite(self.snr).all(): raise ValueError("Invalid train SNR")
        self.size,self.seed,self.epoch=int(size),int(seed),0
        print("CHALLENGE_REMIX",len(self.pool),float(self.snr_low),float(self.snr_high),flush=True)
    def __len__(self): return self.size
    def __getitem__(self,i):
        rng=np.random.default_rng(self.seed+100003*self.epoch+i)
        a,b=rng.choice(self.pool,2,replace=False)
        if self.noise_groups:
            g=self.noise_groups[rng.choice(len(self.noise_groups),p=self.noise_group_probs)]
            choices=g[g!=a]
            if len(choices): b=rng.choice(choices)
        s=self.clean[a].copy()
        n=np.roll(self.noise[b],int(rng.integers(0,len(s))))
        snr=float(np.clip(rng.choice(self.snr),-15,35)) if self.empirical else rng.uniform(self.snr_low,self.snr_high)
        n=n*(np.sqrt(np.mean(s*s)+1e-12)/np.sqrt(np.mean(n*n)+1e-12))*10**(-snr/20)
        x=s+n
        scale=max(1.,float(np.max(np.abs(x))),float(np.max(np.abs(s))))
        return torch.from_numpy((x/scale).astype(np.float32)),torch.from_numpy((s/scale).astype(np.float32))


class ConvBlock(nn.Module):
    def __init__(self, input_ch: int, output_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, 3, padding=1),
            nn.GroupNorm(min(8, output_ch), output_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_ch, output_ch, 3, padding=1),
            nn.GroupNorm(min(8, output_ch), output_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualPhaseUNet(nn.Module):
    """Separate sigmoid magnitude attenuation and zero-initialized phase residual.

    Magnitude head: ratio in (0,1) as in the original U-Net.
    Phase head: bounded correction in (-0.5, +0.5) radians.
    Both heads operate on shared log-magnitude plus normalized phase features.
    """

    max_gain = 1.0
    max_phase = 0.5

    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.middle = ConvBlock(64, 96)
        self.dec3 = ConvBlock(96 + 64, 64)
        self.dec2 = ConvBlock(64 + 32, 32)
        self.dec1 = ConvBlock(32 + 16, 16)
        self.mask_head = nn.Conv2d(16, 1, 1)
        self.phase_head = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)
        self.register_buffer("window", torch.hann_window(N_FFT))

    @staticmethod
    def upcat(x, shortcut):
        return torch.cat(
            [F.interpolate(x, size=shortcut.shape[-2:], mode="bilinear", align_corners=False), shortcut],
            dim=1,
        )

    def forward(self, noisy):
        spec = torch.stft(
            noisy, n_fft=N_FFT, hop_length=HOP, window=self.window, return_complex=True
        )
        magnitude = spec.abs()
        features = torch.stack(
            (torch.log1p(magnitude),
             spec.real / (magnitude + 1.0),
             spec.imag / (magnitude + 1.0)),
            dim=1,
        )
        a = self.enc1(features)
        b = self.enc2(F.avg_pool2d(a, 2, ceil_mode=True))
        c = self.enc3(F.avg_pool2d(b, 2, ceil_mode=True))
        d = self.middle(F.avg_pool2d(c, 2, ceil_mode=True))
        d = self.dec3(self.upcat(d, c))
        d = self.dec2(self.upcat(d, b))
        d = self.dec1(self.upcat(d, a))
        gain = self.max_gain * torch.sigmoid(self.mask_head(d)).squeeze(1)
        phase_delta = self.max_phase * torch.tanh(self.phase_head(d)).squeeze(1)
        phase_factor = torch.polar(torch.ones_like(phase_delta), phase_delta)
        enhanced_spec = self.spectrum(spec,d,gain,phase_factor)
        return torch.istft(
            enhanced_spec, n_fft=N_FFT, hop_length=HOP,
            window=self.window, length=noisy.shape[-1],
        )

    def spectrum(self,spec,features,gain,phase_factor):
        return spec*gain*phase_factor


class RelaxedPhaseUNet(ResidualPhaseUNet):
    """Experimental: gains up to 2 and full wrapped phase range."""
    max_gain = 2.0
    max_phase = 3.141592653589793


class ComplexResidualUNet(RelaxedPhaseUNet):
    """Additive real/imaginary residual, independent of a nonzero noisy bin."""
    def __init__(self):
        super().__init__()
        self.complex_head=nn.Conv2d(16,2,1)
        nn.init.zeros_(self.complex_head.weight)
        nn.init.zeros_(self.complex_head.bias)
        # Retain base module structure but use only the direct complex head.
        self.mask_head.requires_grad_(False)
        self.phase_head.requires_grad_(False)
    def spectrum(self,spec,features,gain,phase_factor):
        delta=self.complex_head(features)
        scale=spec.abs().mean(dim=(-2,-1),keepdim=True).detach().clamp_min(.01)
        return spec+torch.complex(delta[:,0],delta[:,1])*scale


class DualAxisEnhancer(nn.Module):
    """Native dual-axis recurrent spectrogram network; not official CMGAN weights."""
    def __init__(self):
        super().__init__()
        w=24
        self.enc=nn.Sequential(nn.Conv2d(3,w,3,padding=1),nn.GroupNorm(4,w),nn.SiLU(),
                               nn.Conv2d(w,w,3,stride=(2,2),padding=1),nn.SiLU())
        self.time=nn.GRU(w,w,batch_first=True,bidirectional=True)
        self.freq=nn.GRU(2*w,w,batch_first=True,bidirectional=True)
        self.head=nn.Conv2d(2*w,2,1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.register_buffer("window",torch.hann_window(N_FFT))
    def forward(self,x):
        y=torch.stft(x,N_FFT,HOP,window=self.window,return_complex=True)
        mag=y.abs()
        z=self.enc(torch.stack((mag.log1p(),y.real/(mag+1),y.imag/(mag+1)),1))
        b,w,f,t=z.shape
        z,_=self.time(z.permute(0,2,3,1).reshape(b*f,t,w))
        z=z.reshape(b,f,t,2*w).permute(0,2,1,3).reshape(b*t,f,2*w)
        z,_=self.freq(z)
        z=z.reshape(b,t,f,2*w).permute(0,3,2,1)
        h=F.interpolate(self.head(z),size=y.shape[-2:],mode="bilinear",align_corners=False)
        gain=2*torch.sigmoid(h[:,0])
        phase=math.pi*torch.tanh(h[:,1])
        return torch.istft(y*torch.polar(gain,phase),N_FFT,HOP,window=self.window,length=x.shape[-1])


class PESQCritic(nn.Module):
    """Differentiable train-only regression of real PESQ scores."""
    def __init__(self):
        super().__init__()
        self.register_buffer("window",torch.hann_window(512))
        layers=[]
        for a,b in ((3,16),(16,32),(32,48),(48,64)):
            layers.extend((nn.Conv2d(a,b,3,stride=2,padding=1),nn.GroupNorm(4,b),nn.SiLU()))
        self.net=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,1),nn.Sigmoid())
    def forward(self,pred,clean):
        def spec(x): return torch.stft(x,512,128,window=self.window,return_complex=True).abs().log1p()
        p,c=spec(pred),spec(clean)
        return self.net(torch.stack((p,c,(p-c).abs()),1)).flatten()

def critic_step(critic, optimizer, pred, noisy, clean, pesq_fn):
    # Only training batches; score a small, bounded subset to control CPU cost.
    n=min(2,len(clean))
    with torch.no_grad():
        target=clean[:n].detach().float()
        candidates=torch.cat((pred[:n].detach().float(),noisy[:n].detach().float(),target))
        candidates=candidates.clamp(-1,1)
        reference=target.repeat(3,1).clamp(-1,1)
        vals=score_waveforms(reference.cpu().numpy(),candidates.cpu().numpy(),pesq_fn,workers=0)
        labels=torch.tensor(vals,device=target.device,dtype=target.dtype).add(.5).div(5.2).clamp(0,1)
    critic.train().requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    loss=F.mse_loss(critic(candidates,reference),labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(),5.,error_if_nonfinite=True)
    optimizer.step()
    critic.eval().requires_grad_(False)
    return float(loss.detach())


class CriticReplay:
    """Bounded CPU-only replay of train reference groups; no audio persisted."""
    def __init__(self,capacity=64,seed=42):
        from collections import deque
        if capacity<2: raise ValueError("Replay capacity must be >=2")
        self.items=deque(maxlen=capacity)
        self.rng=random.Random(seed)
    def add(self,candidates,reference,labels):
        self.items.append(tuple(x.detach().cpu().clone() for x in (candidates,reference,labels)))
    def sample(self,n,device):
        items=self.rng.sample(list(self.items),min(n,len(self.items)))
        return tuple(torch.stack([item[i] for item in items]).to(device) for i in range(3))


def quality_group(pred,noisy,clean):
    # Multiple quality levels on the same reference, for relative ranking.
    return torch.stack((pred,noisy,.5*noisy+.5*clean,clean),dim=1).clamp(-1,1)


def rank_regression_loss(pred,label):
    reg=F.mse_loss(pred,label)
    gap=label[:,:,None]-label[:,None,:]
    pred_gap=pred[:,:,None]-pred[:,None,:]
    valid=gap>.04
    rank=F.relu(.5*gap-pred_gap)
    return reg+.25*(rank*valid).sum()/valid.sum().clamp_min(1)


def critic_replay_step(critic,optimizer,replay,pred,noisy,clean,pesq_fn):
    with torch.no_grad():
        ref=clean[:1].detach().float().clamp(-1,1)
        cand=quality_group(pred[:1].detach().float(),noisy[:1].float(),ref)[0]
        refs=ref.expand_as(cand)
        scores=score_waveforms(refs.cpu().numpy(),cand.cpu().numpy(),pesq_fn,workers=0)
        label=torch.tensor(scores,device=ref.device,dtype=ref.dtype).add(.5).div(5.2).clamp(0,1)
        replay.add(cand,refs,label)
    critic.train().requires_grad_(True)
    loss=0.
    # Fresh example plus replay; repeated updates on cached PESQ targets.
    for k in range(2):
        if k==0: x,r,y=cand[None],refs[None],label[None]
        else: x,r,y=replay.sample(2,ref.device)
        optimizer.zero_grad(set_to_none=True)
        p=critic(x.flatten(0,1),r.flatten(0,1)).reshape_as(y)
        objective=rank_regression_loss(p,y)
        if not torch.isfinite(objective): raise RuntimeError("Non-finite critic loss")
        objective.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        loss+=float(objective.detach())
    critic.eval().requires_grad_(False)
    return loss/2


def adaptive_metric_weight(calibration,epoch,updates):
    # Pre-registered engineering gate; no challenge validation score used.
    if epoch<3 or updates<16 or not calibration: return 0.
    corr=calibration.get("pearson")
    mae=calibration.get("mae")
    if corr is None or mae is None or not math.isfinite(corr) or not math.isfinite(mae) or corr<.6 or mae>.5: return 0.
    return .05*min(1.,max(0.,(epoch-2)/3))


def phase_complex_loss(pred,clean):
    """Power-compressed complex loss and target-energy-weighted circular phase loss.
    Phase comparison uses normalized complex vectors, not discontinuous raw angles.
    """
    total=pred.new_zeros(())
    for size,hop in ((256,64),(512,128),(1024,256)):
        w=torch.hann_window(size,device=pred.device,dtype=pred.dtype)
        a=torch.stft(pred,size,hop,window=w,return_complex=True)
        b=torch.stft(clean,size,hop,window=w,return_complex=True)
        am,bm=a.abs(),b.abs()
        eps=1e-4
        ac=a/(am+eps).pow(.7); bc=b/(bm+eps).pow(.7)
        norm=bc.abs().mean(dim=(-2,-1)).detach().clamp_min(.05)
        compressed=(ac-bc).abs().mean(dim=(-2,-1))/norm
        # Smooth normalization keeps silence finite; weight by target amplitude.
        au=a/(am.square()+eps**2).sqrt(); bu=b/(bm.square()+eps**2).sqrt()
        weight=bm.detach()
        circular=(1-(au*bu.conj()).real).clamp(0,2)
        phase=(weight*circular).sum(dim=(-2,-1))/weight.sum(dim=(-2,-1)).clamp_min(eps)
        total=total+(.7*compressed+.3*phase).mean()
    return total/3


def enhancement_loss(pred, clean, variant: str):
    """Phase-aware complex STFT + multi-resolution magnitude + waveform losses."""
    resolutions = ((256, 64), (512, 128), (1024, 256))
    spec_loss = pred.new_zeros(())
    complex_loss = pred.new_zeros(())
    for size, hop in resolutions:
        window = torch.hann_window(size, device=pred.device, dtype=pred.dtype)
        a = torch.stft(pred, size, hop_length=hop, window=window, return_complex=True)
        b = torch.stft(clean, size, hop_length=hop, window=window, return_complex=True)
        spec_loss = spec_loss + F.l1_loss(torch.log1p(a.abs()), torch.log1p(b.abs()))
        if size == 512:
            complex_delta = torch.view_as_real(a - b)
            normalizer = b.abs().mean().detach().clamp_min(0.1)
            complex_loss = complex_delta.abs().mean() / normalizer
    loss = (
        0.20 * F.l1_loss(pred, clean)
        + 0.60 * spec_loss / len(resolutions)
        + 0.20 * complex_loss
    )
    if variant == "V2_CRM_SISDR":
        clean_centered = clean - clean.mean(dim=-1, keepdim=True)
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        projection = (
            (pred_centered * clean_centered).sum(dim=-1, keepdim=True)
            / clean_centered.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        ) * clean_centered
        noise_residue = pred_centered - projection
        ratio = (projection.square().sum(dim=-1) + 1e-8) / (
            noise_residue.square().sum(dim=-1) + 1e-8
        )
        si_sdr = 10.0 * torch.log10(ratio)
        loss = loss - 0.01 * si_sdr.mean()
    return loss

def ensure_pesq():
    try:
        return importlib.import_module("pesq").pesq
    except ImportError:
        print("Installing pesq==0.0.4 (Kaggle Internet must be enabled)", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pesq==0.0.4"], check=True)
        return importlib.import_module("pesq").pesq


_PESQ_POOL=None

def _pesq_one(pair):
    from pesq import pesq
    reference,wave=pair
    return float(pesq(16000,reference,wave,"wb"))

def close_pesq_pool():
    global _PESQ_POOL
    if _PESQ_POOL is not None:
        _PESQ_POOL.shutdown(wait=True)
        _PESQ_POOL=None

def score_waveforms(clean,pred,pesq_function,workers=None):
    """Exact same clips and score order; native PESQ isolated in spawned processes."""
    import os
    pairs=[(np.clip(c,-1.0,1.0),np.clip(y,-1.0,1.0)) for c,y in zip(clean,pred)]
    if len(clean)!=len(pred): raise ValueError("PESQ pair count mismatch")
    if workers is None:
        workers=min(2,max(0,(os.cpu_count() or 1)-1)) if Path("/kaggle").exists() else 0
    if workers>1 and getattr(pesq_function,"__module__","").startswith("pesq"):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        global _PESQ_POOL
        if _PESQ_POOL is None:
            _PESQ_POOL=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn"))
        values=list(_PESQ_POOL.map(_pesq_one,pairs,chunksize=4))
    else:
        values=[float(pesq_function(16000,c,y,"wb")) for c,y in pairs]
    if not all(math.isfinite(v) for v in values): raise RuntimeError("Non-finite PESQ")
    return values

@torch.no_grad()
def evaluate(model, loader, device, pesq_function, save_rows=False):
    model.eval()
    predictions=[]; references=[]
    for noisy, clean in loader:
        predictions.append(model(noisy.to(device,non_blocking=True)).cpu().float().numpy())
        references.append(clean.numpy())
    if not predictions: raise RuntimeError("Empty validation")
    pred=np.concatenate(predictions); clean=np.concatenate(references)
    if len(pred)!=len(loader.dataset): raise RuntimeError("Validation is incomplete")
    values=score_waveforms(clean,pred,pesq_function)
    return float(np.mean(values)),[[i,v] for i,v in enumerate(values)] if save_rows else []

def write_csv(path: Path, fields, records):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(records)



# Public training corpora: LibriSpeech, MUSAN and optional SLR73/SLR28.
TRAIN_RIRS=[]
def public_assets(evaluation_only=False,include_test=False,prepare_only=False):
    import urllib.request, tarfile, hashlib, time
    def emit(key,stage,**values):
        print("V4_TELEMETRY "+json.dumps({"event":"data","dataset":key,"stage":stage,**values},allow_nan=False),flush=True)
    base = WORK/"public_cache" if prepare_only else (Path("/tmp/ralph_v4_external") if Path("/kaggle").exists() else WORK / "external")
    base.mkdir(parents=True, exist_ok=True)
    cache=EXPERIMENT_CONFIG.get("public_cache")
    if cache:
        candidates=list(Path("/kaggle/input").rglob("v4_cache_manifest.json"))
        slug=cache.rsplit("/",1)[-1]
        matches=[p for p in candidates if slug in p.parts]
        if len(matches)!=1: raise RuntimeError("Configured public cache not mounted uniquely")
        cache_root=matches[0].parent
        cache_meta=json.loads(matches[0].read_text(encoding="utf-8"))
        if cache_meta.get("format")!="ralph-v4-public-cache-1": raise RuntimeError("Invalid public cache format")
    else:
        cache_root=None; cache_meta={}
    sources = {"train":"https://www.openslr.org/resources/12/train-clean-100.tar.gz",
               "dev":"https://www.openslr.org/resources/12/dev-clean.tar.gz",
               "test":"https://www.openslr.org/resources/12/test-clean.tar.gz",
               "noise":"https://www.openslr.org/resources/17/musan.tar.gz"}
    if not include_test or prepare_only: sources.pop("test")
    if evaluation_only:
        sources.pop("train",None)
    records = {}
    if not evaluation_only and (EXPERIMENT_CONFIG.get("data_profile")=="spanish" or prepare_only):
        sources["extra_speech"]="https://www.openslr.org/resources/73/es_pe_female.zip"
    if not evaluation_only and (EXPERIMENT_CONFIG.get("data_profile")=="rir" or prepare_only):
        sources["rir"]="https://www.openslr.org/resources/28/rirs_noises.zip"
    def prepare_one(item):
        key,url=item
        emit(key,"prepare")
        archive = base / (key+(".zip" if url.endswith(".zip") else ".tar.gz"))
        target = base / key
        if cache_root is not None and key!="test":
            cached=cache_root/key
            if cache_meta.get("sources",{}).get(key)!=url or not (cached/"cache_complete.txt").is_file() or not (cached/"archive_sha256.txt").is_file(): raise RuntimeError("Missing or incompatible public cache entry: "+key)
            if target.exists() and not target.is_symlink():
                import shutil
                shutil.rmtree(target)
            if not target.exists(): target.symlink_to(cached,target_is_directory=True)
        marker=target/"cache_complete.txt"
        digest_path=target/"archive_sha256.txt"
        if not marker.exists() or not digest_path.exists():
            import shutil
            if target.exists(): shutil.rmtree(target)
            target.mkdir(exist_ok=True)
            if not archive.exists():
                started=time.monotonic()
                received=0; last_report=started
                with urllib.request.urlopen(url,timeout=60) as response, open(str(archive)+".partial","wb") as out:
                    header=response.headers.get("Content-Length") if hasattr(response,"headers") else None
                    total=int(header) if header and header.isdigit() and int(header)>0 else None
                    emit(key,"download",bytes=0,total_bytes=total,elapsed_s=0)
                    while True:
                        chunk=response.read(1024*1024)
                        if not chunk: break
                        out.write(chunk); received+=len(chunk)
                        now=time.monotonic()
                        if now-last_report>=5:
                            emit(key,"download",bytes=received,total_bytes=total,elapsed_s=now-started)
                            last_report=now
                    emit(key,"download",bytes=received,total_bytes=total,elapsed_s=time.monotonic()-started)
                    if total is not None and received!=total: raise IOError("Incomplete public archive download")
                Path(str(archive)+".partial").replace(archive)
            emit(key,"checksum")
            h=hashlib.sha256()
            with archive.open("rb") as f:
                for chunk in iter(lambda:f.read(1048576),b""):
                    h.update(chunk)
            # Deterministic subset: keep enough training speakers/files without
            # retaining the entire compressed and extracted corpus at once.
            def retain(name,size):
                path=Path(name)
                if path.suffix.lower() not in (".wav",".flac"):
                    return size<2*1024*1024 and (path.name.lower().endswith(".txt") or "license" in path.name.lower())
                if key=="noise" and not ({"noise","music"} & set(path.parts)): return False
                if key=="rir" and "simulated_rirs" not in path.parts: return False
                if key=="train" and int(hashlib.sha256(name.encode()).hexdigest()[:8],16)%4!=0: return False
                if key=="extra_speech" and int(hashlib.sha256(name.encode()).hexdigest()[:8],16)%2!=0: return False
                return size<=32*1024*1024
            emit(key,"extract")
            if archive.suffix==".zip":
                with zipfile.ZipFile(archive) as zf:
                    for member in zf.infolist():
                        if not (target/member.filename).resolve().is_relative_to(target.resolve()): raise ValueError("Unsafe archive path")
                        if not member.is_dir() and retain(member.filename,member.file_size):
                            zf.extract(member,target)
            else:
                with tarfile.open(archive,"r:gz") as tf:
                    for member in tf:
                        if member.isfile() and retain(member.name,member.size):
                            tf.extract(member,target,filter="data")
            digest_path.write_text(h.hexdigest(),encoding="ascii")
            marker.touch()
            archive.unlink(missing_ok=True)
        emit(key,"ready")
        return key,{"url":url,"sha256":digest_path.read_text(encoding="ascii"),"license":"CC-BY-SA-4.0" if key=="extra_speech" else "Apache-2.0" if key=="rir" else "CC-BY-4.0 (MUSAN individual terms apply)"}
    # Two concurrent archives on the CPU cache builder; bounded disk/network load.
    # GPU jobs normally mount the cache; no parallel downloads on GPU by default.
    if prepare_only and len(sources)>1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            records=dict(pool.map(prepare_one,sources.items()))
    else:
        records=dict(map(prepare_one,sources.items()))
    if prepare_only:
        (base/"v4_cache_manifest.json").write_text(json.dumps({"format":"ralph-v4-public-cache-1","sources":sources,"records":records},indent=2),encoding="utf-8")
        return base
    TRAIN_RIRS.clear()
    if "rir" in sources:
        TRAIN_RIRS[:] = sorted(p for p in (base/"rir").rglob("*.wav") if "simulated_rirs" in p.parts)
        if not TRAIN_RIRS: raise RuntimeError("No simulated room impulse responses")
    train=sorted((base/"train").rglob("*.flac"))
    if "extra_speech" in sources:
        extra=sorted((base/"extra_speech").rglob("*.wav"))
        if not extra: raise RuntimeError("Additional speech corpus empty")
        train+=extra
    dev=sorted((base/"dev").rglob("*.flac"))
    noises=sorted(p for p in (base/"noise").rglob("*.wav") if "noise" in p.relative_to(base/"noise").parts or "music" in p.relative_to(base/"noise").parts)
    groups=sorted({str(p.parent.relative_to(base/"noise")) for p in noises})
    random.Random(42).shuffle(groups)
    if len(groups)<3: raise RuntimeError("Need three disjoint noise groups")
    k=max(1,len(groups)//5)
    held=set(groups[:k])
    test_held=set(groups[k:2*k])
    tr=[p for p in noises if str(p.parent.relative_to(base/"noise")) not in held|test_held]
    dv=[p for p in noises if str(p.parent.relative_to(base/"noise")) in held]
    te=[p for p in noises if str(p.parent.relative_to(base/"noise")) in test_held]
    test=sorted((base/"test").rglob("*.flac"))
    train,dev,tr,dv,test,te=map(quality_paths,(train,dev,tr,dv,test,te))
    if not all((dev,tr,dv,te)) or (not evaluation_only and not train) or (include_test and not test):
        raise RuntimeError("Empty external corpus split")
    records["counts"]={"train":len(train),"dev":len(dev),"test":len(test),"train_noise":len(tr),"dev_noise":len(dv),"test_noise":len(te),"rir":len(TRAIN_RIRS)}
    (WORK/"external_manifest.json").write_text(json.dumps(records,indent=2))
    return train,dev,tr,dv,test,te

class SyntheticPairs(Dataset):
    def __init__(self,speech,noise,size,seed,dynamic,stress=False):
        self.speech,self.noise,self.size,self.seed,self.dynamic=speech,noise,size,seed,dynamic
        self.stress=stress
        self.rirs=tuple(TRAIN_RIRS) if dynamic else ()
        self.augmentation=EXPERIMENT_CONFIG.get("augmentation","base")
        self.perf_seed=int(EXPERIMENT_CONFIG["seed"]) if dynamic and (EXPERIMENT_CONFIG.get("perf_reproducible") or EXPERIMENT_CONFIG.get("v5_stage") or EXPERIMENT_CONFIG.get("v52_stage")) else None
        self.epoch=0
    def __len__(self):
        return self.size
    def __getitem__(self,i):
        import scipy.signal
        import math
        rng=(random.Random(self.perf_seed*1000000007+self.seed*1000003+self.epoch*100003+i) if self.perf_seed is not None else random) if self.dynamic else random.Random(self.seed+i)
        def clip(paths):
            path=rng.choice(paths)
            with sf.SoundFile(str(path)) as audio:
                sr=audio.samplerate
                if sr==16000 and len(audio)>=48000:
                    # Identical random crop to full-file decode, without reading unused audio.
                    k=rng.randint(0,len(audio)-48000)
                    audio.seek(k)
                    chunk=audio.read(48000,dtype="float32",always_2d=True).mean(axis=1).astype(np.float32)
                    if len(chunk)!=48000 or not np.isfinite(chunk).all(): raise ValueError("Invalid audio crop")
                    return chunk
                y=audio.read(dtype="float32",always_2d=True).mean(axis=1)
            if sr!=16000:
                g=math.gcd(sr,16000)
                y=scipy.signal.resample_poly(y,16000//g,sr//g)
            if len(y)==0 or not np.isfinite(y).all(): raise ValueError("Invalid audio")
            if len(y)<48000: y=np.tile(y,(48000+len(y)-1)//len(y))
            k=rng.randint(0,len(y)-48000)
            return y[k:k+48000].astype(np.float32)
        c,n=clip(self.speech),clip(self.noise)
        snr=rng.uniform(-10,5) if self.stress else rng.uniform(-5,20)
        n=n*(np.sqrt(np.mean(c*c)+1e-8)/np.sqrt(np.mean(n*n)+1e-8))*10**(-snr/20)
        x=c+n
        if self.dynamic and self.rirs and rng.random()<0.5:
            h,sr=sf.read(rng.choice(self.rirs),dtype="float32",always_2d=True)
            h=h[:,0]
            if sr!=16000: h=scipy.signal.resample_poly(h,16000//math.gcd(sr,16000),sr//math.gcd(sr,16000))
            if len(h) and np.isfinite(h).all() and np.max(np.abs(h))>1e-8:
                h=h[np.argmax(np.abs(h)):][:16000]
                h=h/max(1e-8,float(np.linalg.norm(h)))
                x=scipy.signal.fftconvolve(x,h)[:len(x)]
        if self.stress or (self.dynamic and self.augmentation=="room_codec"):
            # Train-only synthetic room tails and bandwidth degradation.
            if rng.random()<0.6:
                impulse=np.zeros(2048,dtype=np.float32); impulse[0]=1
                for k in range(1,7):
                    delay=rng.randint(20,2047)
                    impulse[delay]+=rng.uniform(-.3,.3)*np.exp(-delay/1000)
                x=scipy.signal.fftconvolve(x,impulse)[:len(x)]
            if rng.random()<0.4:
                x=scipy.signal.resample_poly(scipy.signal.resample_poly(x,1,2),2,1)[:len(x)]
        scale=max(1.0,float(np.max(np.abs(x))),float(np.max(np.abs(c))))
        return torch.from_numpy((x/scale).astype(np.float32)),torch.from_numpy((c/scale).astype(np.float32))



def cmgan_assets():
    """MIT upstream CMGAN, pinned commit and verified git blob hashes."""
    import hashlib, urllib.request
    revision="39946005d9b66fa1e824edf4bb6bc9a06088e443"
    blobs={"LICENSE":"6750370caf9bdc4710c5c1c0da12450a7d1bf336",
           "src/models/conformer.py":"4594f11803a988f2540a30830c28e51c88f25126",
           "src/models/generator.py":"d0dfacc36128d8b51386c3bb7c0fc8edfad82074",
           "src/best_ckpt/ckpt":"cbafab9a5fe739f3662c65cfe7be009dc475e0d1"}
    base=Path("/tmp/ralph_cmgan_assets") if Path("/kaggle").exists() else WORK/"cmgan_assets"
    base.mkdir(parents=True,exist_ok=True)
    for path,expected in blobs.items():
        dest=base/path
        dest.parent.mkdir(parents=True,exist_ok=True)
        def valid(data): return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()==expected
        if not dest.exists() or not valid(dest.read_bytes()):
            url=f"https://raw.githubusercontent.com/ruizhecao96/CMGAN/{revision}/{path}"
            with urllib.request.urlopen(url,timeout=60) as response: data=response.read(10_000_001)
            if len(data)>10_000_000 or not valid(data): raise RuntimeError("CMGAN asset hash mismatch")
            dest.write_bytes(data)
    pkg=base/"ralph_cmgan_pinned"
    pkg.mkdir(exist_ok=True)
    (pkg/"__init__.py").write_text("",encoding="utf-8")
    (pkg/"conformer.py").write_bytes((base/"src/models/conformer.py").read_bytes())
    source=(base/"src/models/generator.py").read_text(encoding="utf-8")
    (pkg/"generator.py").write_text(source.replace("from models.conformer import","from .conformer import"),encoding="utf-8")
    if str(base) not in sys.path: sys.path.insert(0,str(base))
    try: import einops
    except ImportError:
        subprocess.check_call([sys.executable,"-m","pip","install","-q","einops==0.8.1"])
    module=importlib.import_module("ralph_cmgan_pinned.generator")
    return module,base/"src/best_ckpt/ckpt"


class CMGANTransfer(torch.nn.Module):
    """Official pretrained generator; frozen encoder/Conformer, trainable decoders."""
    def __init__(self):
        super().__init__()
        module,checkpoint=cmgan_assets()
        self.net=module.TSCNet(num_channel=64,num_features=201)
        self.net.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True),strict=True)
        self.net.requires_grad_(False)
        self.net.mask_decoder.requires_grad_(True)
        self.net.complex_decoder.requires_grad_(True)
        self.register_buffer("window",torch.hamming_window(400))
    def forward(self,x):
        scale=x.square().mean(-1,keepdim=True).clamp_min(1e-8).rsqrt()
        y=torch.stft(x*scale,400,100,window=self.window,return_complex=True)
        mag=y.abs()
        compressed=y*mag.clamp_min(1e-8).pow(-0.7)
        spec=torch.stack((compressed.real,compressed.imag),1).transpose(-1,-2)
        # Freeze upstream attention to limit training memory.
        for m in (self.net.dense_encoder,self.net.TSCB_1,self.net.TSCB_2,self.net.TSCB_3,self.net.TSCB_4): m.eval()
        re,im=self.net(spec)
        z=torch.complex(re[:,0].transpose(-1,-2),im[:,0].transpose(-1,-2))
        z=z*z.abs().clamp_min(1e-8).pow(1/0.3-1)
        return torch.istft(z,400,100,window=self.window,length=x.shape[-1])/scale


class MetricGANTransfer(nn.Module):
    """Fine-tune public generator weights through differentiable STFT masking."""
    def __init__(self):
        super().__init__()
        try:
            import speechbrain
        except ImportError:
            subprocess.check_call([sys.executable,"-m","pip","install","-q","speechbrain==1.0.3","huggingface_hub<1"])
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        engine=SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank",
            savedir=str(Path("/tmp/ralph_v4_metricgan") if Path("/kaggle").exists() else WORK/"metricgan_weights"),run_opts={"device":"cpu"},freeze_params=False)
        self.generator=engine.mods.enhance_model
        self.features=engine.hparams.compute_stft
        self.resynthesize=engine.hparams.resynth
        self.generator.requires_grad_(True)
    def forward(self,x):
        from speechbrain.processing.features import spectral_magnitude
        mag=spectral_magnitude(self.features(x),power=0.5)
        features=torch.log1p(mag)
        mask=self.generator(features,lengths=torch.ones(x.size(0),device=x.device))
        if isinstance(mask,tuple): mask=mask[0]
        return self.resynthesize(torch.expm1(mask*features),x)


class SepformerTransfer(nn.Module):
    """Registered SepFormer encoder/mask network, trainable decoder adaptation."""
    def __init__(self):
        super().__init__()
        try:
            import speechbrain
        except ImportError:
            subprocess.check_call([sys.executable,"-m","pip","install","-q","speechbrain==1.0.3","huggingface_hub<1"])
        from speechbrain.inference.separation import SepformerSeparation
        family=EXPERIMENT_CONFIG.get("architecture","sepformer_dns_transfer")
        source={"sepformer_dns_transfer":"speechbrain/sepformer-dns4-16k-enhancement","sepformer_wham_transfer":"speechbrain/sepformer-wham16k-enhancement"}[family]
        engine=SepformerSeparation.from_hparams(source=source,savedir=str(Path("/tmp/ralph_v4_sepformer")/family if Path("/kaggle").exists() else WORK/family),run_opts={"device":"cpu"},freeze_params=False)
        if int(engine.hparams.num_spks)!=1: raise ValueError("Expected one enhanced source")
        self.encoder=engine.mods.encoder.requires_grad_(False)
        self.masknet=engine.mods.masknet.requires_grad_(False)
        self.decoder=engine.mods.decoder.requires_grad_(True)
    def forward(self,x):
        self.encoder.eval(); self.masknet.eval()
        with torch.no_grad():
            z=self.encoder(x)
            mask=self.masknet(z)
            z=z*mask[0]
        y=self.decoder(z)
        if y.ndim==3 and y.shape[1]==1: y=y[:,0]
        if y.ndim!=2: raise RuntimeError("SepFormer decoder shape")
        return F.pad(y,(0,max(0,x.shape[-1]-y.shape[-1])))[:,:x.shape[-1]]


class GeneratedEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        raise RuntimeError('Generated model not supplied')

class TemporalEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp=nn.Conv1d(1,32,15,padding=7)
        self.blocks=nn.ModuleList(nn.Sequential(nn.Conv1d(32,32,9,padding=4*d,dilation=d,groups=32),
            nn.Conv1d(32,32,1),nn.GELU()) for d in (1,2,4,8,16,32,64,128))
        self.out=nn.Conv1d(32,1,15,padding=7)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    def forward(self,x):
        z=self.inp(x.unsqueeze(1))
        for block in self.blocks: z=z+block(z)
        return x+self.out(z).squeeze(1)


class ComposedEnhancer(nn.Module):
    """A new topology assembled from validated JSON, no generated Python execution."""
    def __init__(self):
        super().__init__()
        spec=EXPERIMENT_CONFIG["network_spec"]
        width=spec["width"]
        activation=nn.GELU if spec["activation"]=="gelu" else nn.SiLU
        self.inp=nn.Conv1d(1,width,9,padding=4)
        self.blocks=nn.ModuleList()
        for b in spec["blocks"]:
            k,d=b["kernel"],b["dilation"]
            pad=d*(k-1)//2
            if b["type"]=="dense":
                layers=[nn.Conv1d(width,width,k,padding=pad,dilation=d)]
            elif b["type"]=="gated":
                layers=[nn.Conv1d(width,2*width,k,padding=pad,dilation=d,groups=width),nn.GLU(dim=1),nn.Conv1d(width,width,1)]
            else:
                layers=[nn.Conv1d(width,width,k,padding=pad,dilation=d,groups=width),nn.Conv1d(width,width,1)]
            if spec["norm"]: layers.append(nn.GroupNorm(1,width))
            layers.append(activation())
            self.blocks.append(nn.Sequential(*layers))
        self.out=nn.Conv1d(width,1,9,padding=4)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self,x):
        z=self.inp(x.unsqueeze(1))
        for block in self.blocks: z=z+block(z)
        return x+self.out(z).squeeze(1)


class NoisyOnly(Dataset):
    """Read only noisy manifests; never read validation/test clean files."""
    def __init__(self,root,transductive=False):
        self.paths=[]
        for split in (("train","validation","test") if transductive else ("train",)):
            p=root/split/"samples.csv"
            if p.is_file():
                for name in pd.read_csv(p,usecols=["noisy_path"])["noisy_path"]:
                    q=(root/name).resolve()
                    if not q.is_relative_to(root.resolve()): raise ValueError("Unsafe path")
                    self.paths.append(q)
        self.paths=list(dict.fromkeys(self.paths))
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        x,sr=sf.read(self.paths[i],dtype="float32",always_2d=True)
        if sr!=16000 or x.shape!=(48000,1): raise ValueError("Unexpected noisy shape")
        return torch.from_numpy(x[:,0])



def quality_paths(paths):
    valid=[]
    seen=set()
    for p in paths:
        try:
            info=sf.info(p)
            if info.frames<info.samplerate//4 or info.samplerate<8000: continue
            with sf.SoundFile(p) as file:
                y=file.read(min(info.frames,info.samplerate*3),dtype="float32",always_2d=True)
            if not np.isfinite(y).all() or np.sqrt(np.mean(y*y))<1e-5: continue
            import hashlib
            digest=hashlib.sha256(y.tobytes()).hexdigest()
            if digest in seen: continue
            seen.add(digest); valid.append(p)
        except (OSError,ValueError,RuntimeError): continue
    return valid

class AdaptiveMixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=ResidualPhaseUNet()
        self.b=TemporalEnhancer()
        self.gate=nn.Sequential(nn.Linear(3,16),nn.GELU(),nn.Linear(16,3))
    def forward(self,x):
        rms=x.square().mean(-1).clamp_min(1e-8).sqrt()
        delta=(x[:,1:]-x[:,:-1]).square().mean(-1).clamp_min(1e-8).sqrt()
        feat=torch.stack([rms.log(),delta.log(),x.abs().amax(-1)],-1)
        w=self.gate(feat).softmax(-1)
        return w[:,0,None]*self.a(x)+w[:,1,None]*self.b(x)+w[:,2,None]*spectral_gate(x)

def spectral_gate(x):
    win=torch.hann_window(512,device=x.device,dtype=x.dtype)
    z=torch.stft(x,512,128,window=win,return_complex=True)
    mag=z.abs()
    noise=mag.quantile(.2,dim=-1,keepdim=True)
    gain=(1-noise.square()/(mag.square()+1e-8)).clamp(.08,1).sqrt()
    return torch.istft(z*gain,512,128,window=win,length=x.shape[-1])

def additional_metrics(model,loader,device):
    from pystoi import stoi
    scores=[]
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            pred=model(x.to(device)).cpu().clamp(-1,1).numpy()
            for a,b in zip(pred,y.numpy()):
                a=a-a.mean(); b=b-b.mean()
                p=np.dot(a,b)*b/(np.dot(b,b)+1e-8)
                si=10*np.log10((np.dot(p,p)+1e-8)/(np.dot(a-p,a-p)+1e-8))
                scores.append((float(si),float(stoi(b,a,16000,extended=False))))
    if not np.isfinite(scores).all(): raise RuntimeError("Nonfinite auxiliary metric")
    return {"si_sdr":float(np.mean(scores,axis=0)[0]),"stoi":float(np.mean(scores,axis=0)[1])}



@torch.no_grad()
def dnsmos_metrics(model,loader,device):
    from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
    rows=[]
    model.eval()
    for x,_ in loader:
        pred=model(x.to(device)).cpu().clamp(-1,1)
        rows.extend(deep_noise_suppression_mean_opinion_score(pred,16000,False).cpu().reshape(-1,4).tolist())
    if not rows or not np.isfinite(rows).all(): raise RuntimeError("Invalid DNSMOS")
    return dict(zip(("p808","sig","bak","ovrl"),map(float,np.mean(rows,axis=0))))

def word_error_count(reference,hypothesis):
    a,b=reference.lower().split(),hypothesis.lower().split()
    prev=list(range(len(b)+1))
    for i,w in enumerate(a,1):
        cur=[i]
        for j,v in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(w!=v)))
        prev=cur
    return prev[-1],len(a)

@torch.no_grad()
def transcription_audit(model,speech,noise,device):
    """WER uses full held-out utterances with original LibriSpeech transcripts.
    This is an audit-only metric, never fed to the research planner.
    """
    from transformers import pipeline
    import scipy.signal
    rng=random.Random(82491)
    asr=pipeline("automatic-speech-recognition",model="openai/whisper-tiny.en",device=-1)
    transcripts={}
    for file in sorted({p.parent for p in speech}):
        for path in file.glob("*.trans.txt"):
            for row in path.read_text(encoding="utf-8").splitlines():
                if " " in row:
                    key,value=row.split(" ",1); transcripts[key]=value
    chosen=[p for p in speech if p.stem in transcripts]
    rng.shuffle(chosen)
    errors=words=0
    for p in chosen[:24]:
        c,sr=sf.read(p,dtype="float32",always_2d=True)
        c=c.mean(1)
        if sr!=16000: c=scipy.signal.resample_poly(c,16000//math.gcd(sr,16000),sr//math.gcd(sr,16000))
        n,nsr=sf.read(rng.choice(noise),dtype="float32",always_2d=True)
        n=n.mean(1)
        if nsr!=16000: n=scipy.signal.resample_poly(n,16000//math.gcd(nsr,16000),nsr//math.gcd(nsr,16000))
        if not len(c) or not len(n): continue
        n=np.tile(n,(len(c)+len(n)-1)//len(n))[:len(c)]
        x=c+n*np.sqrt((np.mean(c*c)+1e-8)/(np.mean(n*n)+1e-8))*.56234
        x=x/max(1,float(np.max(np.abs(x))))
        out=[]
        for k in range(0,len(x),48000):
            clip=x[k:k+48000]
            y=model(torch.from_numpy(np.pad(clip,(0,48000-len(clip))).astype(np.float32))[None].to(device))
            out.append(y.cpu().numpy()[0,:len(clip)])
        wave=np.concatenate(out).clip(-1,1)
        text=asr({"array":wave,"sampling_rate":16000})["text"]
        import re
        normalize=lambda t:re.sub(r"[^a-z0-9' ]"," ",t.lower())
        e,w=word_error_count(normalize(transcripts[p.stem]),normalize(text))
        errors+=e; words+=w
    if not words: raise RuntimeError("No transcript words available")
    return {"wer":errors/words,"words":words}



@torch.no_grad()
def enhance_long(model,x,device):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all(): raise ValueError("Invalid audit waveform")
    out=np.zeros(len(x),np.float64); weights=np.zeros(len(x),np.float64)
    taper=np.maximum(np.hanning(48000),1e-3)
    for start in range(0,len(x),24000):
        segment=x[start:start+48000]; n=len(segment)
        y=model(torch.from_numpy(np.pad(segment,(0,48000-n)))[None].to(device))
        if y.shape!=(1,48000) or not torch.isfinite(y).all(): raise RuntimeError("Audit output contract failed")
        out[start:start+n]+=y.cpu().numpy()[0,:n]*taper[:n]
        weights[start:start+n]+=taper[:n]
        if start+n==len(x): break
    return (out/weights).astype(np.float32).clip(-1,1)

def decode_audit_audio(audio):
    import io, scipy.signal
    if isinstance(audio,dict) and audio.get("bytes") is not None:
        x,sr=sf.read(io.BytesIO(audio["bytes"]),dtype="float32",always_2d=True)
        x=x.mean(1)
    elif isinstance(audio,dict) and "array" in audio:
        x=np.asarray(audio["array"],dtype=np.float32); sr=int(audio["sampling_rate"])
        if x.ndim!=1: raise ValueError("Expected mono URGENT audio")
    else:
        raise ValueError("Missing audio payload")
    if sr<=0 or not len(x) or not np.isfinite(x).all(): raise ValueError("Invalid URGENT audio")
    if sr!=16000:
        g=math.gcd(sr,16000); x=scipy.signal.resample_poly(x,16000//g,sr//g).astype(np.float32)
    return x

def parquet_audio_rows(url,columns=None):
    """Synchronous Arrow batches, no background dataset iterator at shutdown."""
    import fsspec
    import pyarrow.parquet as pq
    with fsspec.open(url,"rb",block_size=4*1024*1024,cache_type="readahead") as stream:
        parquet=pq.ParquetFile(stream,pre_buffer=False)
        try:
            for batch in parquet.iter_batches(batch_size=8,columns=columns,use_threads=False):
                yield from batch.to_pylist()
        finally:
            parquet.close()

def urgent_rows(blind=False):
    """Read pinned Parquet directly, never execute a dataset repository script."""
    config="test_blind" if blind else "test_nonblind"
    revision="231e45142538c710ae3ec3d7c8b6e81e9f848b01"
    url=f"https://huggingface.co/datasets/urgent-challenge/urgent2024_official/resolve/{revision}/data/{config}/{config}-00000-of-00001.parquet"
    return parquet_audio_rows(url)

def urgent_audit(model,device,pesq_fn,limit=24,blind=False):
    """Frozen out-of-corpus evaluation; blind split is mixed real/simulated, no PESQ."""
    config="test_blind" if blind else "test_nonblind"
    split=urgent_rows(blind)
    rows=[]
    if blind:
        from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score as dns
    from contextlib import closing
    with closing(split):
        for i,sample in enumerate(split):
            if i%7!=0: continue
            x=decode_audit_audio(sample["noisy_audio"])
            y=enhance_long(model,x,device)
            if blind:
                scores=dns(torch.from_numpy(y)[None],16000,False).reshape(-1,4).mean(0).tolist()
            else:
                c=decode_audit_audio(sample["clean_audio"])
                if abs(len(c)-len(y))>160: raise RuntimeError("URGENT reference alignment mismatch")
                n=min(len(c),len(y)); c=c[:n]; y=y[:n]
                scores=[float(pesq_fn(16000,c.clip(-1,1),y,"wb"))]
            if not np.isfinite(scores).all(): raise RuntimeError("Nonfinite URGENT score")
            rows.append([str(sample["id"]),*scores])
            if len(rows)>=limit: break
    if len(rows)!=limit: raise RuntimeError("Incomplete URGENT audit")
    keys=("p808","sig","bak","ovrl") if blind else ("pesq_wb",)
    write_csv(WORK/("urgent_blind.csv" if blind else "urgent_paired.csv"),["id",*keys],rows)
    return {"count":len(rows),"scores":dict(zip(keys,map(float,np.mean([r[1:] for r in rows],axis=0)))),"split":config,"sample_rule":"every seventh item; fixed 24 items; pretrained overlap not established"}



def fleurs_speech(split,limit=256):
    """Approximate clean read speech; fixed official Korean train/test split."""
    if split not in ("train","test"): raise ValueError("Unsupported FLEURS split")
    revision="44ebc3e0222cda1571220d148f3e3163843747dd"
    root=(Path("/tmp/ralph_v4_fleurs") if Path("/kaggle").exists() else WORK/"fleurs")/split
    root.mkdir(parents=True,exist_ok=True)
    url=f"https://huggingface.co/datasets/google/fleurs/resolve/{revision}/ko_kr/{split}-00000-of-00001.parquet"
    data=parquet_audio_rows(url,columns=["audio"])
    paths=[]
    from contextlib import closing
    with closing(data):
        for i,row in enumerate(data):
            if len(paths)>=limit: break
            x=decode_audit_audio(row["audio"])
            if len(x)<16000 or np.sqrt(np.mean(x*x))<1e-5: continue
            dest=root/f"clip_{i:05d}.wav"
            sf.write(dest,x.clip(-1,1),16000,subtype="PCM_16")
            paths.append(dest)
    if len(paths)!=limit: raise RuntimeError("Incomplete FLEURS "+split)
    return paths



@torch.no_grad()
def update_ema(teacher,model):
    if teacher is None: return
    for t,p in zip(teacher.parameters(),model.parameters()):
        t.lerp_(p,0.01)
    for t,p in zip(teacher.buffers(),model.buffers()):
        if t.is_floating_point(): t.lerp_(p,0.01)
        else: t.copy_(p)

def supervised_step(model,optimizer,noisy,clean,cfg,distill=None,teacher=None,scaler=None,critic=None,critic_optimizer=None,pesq_fn=None,metric_weight=0.,update_critic=False,metric_probe=None,critic_replay=None):
    variant=cfg["id"]
    optimizer.zero_grad(set_to_none=True)
    # Only convolutional forward uses AMP; STFT, objectives and teachers stay FP32.
    if scaler is not None:
        with torch.autocast(device_type="cuda",dtype=torch.float16):
            pred=model(noisy)
        pred=pred.float()
    else:
        pred=model(noisy)
    loss = enhancement_loss(pred, clean, variant)
    if cfg.get("phase_loss"): loss=loss+0.15*phase_complex_loss(pred.float(),clean.float())
    if distill is not None:
        with torch.no_grad(): pseudo=distill(noisy).clamp(-1,1)
        loss=loss+0.1*F.l1_loss(pred,pseudo)
    if cfg.get("loss_mode") == "composed":
        weights=cfg["loss_weights"]
        a=pred-pred.mean(-1,keepdim=True)
        b=clean-clean.mean(-1,keepdim=True)
        proj=(a*b).sum(-1,keepdim=True)*b/(b.square().sum(-1,keepdim=True)+1e-8)
        si=(10*torch.log10((proj.square().sum(-1)+1e-8)/((a-proj).square().sum(-1)+1e-8))).clamp(-30,30).mean()
        win=torch.hann_window(512,device=pred.device,dtype=pred.dtype)
        p_mag=torch.stft(pred,512,128,window=win,return_complex=True).abs()
        c_mag=torch.stft(clean,512,128,window=win,return_complex=True).abs()
        loss=loss+weights["wave"]*F.l1_loss(pred,clean)+weights["spectral"]*F.l1_loss(p_mag.log1p(),c_mag.log1p())-0.01*weights["sisdr"]*si
    elif cfg.get("loss_mode") == "sisdr":
        a=pred-pred.mean(-1,keepdim=True)
        b=clean-clean.mean(-1,keepdim=True)
        proj=(a*b).sum(-1,keepdim=True)*b/(b.square().sum(-1,keepdim=True)+1e-8)
        loss=loss-0.01*(10*torch.log10((proj.square().sum(-1)+1e-8)/((a-proj).square().sum(-1)+1e-8))).clamp(-30,30).mean()
    elif cfg.get("loss_mode") == "waveform":
        loss=loss+0.2*F.l1_loss(pred,clean)
    if critic is not None:
        if update_critic:
            if critic_replay is None: critic_step(critic,critic_optimizer,pred,noisy,clean,pesq_fn)
            else: critic_replay_step(critic,critic_optimizer,critic_replay,pred,noisy,clean,pesq_fn)
        if metric_weight>0:
            critic.eval().requires_grad_(False)
            metric_loss=metric_weight*(1-critic(pred.float().clamp(-1,1),clean.float().clamp(-1,1))).square().mean()
            if metric_probe is not None:
                grad=torch.autograd.grad(metric_loss,pred,retain_graph=True)[0]
                norm=float(grad.detach().norm())
                if not math.isfinite(norm) or norm<=0: raise RuntimeError("PESQ surrogate gradient is zero or non-finite")
                metric_probe["gradient_norm"]=norm
            loss=loss+metric_loss
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite train loss")
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
        scale_before=scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale()>=scale_before: update_ema(teacher,model)
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0,error_if_nonfinite=True)
        optimizer.step()
        update_ema(teacher,model)
    return float(loss.detach())

@torch.no_grad()
def critic_calibration(model,critic,loader,device,pesq_fn):
    """Engineering fixture only; no gradients and no use in checkpoint selection."""
    model.eval(); critic.eval()
    truth=[]; estimates=[]
    for noisy,clean in loader:
        clean=clean.to(device).float().clamp(-1,1)
        pred=model(noisy.to(device)).float().clamp(-1,1)
        truth.extend(score_waveforms(clean.cpu().numpy(),pred.cpu().numpy(),pesq_fn))
        estimates.extend((critic(pred,clean)*5.2-.5).cpu().tolist())
    a,b=np.asarray(truth),np.asarray(estimates)
    if len(a)!=len(loader.dataset) or not np.isfinite(a).all() or not np.isfinite(b).all(): raise RuntimeError("Invalid critic calibration")
    pearson=float(np.corrcoef(a,b)[0,1]) if len(a)>1 and a.std()>1e-8 and b.std()>1e-8 else None
    return {"count":len(a),"mae":float(np.mean(np.abs(a-b))),"rmse":float(np.sqrt(np.mean((a-b)**2))),"pearson":pearson,"split":"train-derived engineering fixture","final_epoch_model":True}


def consistency_step(model,optimizer,teacher,x,mode="consistency"):
    with torch.no_grad():
        target=teacher(x).clamp(-1,1)
        if mode=="remix" and x.shape[0]>1:
            # Recombine estimated speech and permuted estimated noise.
            residue=x-target
            perturbed=target+residue.roll(1,dims=0)
        else:
            perturbed=x+torch.randn_like(x)*0.01
    optimizer.zero_grad(set_to_none=True)
    loss=0.1*F.l1_loss(model(perturbed.clamp(-1,1)),target)
    if not torch.isfinite(loss): raise RuntimeError("Nonfinite consistency loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
    optimizer.step()
    update_ema(teacher,model)
    return float(loss.detach())


def smoke_pairs(train):
    """Offline engineering fixture only; real train speech + synthetic noise.
    No public download or validation-clean input to training. Never ranked.
    """
    clean=torch.from_numpy(train.clean[:80]).clone()
    rng=torch.Generator().manual_seed(20261125)
    noise=torch.randn(clean.shape,generator=rng)
    rms=clean.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-5)
    noise=noise/noise.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-5)
    snr=torch.linspace(0,15,len(clean)).reshape(-1,1)
    noisy=clean+noise*rms*10**(-snr/20)
    scale=torch.maximum(noisy.abs().amax(-1,keepdim=True),clean.abs().amax(-1,keepdim=True)).clamp_min(1)
    pairs=torch.utils.data.TensorDataset(noisy/scale,clean/scale)
    return torch.utils.data.Subset(pairs,list(range(16))),torch.utils.data.Subset(pairs,list(range(16,80)))

def fixed_pairs(dataset):
    """Materialize deterministic validation once; identical order and values."""
    pairs=[dataset[i] for i in range(len(dataset))]
    return torch.utils.data.TensorDataset(torch.stack([x for x,y in pairs]),torch.stack([y for x,y in pairs]))

def make_loader(dataset,batch_size,shuffle,device,dynamic=False):
    import os
    workers=min(int(EXPERIMENT_CONFIG.get("loader_workers",0)),2,max(0,(os.cpu_count() or 1)-1)) if device.type=="cuda" and dynamic else 0
    opts={"num_workers":workers,"pin_memory":device.type=="cuda"}
    if workers: opts.update(persistent_workers=False,prefetch_factor=2,multiprocessing_context="spawn")
    if shuffle and (EXPERIMENT_CONFIG.get("perf_reproducible") or EXPERIMENT_CONFIG.get("v52_stage")):
        opts["generator"]=torch.Generator().manual_seed(int(EXPERIMENT_CONFIG["seed"]))
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,**opts)

def set_data_epoch(dataset,epoch):
    if isinstance(dataset,(SyntheticPairs,ChallengeRemix)):
        dataset.epoch=epoch
    elif isinstance(dataset,torch.utils.data.ConcatDataset):
        for child in dataset.datasets: set_data_epoch(child,epoch)
    elif isinstance(dataset,torch.utils.data.Subset):
        set_data_epoch(dataset.dataset,epoch)

def fingerprint_pairs(*datasets):
    import hashlib
    digest=hashlib.sha256()
    for data in datasets:
        digest.update(str(len(data)).encode())
        for i in range(len(data)):
            for wave in data[i]:
                arr=wave.detach().cpu().contiguous().numpy()
                digest.update(str(arr.shape).encode())
                digest.update(arr.tobytes())
    return digest.hexdigest()

def run():
    cfg = EXPERIMENT_CONFIG
    variant = cfg["id"]
    if variant not in ("V3_RESIDUAL_PHASE",):
        raise ValueError(f"Unknown experiment: {variant}")
    WORK.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg["seed"]))
    print(
        f"CUDA_DIAGNOSTIC torch={torch.__version__} "
        f"cuda_build={torch.version.cuda} "
        f"cuda_available={torch.cuda.is_available()} "
        f"device_count={torch.cuda.device_count()}",
        flush=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and Path("/kaggle").exists():
        raise RuntimeError("Kaggle GPU unavailable; refusing CPU full training")
    print(f"EXPERIMENT={variant} device={device} seed={cfg['seed']}", flush=True)
    hardware={"device":torch.cuda.get_device_name(device) if device.type=="cuda" else "cpu","torch":str(torch.__version__),"cuda":torch.version.cuda,"cpu_count":__import__("os").cpu_count()}
    expected=cfg.get("expected_perf")
    if expected and hardware!=expected["hardware"]: raise RuntimeError("Approved performance hardware changed; use serial FP32")

    is_audit=bool(cfg.get("final_audit"))
    smoke=bool(cfg.get("smoke"))
    perf=bool(cfg.get("perf_reproducible"))
    data_fingerprint=None
    training_fingerprint=None
    root=None
    critic_dev=None
    heldout=[]
    if not smoke:
        tr, dv, tn, dn, ts, tnoise = public_assets(evaluation_only=is_audit,include_test=is_audit)
    if not is_audit:
        root = locate_data_root()
        train_set = SpeechPairs(root, "train")
        if len(train_set) != 1296:
            raise RuntimeError("Unexpected train split size")
        original_train=train_set
        if cfg.get("v52_stage"):
            # Same fixed, disjoint train-only holdout for every V5.2 ablation.
            heldout=sorted(np.random.default_rng(52042).choice(len(original_train),32,replace=False).tolist())
            heldout_set=set(heldout)
            critic_dev=make_loader(torch.utils.data.Subset(original_train,heldout),4,False,device)
            train_set=torch.utils.data.Subset(original_train,[i for i in range(len(original_train)) if i not in heldout_set])
        if smoke:
            fixture_train,fixture_val=smoke_pairs(train_set)
            train_set=torch.utils.data.ConcatDataset([torch.utils.data.Subset(train_set,list(range(16))),fixture_train])
            external_val=make_loader(fixture_val,4,False,device)
            print("SMOKE FIXTURE: no public download; 64 train-derived synthetic checks, not external generalization",flush=True)
        else:
            if cfg.get("data_profile")=="korean":
                subprocess.check_call([sys.executable,"-m","pip","install","-q","datasets==3.6.0","pyarrow==20.0.0","huggingface_hub<1"])
                korean=fleurs_speech("train")
                n=int(cfg.get("external_size",1024))
                train_set=torch.utils.data.ConcatDataset([train_set,SyntheticPairs(tr,tn,n//2,42,True),SyntheticPairs(korean,tn,n-n//2,43,True)])
            else:
                train_set=torch.utils.data.ConcatDataset([train_set,SyntheticPairs(tr,tn,int(cfg.get("external_size",1024)),42,True)])
            if cfg.get("challenge_remix",0):
                train_set=torch.utils.data.ConcatDataset([train_set,ChallengeRemix(original_train,int(cfg["challenge_remix"]),cfg["seed"],empirical=cfg.get("empirical_remix",False),balanced=cfg.get("balanced_remix",False),exclude=heldout)])
            external_data=torch.utils.data.ConcatDataset([SyntheticPairs(dv,dn,32,2026,False),SyntheticPairs(dv,dn,32,3026,False,stress=True)])
            external_val=make_loader(fixed_pairs(external_data),4,False,device)
        val_set = SpeechPairs(root, "validation")
        if len(val_set) != 164:
            raise RuntimeError("Unexpected validation split size")
        if perf:
            data_fingerprint=fingerprint_pairs(val_set,external_val.dataset)
            # Exact train pairs at each seed/epoch are defined by this schedule; hash epoch one.
            set_data_epoch(train_set,1)
            training_fingerprint=fingerprint_pairs(train_set)
            if expected and (data_fingerprint!=expected["data_fingerprint"] or training_fingerprint!=expected["training_fingerprint"]):
                raise RuntimeError("Approved performance data changed; use serial FP32")
        train_loader = make_loader(train_set,cfg["batch_size"],True,device,dynamic=not smoke)
        val_loader = make_loader(val_set,cfg["batch_size"],False,device)
    pesq_function = ensure_pesq()
    if cfg.get("public_recipe"):
        class RecipeModel(nn.Module):
            def __init__(self):
                super().__init__(); self.fn=public_recipe(cfg["public_recipe"],device)
            def forward(self,x): return torch.cat([self.fn(v) for v in x.split(1)])
        model=RecipeModel().to(device)
    elif cfg.get("architecture") in ("sepformer_dns_transfer","sepformer_wham_transfer"):
        model = SepformerTransfer().to(device)
    elif cfg.get("architecture") == "cmgan_transfer":
        model = CMGANTransfer().to(device)
    elif cfg.get("architecture") == "metricgan":
        model = MetricGANTransfer().to(device)
    elif cfg.get("architecture") == "composed":
        model = ComposedEnhancer().to(device)
    elif cfg.get("architecture") == "moe":
        model = AdaptiveMixture().to(device)
    elif cfg.get("architecture") == "generated":
        model = GeneratedEnhancer().to(device)
    elif cfg.get("architecture") == "temporal":
        model = TemporalEnhancer().to(device)
    elif cfg.get("architecture") == "dual_axis":
        model = DualAxisEnhancer().to(device)
    elif cfg.get("architecture") == "residual_relaxed":
        model = RelaxedPhaseUNet().to(device)
    elif cfg.get("architecture") == "complex_residual":
        model = ComplexResidualUNet().to(device)
    else:
        model = ResidualPhaseUNet().to(device)
    warm_ref = cfg.get("warm_ref")
    if warm_ref:
        slug = warm_ref.rsplit("/", 1)[-1]
        matches = list(Path("/kaggle/input").rglob("best_model.pt"))
        matches = [p for p in matches if slug in p.parts]
        if len(matches) != 1:
            raise RuntimeError("Expected one checkpoint from configured private kernel")
        model.load_state_dict(torch.load(matches[0], map_location=device, weights_only=True), strict=True)
        print("Warm start loaded", flush=True)
    unlabeled = NoisyOnly(root, bool(cfg.get("transductive"))) if cfg.get("self_supervised") and not is_audit else None
    unlabeled_loader = DataLoader(unlabeled,batch_size=cfg["batch_size"],shuffle=True) if unlabeled is not None else None
    if unlabeled_loader is not None:
        import copy
        teacher=copy.deepcopy(model).eval().requires_grad_(False)
    if cfg.get("final_audit"):
        model.eval()
        if not warm_ref and not cfg.get("public_recipe"):
            raise RuntimeError("Final audit requires frozen checkpoint or recipe")
        test_loader=DataLoader(torch.utils.data.ConcatDataset([SyntheticPairs(ts,tnoise,64,82491,False),SyntheticPairs(ts,tnoise,64,92491,False,stress=True)]),batch_size=4,shuffle=False)
        score,rows=evaluate(model,test_loader,device,pesq_function,save_rows=True)
        write_csv(WORK/"audit_per_file.csv",["index","pesq_wb"],rows)
        audit={"experiment":"V4_FINAL_AUDIT","audit_pesq":score,"audit_files":128,"warm_ref":warm_ref,"public_recipe":cfg.get("public_recipe")}
        if cfg.get("audit_korean"):
            try:
                subprocess.check_call([sys.executable,"-m","pip","install","-q","datasets==3.6.0","pyarrow==20.0.0","huggingface_hub<1"])
                korean=fleurs_speech("test",64)
                korean_loader=DataLoader(SyntheticPairs(korean,tnoise,64,99801,False,stress=True),batch_size=4,shuffle=False)
                kscore,krows=evaluate(model,korean_loader,device,pesq_function,save_rows=True)
                write_csv(WORK/"korean_audit.csv",["index","pesq_wb"],krows)
                audit["korean_synthetic"]={"pesq":kscore,"count":64,"reference":"FLEURS read-speech recordings; approximate clean"}
            except Exception as exc: audit["korean_error"]=repr(exc)
        if cfg.get("audit_urgent"):
            try:
                subprocess.check_call([sys.executable,"-m","pip","install","-q","datasets==3.6.0","pyarrow==20.0.0","huggingface_hub<1","torchmetrics==1.6.3","onnxruntime","librosa"])
                audit["urgent_paired"]=urgent_audit(model,device,pesq_function)
                audit["urgent_blind"]=urgent_audit(model,device,pesq_function,blind=True)
            except Exception as exc: audit["urgent_error"]=repr(exc)
        if cfg.get("audit_aux"):
            for name in ("dnsmos","wer"):
                try:
                    if name=="dnsmos":
                        subprocess.check_call([sys.executable,"-m","pip","install","-q","torchmetrics==1.6.3","onnxruntime","librosa"])
                        audit[name]=dnsmos_metrics(model,test_loader,device)
                    else:
                        subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers==4.51.3","huggingface_hub<1"])
                        audit[name]=transcription_audit(model,ts,tnoise,device)
                except Exception as exc: audit[name+"_error"]=repr(exc)
        (WORK/"result.json").write_text(json.dumps(audit,indent=2))
        return
    distill = None
    if cfg.get("distill"):
        distill=MetricGANTransfer().to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 0.00035)), weight_decay=0.0001)
    critic=PESQCritic().to(device) if cfg.get("pesq_aware") else None
    critic_optimizer=torch.optim.AdamW(critic.parameters(),lr=0.0003) if critic is not None else None
    replay=CriticReplay(seed=cfg["seed"]) if critic is not None and cfg.get("adaptive_critic") else None
    if replay is not None and critic_dev is None: raise ValueError("Adaptive critic requires disjoint train-only calibration")
    critic_reports=[]
    last_calibration=None
    use_amp=bool(cfg.get("amp")) and device.type=="cuda" and cfg.get("architecture") in ("temporal","composed")
    scaler=torch.amp.GradScaler("cuda") if use_amp else None
    print(f"TRAIN_PRECISION={'amp_fp16_forward_fp32_loss' if use_amp else 'fp32'}",flush=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(cfg["epochs"]), 1), eta_min=0.000025
    )
    if cfg.get("architecture")=="generated" and sum(p.numel() for p in model.parameters())>5000000:
        raise RuntimeError("Generated model exceeds parameter budget")
    probe=torch.randn(1,48000,device=device)*0.05
    out=model(probe)
    if out.shape!=probe.shape or not torch.isfinite(out).all():
        raise RuntimeError("Model contract failed")
    out.square().mean().backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
        raise RuntimeError("No finite gradient")
    optimizer.zero_grad(set_to_none=True)
    best_score, best_epoch, stale = float("-inf"), 0, 0
    best_joint = float("-inf")
    history = []
    started = time.monotonic()
    epoch_times=[]
    metric_probe={}
    critic_updates=0
    metric_steps=0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        if perf or cfg.get("challenge_remix") or cfg.get("v5_stage") or cfg.get("v52_stage"): set_data_epoch(train_set,epoch)
        losses = []
        epoch_started=time.monotonic()
        weight=adaptive_metric_weight(last_calibration,epoch,critic_updates*2) if replay is not None else (.05 if critic is not None and epoch>=3 else 0.)
        for batch_idx,(noisy, clean) in enumerate(train_loader,1):
            noisy, clean = noisy.to(device,non_blocking=True), clean.to(device,non_blocking=True)
            metric_active=critic is not None and weight>0
            update_metric=critic is not None and (batch_idx-1)%(4 if replay is not None else 8)==0
            probe=metric_probe if smoke and metric_active and not metric_probe else None
            losses.append(supervised_step(model,optimizer,noisy,clean,cfg,distill,teacher if unlabeled_loader is not None else None,scaler=scaler,critic=critic,critic_optimizer=critic_optimizer,pesq_fn=pesq_function,metric_weight=weight,update_critic=update_metric,metric_probe=probe,critic_replay=replay))
            critic_updates+=int(update_metric)
            metric_steps+=int(metric_active)
            if batch_idx==1 or batch_idx%10==0 or batch_idx==len(train_loader):
                telemetry={"event":"batch","epoch":epoch,"epochs":int(cfg["epochs"]),"batch":batch_idx,"batches":len(train_loader),"loss":float(losses[-1]),"elapsed_s":time.monotonic()-epoch_started}
                if device.type=="cuda":
                    telemetry.update(gpu_alloc_mib=torch.cuda.memory_allocated(device)/1048576,gpu_total_mib=torch.cuda.get_device_properties(device).total_memory/1048576)
                    try: telemetry["gpu_util"]=float(torch.cuda.utilization(device))
                    except Exception: pass
                if math.isfinite(telemetry["loss"]):
                    print("V4_TELEMETRY "+json.dumps(telemetry,allow_nan=False),flush=True)
        if unlabeled_loader is not None and epoch >= 2:
            for k,x in enumerate(unlabeled_loader):
                if k>=64: break
                consistency_step(model,optimizer,teacher,x.to(device),cfg.get("self_mode","consistency"))
        if critic is not None and critic_dev is not None:
            last_calibration=critic_calibration(model,critic,critic_dev,device,pesq_function)
            last_calibration["split"]="reserved train-only pairs (32)"
            last_calibration["final_epoch_model"]=False
            report={"epoch":epoch,"metric_weight":weight,"critic_label_groups":critic_updates,"critic_optimizer_steps":critic_updates*(2 if replay is not None else 1),**last_calibration}
            critic_reports.append(report)
            print("V52_CRITIC "+json.dumps(report,allow_nan=False),flush=True)
            (WORK/"critic_history.json").write_text(json.dumps(critic_reports,indent=2,allow_nan=False),encoding="utf-8")
        scheduler.step()
        val_pesq, _ = evaluate(model, val_loader, device, pesq_function)
        external_pesq, domain_rows = evaluate(model, external_val, device, pesq_function, save_rows=True)
        domain_scores=[val_pesq,float(np.mean([r[1] for r in domain_rows[:32]])),float(np.mean([r[1] for r in domain_rows[32:]]))]
        joint = 0.5*float(np.mean(domain_scores))+0.5*min(domain_scores)
        print(f"external_pesq={external_pesq:.4f} joint={joint:.4f}",flush=True)
        selection_score = val_pesq if cfg.get("selection_target") == "challenge" else joint
        if selection_score > best_joint + 0.001:
            best_joint = selection_score
            best_score, best_epoch, stale = val_pesq, epoch, 0
            torch.save(model.state_dict(), WORK / "best_model.pt")
        else:
            stale += 1
        history.append([epoch, float(np.mean(losses)), val_pesq, best_score, stale])
        write_csv(WORK / "history.csv",
                  ["epoch", "train_loss", "validation_pesq_wb", "best_pesq_wb", "stale_epochs"],
                  history)
        print(f"epoch={epoch} loss={np.mean(losses):.5f} val_pesq={val_pesq:.4f} best={best_score:.4f}", flush=True)
        telemetry={"event":"epoch","epoch":epoch,"loss":float(np.mean(losses)),"challenge":float(val_pesq),"external":float(external_pesq),"robust":float(joint),"best":float(best_score),"elapsed_s":time.monotonic()-epoch_started}
        if all(math.isfinite(v) for v in telemetry.values() if isinstance(v,(float,int))):
            print("V4_TELEMETRY "+json.dumps(telemetry,allow_nan=False),flush=True)
        epoch_times.append(time.monotonic()-epoch_started)
        if not perf and epoch >= 5 and stale >= int(cfg["patience"]):
            break

    calibration=None
    if cfg.get("smoke_metric_protocol")=="v5.1-three-epoch":
        if critic_updates<1 or metric_steps<1 or not metric_probe: raise RuntimeError("Metric smoke did not exercise critic and generator gradients")
        calibration=critic_calibration(model,critic,external_val,device,pesq_function)
        print("CRITIC_CALIBRATION",json.dumps(calibration,allow_nan=False),flush=True)
    model.load_state_dict(torch.load(WORK / "best_model.pt", map_location=device, weights_only=True))
    best_score, rows = evaluate(model, val_loader, device, pesq_function, save_rows=True)
    write_csv(WORK / "validation_per_file.csv", ["index", "pesq_wb"], rows)
    external_score, external_rows = evaluate(model, external_val, device, pesq_function, save_rows=True)
    write_csv(WORK / "external_validation.csv", ["index","pesq_wb"], external_rows)
    metrics={}
    try:
        subprocess.check_call([sys.executable,"-m","pip","install","-q","pystoi==0.4.1"])
        metrics={"challenge":additional_metrics(model,val_loader,device),"external":additional_metrics(model,external_val,device)}
    except Exception as exc:
        metrics={"error":repr(exc)}
    result = {
        "smoke":bool(cfg.get("smoke")),
        "smoke_metric_protocol":cfg.get("smoke_metric_protocol"),
        "critic_updates":critic_updates,
        "metric_generator_steps":metric_steps,
        "metric_gradient_norm":metric_probe.get("gradient_norm"),
        "critic_calibration":calibration,
        "precision":"amp_fp16_forward_fp32_loss" if use_amp else "fp32",
        "perf_reproducible":perf,
        "loader_workers":train_loader.num_workers,
        "data_fingerprint":data_fingerprint,
        "training_fingerprint":training_fingerprint,
        "epoch_seconds":epoch_times,
        "hardware":hardware,
        "metrics":metrics,
        "evaluation_protocol":"v4-smoke-fixture-v1" if smoke else "v4.2-fixed-dev-v2",
        "domain_scores":{"challenge":best_score,"external_normal":float(np.mean([r[1] for r in external_rows[:32]])),"external_stress":float(np.mean([r[1] for r in external_rows[32:]]))},
        "robust_score":0.5*float(np.mean([best_score,float(np.mean([r[1] for r in external_rows[:32]])),float(np.mean([r[1] for r in external_rows[32:]]))]))+0.5*min(best_score,float(np.mean([r[1] for r in external_rows[:32]])),float(np.mean([r[1] for r in external_rows[32:]]))),
        "distill":bool(cfg.get("distill")),
        "experiment": variant,
        "selection_target": cfg.get("selection_target","robust"),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_pesq_wb": best_score,
        "validation_files": len(val_set),
        "external_validation_pesq_wb": external_score,
        "external_validation_files":64,
        "joint_score": (best_score+external_score)/2,
        "warm_ref": warm_ref,
        "external_size":cfg.get("external_size",1024),
        "lr":cfg.get("lr",0.00035),
        "baseline_validation_pesq_wb": 1.6961209534144983,
        "delta_over_baseline": best_score - 1.6961209534144983,
        "seed": cfg["seed"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "model": type(model).__name__,
        "model_family":cfg.get("architecture","residual"),
        "loss_mode":cfg.get("loss_mode","base"),
        "pesq_aware":bool(cfg.get("pesq_aware")),
        "v5_stage":cfg.get("v5_stage"),
        "v52_stage":cfg.get("v52_stage"),
        "v52_requested_epochs":int(cfg["epochs"]) if cfg.get("v52_stage") else None,
        "phase_loss":bool(cfg.get("phase_loss")),
        "empirical_remix":bool(cfg.get("empirical_remix")),
        "balanced_remix":bool(cfg.get("balanced_remix")),
        "adaptive_critic":bool(cfg.get("adaptive_critic")),
        "critic_reserved_train_files":len(heldout),
        "critic_optimizer_steps":critic_updates*(2 if replay is not None else 1),
        "critic_last_calibration":last_calibration,
        "challenge_remix":int(cfg.get("challenge_remix",0)),
        "loss_weights":cfg.get("loss_weights"),
        "data_profile":cfg.get("data_profile","base"),
        "augmentation":cfg.get("augmentation","base"),
        "generated_hash":cfg.get("generated_hash"),
        "network_spec":cfg.get("network_spec"),
        "transductive_noisy":bool(cfg.get("transductive")) and bool(cfg.get("self_supervised")),
        "self_supervised":bool(cfg.get("self_supervised")),
        "self_mode":cfg.get("self_mode","consistency"),
        "unlabeled_files":len(unlabeled) if unlabeled is not None else 0,
        "architecture": cfg.get("architecture","residual"),
        "optimizer": "AdamW+CosineAnnealingLR",
        "loss_variant": variant,
        "checkpoint": "best_model.pt",
    }
    (WORK / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


def self_test():
    torch.manual_seed(42)
    model = ResidualPhaseUNet()
    noisy = torch.randn(1, 4096) * 0.1
    clean = noisy * 0.9
    out = model(noisy)
    assert out.shape == noisy.shape
    assert torch.isfinite(out).all()
    # Exactly zero phase correction at initialization.
    assert torch.equal(model.phase_head.weight, torch.zeros_like(model.phase_head.weight))
    loss = enhancement_loss(out, clean, "V3_RESIDUAL_PHASE")
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())
    print("SELF_TEST_PASSED: zero phase init, noisy-only forward and gradients")


if __name__ == "__main__":
    if "--prepare-public-cache" in sys.argv:
        public_assets(prepare_only=True)
    elif "--self-test" in sys.argv:
        self_test()
    else:
        try: run()
        finally: close_pesq_pool()
