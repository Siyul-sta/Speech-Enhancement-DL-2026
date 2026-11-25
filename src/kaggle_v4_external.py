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
        gain = torch.sigmoid(self.mask_head(d)).squeeze(1)
        phase_delta = 0.5 * torch.tanh(self.phase_head(d)).squeeze(1)
        phase_factor = torch.polar(torch.ones_like(phase_delta), phase_delta)
        enhanced_spec = spec * gain * phase_factor
        return torch.istft(
            enhanced_spec, n_fft=N_FFT, hop_length=HOP,
            window=self.window, length=noisy.shape[-1],
        )

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


@torch.no_grad()
def evaluate(model, loader, device, pesq_function, save_rows=False):
    model.eval()
    values, rows, index = [], [], 0
    for noisy, clean in loader:
        pred = model(noisy.to(device)).cpu().numpy()
        noisy_np, clean_np = noisy.numpy(), clean.numpy()
        for j in range(len(pred)):
            # Match the output WAV's PCM16 domain when scoring.
            wave = np.clip(pred[j], -1.0, 1.0)
            reference = np.clip(clean_np[j], -1.0, 1.0)
            score = float(pesq_function(SAMPLE_RATE, reference, wave, "wb"))
            if not np.isfinite(score):
                raise RuntimeError(f"Non-finite PESQ at validation file {index}")
            values.append(score)
            if save_rows:
                rows.append([index, score])
            index += 1
    if index != len(loader.dataset):
        raise RuntimeError("Validation is incomplete")
    return float(np.mean(values)), rows


def write_csv(path: Path, fields, records):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(records)



# Public training corpora: LibriSpeech, MUSAN and optional SLR73/SLR28.
TRAIN_RIRS=[]
def public_assets():
    import urllib.request, tarfile, hashlib
    base = Path("/tmp/ralph_v4_external") if Path("/kaggle").exists() else WORK / "external"
    base.mkdir(parents=True, exist_ok=True)
    sources = {"train":"https://www.openslr.org/resources/12/train-clean-100.tar.gz",
               "dev":"https://www.openslr.org/resources/12/dev-clean.tar.gz",
               "test":"https://www.openslr.org/resources/12/test-clean.tar.gz",
               "noise":"https://www.openslr.org/resources/17/musan.tar.gz"}
    records = {}
    if EXPERIMENT_CONFIG.get("data_profile")=="spanish":
        sources["extra_speech"]="https://www.openslr.org/resources/73/es_pe_female.zip"
    if EXPERIMENT_CONFIG.get("data_profile")=="rir":
        sources["rir"]="https://www.openslr.org/resources/28/rirs_noises.zip"
    for key,url in sources.items():
        archive = base / (key+(".zip" if url.endswith(".zip") else ".tar.gz"))
        target = base / key
        marker=target/".complete"
        digest_path=target/".sha256"
        if not marker.exists() or not digest_path.exists():
            import shutil
            if target.exists(): shutil.rmtree(target)
            target.mkdir(exist_ok=True)
            if not archive.exists():
                with urllib.request.urlopen(url,timeout=60) as response, open(str(archive)+".partial","wb") as out:
                    shutil.copyfileobj(response,out,1024*1024)
                Path(str(archive)+".partial").replace(archive)
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
        records[key]={"url":url,"sha256":digest_path.read_text(encoding="ascii"),"license":"CC-BY-SA-4.0" if key=="extra_speech" else "Apache-2.0" if key=="rir" else "CC-BY-4.0 (MUSAN individual terms apply)"}
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
    if not all((train,dev,tr,dv,test,te)):
        raise RuntimeError("Empty external corpus split")
    records["counts"]={"train":len(train),"dev":len(dev),"test":len(test),"train_noise":len(tr),"dev_noise":len(dv),"test_noise":len(te),"rir":len(TRAIN_RIRS)}
    (WORK/"external_manifest.json").write_text(json.dumps(records,indent=2))
    return train,dev,tr,dv,test,te

class SyntheticPairs(Dataset):
    def __init__(self,speech,noise,size,seed,dynamic,stress=False):
        self.speech,self.noise,self.size,self.seed,self.dynamic=speech,noise,size,seed,dynamic
        self.stress=stress
    def __len__(self):
        return self.size
    def __getitem__(self,i):
        import scipy.signal
        import math
        rng=random if self.dynamic else random.Random(self.seed+i)
        def clip(paths):
            y,sr=sf.read(str(rng.choice(paths)),dtype="float32",always_2d=True)
            y=y.mean(axis=1)
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
        if self.dynamic and TRAIN_RIRS and rng.random()<0.5:
            h,sr=sf.read(rng.choice(TRAIN_RIRS),dtype="float32",always_2d=True)
            h=h[:,0]
            if sr!=16000: h=scipy.signal.resample_poly(h,16000//math.gcd(sr,16000),sr//math.gcd(sr,16000))
            if len(h) and np.isfinite(h).all() and np.max(np.abs(h))>1e-8:
                h=h[np.argmax(np.abs(h)):][:16000]
                h=h/max(1e-8,float(np.linalg.norm(h)))
                x=scipy.signal.fftconvolve(x,h)[:len(x)]
        if self.stress or (self.dynamic and EXPERIMENT_CONFIG.get("augmentation")=="room_codec"):
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



class MetricGANTransfer(nn.Module):
    """Fine-tune public generator weights through differentiable STFT masking."""
    def __init__(self):
        super().__init__()
        subprocess.check_call([sys.executable,"-m","pip","install","-q","speechbrain==1.0.3"])
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

    root = locate_data_root()
    train_set = SpeechPairs(root, "train")
    if len(train_set) != 1296:
        raise RuntimeError("Unexpected train split size")
    tr, dv, tn, dn, ts, tnoise = public_assets()
    train_set = torch.utils.data.ConcatDataset([train_set, SyntheticPairs(tr,tn,int(cfg.get("external_size",1024)),42,True)])
    external_val = DataLoader(torch.utils.data.ConcatDataset([SyntheticPairs(dv,dn,32,2026,False),SyntheticPairs(dv,dn,32,3026,False,stress=True)]),batch_size=4,shuffle=False)
    val_set = SpeechPairs(root, "validation")
    if len(val_set) != 164:
        raise RuntimeError("Unexpected validation split size")
    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=0
    )
    pesq_function = ensure_pesq()
    if cfg.get("architecture") == "metricgan":
        model = MetricGANTransfer().to(device)
    elif cfg.get("architecture") == "moe":
        model = AdaptiveMixture().to(device)
    elif cfg.get("architecture") == "generated":
        model = GeneratedEnhancer().to(device)
    elif cfg.get("architecture") == "temporal":
        model = TemporalEnhancer().to(device)
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
    unlabeled = NoisyOnly(root, bool(cfg.get("transductive"))) if cfg.get("self_supervised") else None
    unlabeled_loader = DataLoader(unlabeled,batch_size=cfg["batch_size"],shuffle=True) if unlabeled is not None else None
    if unlabeled_loader is not None:
        import copy
        teacher=copy.deepcopy(model).eval().requires_grad_(False)
    if cfg.get("final_audit"):
        if not warm_ref:
            raise RuntimeError("Final audit requires frozen checkpoint")
        test_loader=DataLoader(torch.utils.data.ConcatDataset([SyntheticPairs(ts,tnoise,64,82491,False),SyntheticPairs(ts,tnoise,64,92491,False,stress=True)]),batch_size=4,shuffle=False)
        score,rows=evaluate(model,test_loader,device,pesq_function,save_rows=True)
        write_csv(WORK/"audit_per_file.csv",["index","pesq_wb"],rows)
        audit={"experiment":"V4_FINAL_AUDIT","audit_pesq":score,"audit_files":128,"warm_ref":warm_ref}
        if cfg.get("audit_aux"):
            for name in ("dnsmos","wer"):
                try:
                    if name=="dnsmos":
                        subprocess.check_call([sys.executable,"-m","pip","install","-q","torchmetrics","onnxruntime","librosa"])
                        audit[name]=dnsmos_metrics(model,test_loader,device)
                    else:
                        subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers"])
                        audit[name]=transcription_audit(model,ts,tnoise,device)
                except Exception as exc: audit[name+"_error"]=repr(exc)
        (WORK/"result.json").write_text(json.dumps(audit,indent=2))
        return
    distill = None
    if cfg.get("distill"):
        distill=MetricGANTransfer().to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 0.00035)), weight_decay=0.0001)
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
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        losses = []
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(noisy)
            loss = enhancement_loss(pred, clean, variant)
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
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite train loss at epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if unlabeled_loader is not None and epoch >= 2:
            for k,x in enumerate(unlabeled_loader):
                if k>=64: break
                x=x.to(device)
                with torch.no_grad():
                    target=teacher(x).clamp(-1,1)
                perturb=torch.randn_like(x)*0.01
                optimizer.zero_grad(set_to_none=True)
                loss=0.1*F.l1_loss(model((x+perturb).clamp(-1,1)),target)
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite consistency loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                optimizer.step()
                with torch.no_grad():
                    for t,p in zip(teacher.parameters(),model.parameters()):
                        t.lerp_(p,0.01)
        scheduler.step()
        val_pesq, _ = evaluate(model, val_loader, device, pesq_function)
        external_pesq, domain_rows = evaluate(model, external_val, device, pesq_function, save_rows=True)
        domain_scores=[val_pesq,float(np.mean([r[1] for r in domain_rows[:32]])),float(np.mean([r[1] for r in domain_rows[32:]]))]
        joint = 0.5*float(np.mean(domain_scores))+0.5*min(domain_scores)
        print(f"external_pesq={external_pesq:.4f} joint={joint:.4f}",flush=True)
        if joint > best_joint + 0.001:
            best_joint = joint
            best_score, best_epoch, stale = val_pesq, epoch, 0
            torch.save(model.state_dict(), WORK / "best_model.pt")
        else:
            stale += 1
        history.append([epoch, float(np.mean(losses)), val_pesq, best_score, stale])
        write_csv(WORK / "history.csv",
                  ["epoch", "train_loss", "validation_pesq_wb", "best_pesq_wb", "stale_epochs"],
                  history)
        print(f"epoch={epoch} loss={np.mean(losses):.5f} val_pesq={val_pesq:.4f} best={best_score:.4f}", flush=True)
        if epoch >= 5 and stale >= int(cfg["patience"]):
            break

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
        "metrics":metrics,
        "evaluation_protocol":"v4.2-fixed-dev-v2",
        "domain_scores":{"challenge":best_score,"external_normal":float(np.mean([r[1] for r in external_rows[:32]])),"external_stress":float(np.mean([r[1] for r in external_rows[32:]]))},
        "robust_score":0.5*float(np.mean([best_score,float(np.mean([r[1] for r in external_rows[:32]])),float(np.mean([r[1] for r in external_rows[32:]]))]))+0.5*min(best_score,float(np.mean([r[1] for r in external_rows[:32]])),float(np.mean([r[1] for r in external_rows[32:]]))),
        "distill":bool(cfg.get("distill")),
        "experiment": variant,
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
        "loss_weights":cfg.get("loss_weights"),
        "data_profile":cfg.get("data_profile","base"),
        "augmentation":cfg.get("augmentation","base"),
        "generated_hash":cfg.get("generated_hash"),
        "transductive_noisy":bool(cfg.get("transductive")) and bool(cfg.get("self_supervised")),
        "self_supervised":bool(cfg.get("self_supervised")),
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
    if "--self-test" in sys.argv:
        self_test()
    else:
        run()
