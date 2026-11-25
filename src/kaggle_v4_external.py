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
    target = WORK / "speech_data"
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



# Public training corpora: LibriSpeech and MUSAN.
def public_assets():
    import urllib.request, tarfile, hashlib
    base = WORK / "external"
    base.mkdir(parents=True, exist_ok=True)
    sources = {"train":"https://www.openslr.org/resources/12/train-clean-100.tar.gz",
               "dev":"https://www.openslr.org/resources/12/dev-clean.tar.gz",
               "noise":"https://www.openslr.org/resources/17/musan.tar.gz"}
    records = {}
    for key,url in sources.items():
        archive = base / (key+".tar.gz")
        target = base / key
        if not (target / ".complete").exists():
            if not archive.exists():
                urllib.request.urlretrieve(url, str(archive)+".partial")
                Path(str(archive)+".partial").replace(archive)
            target.mkdir(exist_ok=True)
            with tarfile.open(archive,"r:gz") as tf:
                tf.extractall(target, filter="data")
            (target / ".complete").touch()
        h=hashlib.sha256()
        with archive.open("rb") as f:
            for chunk in iter(lambda:f.read(1048576),b""):
                h.update(chunk)
        records[key]={"url":url,"sha256":h.hexdigest()}
    train=sorted((base/"train").rglob("*.flac"))
    dev=sorted((base/"dev").rglob("*.flac"))
    noises=sorted(p for p in (base/"noise").rglob("*.wav") if "noise" in p.relative_to(base/"noise").parts or "music" in p.relative_to(base/"noise").parts)
    groups=sorted({str(p.parent.relative_to(base/"noise")) for p in noises})
    random.Random(42).shuffle(groups)
    held=set(groups[:max(1,len(groups)//5)])
    tr=[p for p in noises if str(p.parent.relative_to(base/"noise")) not in held]
    dv=[p for p in noises if str(p.parent.relative_to(base/"noise")) in held]
    if not all((train,dev,tr,dv)):
        raise RuntimeError("Empty external corpus split")
    records["counts"]={"train":len(train),"dev":len(dev),"train_noise":len(tr),"dev_noise":len(dv)}
    (WORK/"external_manifest.json").write_text(json.dumps(records,indent=2))
    return train,dev,tr,dv

class SyntheticPairs(Dataset):
    def __init__(self,speech,noise,size,seed,dynamic):
        self.speech,self.noise,self.size,self.seed,self.dynamic=speech,noise,size,seed,dynamic
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
            if len(y)==0: raise ValueError("Empty audio")
            if len(y)<48000: y=np.tile(y,(48000+len(y)-1)//len(y))
            k=rng.randint(0,len(y)-48000)
            return y[k:k+48000].astype(np.float32)
        c,n=clip(self.speech),clip(self.noise)
        snr=rng.uniform(-5,20)
        n=n*(np.sqrt(np.mean(c*c)+1e-8)/np.sqrt(np.mean(n*n)+1e-8))*10**(-snr/20)
        x=c+n
        scale=max(1.0,float(np.max(np.abs(x))),float(np.max(np.abs(c))))
        return torch.from_numpy((x/scale).astype(np.float32)),torch.from_numpy((c/scale).astype(np.float32))



class MetricGANTransfer(nn.Module):
    """Fine-tune public generator weights through differentiable STFT masking."""
    def __init__(self):
        super().__init__()
        subprocess.check_call([sys.executable,"-m","pip","install","-q","speechbrain==1.0.3"])
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        engine=SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank",
            savedir=str(WORK/"metricgan_weights"),run_opts={"device":"cpu"},freeze_params=False)
        self.generator=engine.mods.enhance_model
        self.features=engine.hparams.compute_STFT
        self.resynthesize=engine.hparams.resynth
        self.generator.requires_grad_(True)
    def forward(self,x):
        from speechbrain.processing.features import spectral_magnitude
        mag=spectral_magnitude(self.features(x),power=0.5)
        features=torch.log1p(mag)
        mask=self.generator(features,lengths=torch.ones(x.size(0),device=x.device))
        if isinstance(mask,tuple): mask=mask[0]
        return self.resynthesize(torch.expm1(mask*features),x)

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
    def __init__(self,root):
        self.paths=[]
        for split in ("train","validation","test"):
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
    tr, dv, tn, dn = public_assets()
    train_set = torch.utils.data.ConcatDataset([train_set, SyntheticPairs(tr,tn,int(cfg.get("external_size",1024)),42,True)])
    external_val = DataLoader(SyntheticPairs(dv,dn,64,2026,False),batch_size=4,shuffle=False)
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
    unlabeled = NoisyOnly(root) if cfg.get("self_supervised") else None
    unlabeled_loader = DataLoader(unlabeled,batch_size=cfg["batch_size"],shuffle=True) if unlabeled is not None else None
    if unlabeled_loader is not None:
        import copy
        teacher=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 0.00035)), weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(int(cfg["epochs"]), 1), eta_min=0.000025
    )
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
        external_pesq, _ = evaluate(model, external_val, device, pesq_function)
        joint = (val_pesq + external_pesq)/2
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
    result = {
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
        "transductive_noisy":bool(cfg.get("self_supervised")),
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
