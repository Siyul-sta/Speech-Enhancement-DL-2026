"""Standalone Kaggle GPU trainer. Noisy -> U-Net -> enhanced WAV.
Only train/validation data are read. No raw speech or audio is written to output.
Generated config is injected by tools/kaggle_loop.py before Kaggle upload.
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

EXPERIMENT_CONFIG = {"id": "E02_UNET", "epochs": 25, "batch_size": 4, "patience": 7, "seed": 42}
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
        raise FileNotFoundError(f"Kaggle dataset is not mounted: {mount}")
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


class SpectrogramUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.middle = ConvBlock(64, 96)
        self.dec3 = ConvBlock(96 + 64, 64)
        self.dec2 = ConvBlock(64 + 32, 32)
        self.dec1 = ConvBlock(32 + 16, 16)
        self.head = nn.Conv2d(16, 1, 1)
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
        x = torch.log1p(spec.abs()).unsqueeze(1)
        a = self.enc1(x)
        b = self.enc2(F.avg_pool2d(a, 2, ceil_mode=True))
        c = self.enc3(F.avg_pool2d(b, 2, ceil_mode=True))
        d = self.middle(F.avg_pool2d(c, 2, ceil_mode=True))
        d = self.dec3(self.upcat(d, c))
        d = self.dec2(self.upcat(d, b))
        d = self.dec1(self.upcat(d, a))
        mask = torch.sigmoid(self.head(d)).squeeze(1)
        return torch.istft(
            spec * mask, n_fft=N_FFT, hop_length=HOP,
            window=self.window, length=noisy.shape[-1],
        )


def enhancement_loss(pred, clean, variant: str):
    # E02: waveform + 512-point STFT; E03: add 256/1024-point resolutions.
    resolutions = [(512, 128)] if variant == "E02_UNET" else [(256, 64), (512, 128), (1024, 256)]
    spec_loss = torch.zeros((), dtype=pred.dtype, device=pred.device)
    for size, hop in resolutions:
        window = torch.hann_window(size, device=pred.device)
        a = torch.stft(pred, size, hop_length=hop, window=window, return_complex=True).abs()
        b = torch.stft(clean, size, hop_length=hop, window=window, return_complex=True).abs()
        spec_loss = spec_loss + F.l1_loss(torch.log1p(a), torch.log1p(b))
    return 0.25 * F.l1_loss(pred, clean) + 0.75 * spec_loss / len(resolutions)


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


def run():
    cfg = EXPERIMENT_CONFIG
    variant = cfg["id"]
    if variant not in ("E02_UNET", "E03_MRSTFT"):
        raise ValueError(f"Unknown experiment: {variant}")
    WORK.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and Path("/kaggle").exists():
        raise RuntimeError("Kaggle GPU unavailable; refusing CPU full training")
    print(f"EXPERIMENT={variant} device={device} seed={cfg['seed']}", flush=True)

    root = locate_data_root()
    train_set = SpeechPairs(root, "train")
    val_set = SpeechPairs(root, "validation")
    train_loader = DataLoader(
        train_set, batch_size=cfg["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["batch_size"], shuffle=False, num_workers=0
    )
    pesq_function = ensure_pesq()
    model = SpectrogramUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    best_score, best_epoch, stale = float("-inf"), 0, 0
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
        val_pesq, _ = evaluate(model, val_loader, device, pesq_function)
        if val_pesq > best_score + 0.001:
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
    result = {
        "experiment": variant,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_pesq_wb": best_score,
        "validation_files": len(val_set),
        "baseline_validation_pesq_wb": 1.6961209534144983,
        "delta_over_baseline": best_score - 1.6961209534144983,
        "seed": cfg["seed"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "model": "SpectrogramUNet",
        "checkpoint": "best_model.pt",
    }
    (WORK / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


def self_test():
    model = SpectrogramUNet()
    noisy = torch.randn(1, 4096) * 0.1
    clean = noisy * 0.9
    output = model(noisy)
    assert output.shape == noisy.shape
    loss = enhancement_loss(output, clean, "E02_UNET")
    loss.backward()
    assert np.isfinite(float(loss.detach()))
    print("SELF_TEST_PASSED shape, finite loss and backward")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        run()
