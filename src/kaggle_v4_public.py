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



# Public weights are downloaded locally in the private Kaggle kernel.
# Only noisy tensors are passed to enhancement models.
def load_model(name, device):
    if name == "metricgan":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "speechbrain==1.0.3"])
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        engine = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir=str(Path("/tmp/ralph_v4_public_metricgan") if Path("/kaggle").exists() else WORK / "public_weights" / "metricgan"),
            run_opts={"device": str(device)})
        return lambda x: engine.enhance_batch(x, lengths=torch.ones(x.shape[0], device=x.device))
    if name == "deepfilter":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "deepfilternet==0.5.6"])
        from df.enhance import init_df, enhance
        import torchaudio.functional as AF
        model, state, *_ = init_df()
        def infer(x):
            out = enhance(model, state, AF.resample(x.cpu(), 16000, 48000), pad=True)
            return AF.resample(out.cpu(), 48000, 16000).to(x.device)
        return infer
    raise ValueError(name)

def checked_output(fn, noisy):
    y = fn(noisy)
    if y.shape != noisy.shape or not torch.isfinite(y).all():
        raise RuntimeError("Invalid enhanced waveform shape or values")
    return y.clamp(-1, 1)

def run():
    WORK.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    root = locate_data_root()
    data = SpeechPairs(root, "validation")
    if len(data) != 164:
        raise RuntimeError("Expected 164 validation clips")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pesq_fn = ensure_pesq()
    names = ("metricgan", "deepfilter")
    predictions, errors = {}, {}
    noisy = torch.from_numpy(data.noisy)
    for name in names:
        try:
            fn = load_model(name, device)
            with torch.inference_mode():
                predictions[name] = torch.cat([
                    checked_output(fn, x.to(device)).cpu()
                    for x in noisy.split(1)
                ])
        except Exception as exc:
            errors[name] = repr(exc)
            print("MODEL_FAILED", name, repr(exc), flush=True)
    if not predictions:
        raise RuntimeError("No public model succeeded: " + str(errors))
    # Fixed global blend grid; no per-file clean-informed oracle.
    candidates = {}
    for name, pred in predictions.items():
        for alpha in (0.7, 0.85, 1.0):
            candidates[f"{name}:{alpha}"] = alpha * pred + (1-alpha) * noisy
    if len(predictions) == 2:
        for alpha in (0.25, 0.5, 0.75):
            candidates[f"blend:{alpha}"] = alpha * predictions["metricgan"] + (1-alpha) * predictions["deepfilter"]
    scores = {}
    best_rows = None
    best = -float("inf")
    winner = None
    for name, pred in candidates.items():
        rows = []
        for i in range(164):
            score = float(pesq_fn(16000, np.clip(data.clean[i], -1, 1), pred[i].clamp(-1,1).numpy(), "wb"))
            if not np.isfinite(score):
                raise RuntimeError("Nonfinite PESQ")
            rows.append([i, score])
        mean = float(np.mean([row[1] for row in rows]))
        scores[name] = mean
        print("PESQ", name, mean, flush=True)
        if mean > best:
            best, winner, best_rows = mean, name, rows
    write_csv(WORK / "validation_per_file.csv", ["index","pesq_wb"], best_rows)
    result = {"experiment":"V4_PUBLIC_ZERO_SHOT", "validation_files":164,
              "best_validation_pesq_wb":best, "winner":winner, "scores":scores,
              "errors":errors, "selection":"global validation blend, no per-file oracle",
              "seed":42}
    (WORK / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULT", json.dumps(result), flush=True)

if __name__ == "__main__":
    run()
