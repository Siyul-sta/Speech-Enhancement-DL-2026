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
    if EXPERIMENT_CONFIG.get("v571_recipe"):
        family={"cmgan":"cmgan_transfer","metricgan":"metricgan","sepformer_dns":"sepformer_dns_transfer","sepformer_wham":"sepformer_wham_transfer"}[name]
        old=EXPERIMENT_CONFIG.get("architecture")
        EXPERIMENT_CONFIG["architecture"]=family
        try:
            cls={"cmgan_transfer":CMGANTransfer,"metricgan":MetricGANTransfer,"sepformer_dns_transfer":SepformerTransfer,"sepformer_wham_transfer":SepformerTransfer}[family]
            return cls().to(device).eval()
        finally: EXPERIMENT_CONFIG["architecture"]=old
    if name == "cmgan":
        return CMGANTransfer().to(device).eval()
    if name == "metricgan":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "speechbrain==1.0.3","huggingface_hub<1"])
        from speechbrain.inference.enhancement import SpectralMaskEnhancement
        engine = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir=str(Path("/tmp/ralph_v4_public_metricgan") if Path("/kaggle").exists() else WORK / "public_weights" / "metricgan"),
            run_opts={"device": str(device)})
        return lambda x: engine.enhance_batch(x, lengths=torch.ones(x.shape[0], device=x.device))
    if name in ("sepformer_dns","sepformer_wham"):
        subprocess.check_call([sys.executable,"-m","pip","install","-q","speechbrain==1.0.3","huggingface_hub<1"])
        from speechbrain.inference.separation import SepformerSeparation
        model_id={"sepformer_dns":"speechbrain/sepformer-dns4-16k-enhancement","sepformer_wham":"speechbrain/sepformer-wham16k-enhancement"}[name]
        engine=SepformerSeparation.from_hparams(source=model_id,savedir="/tmp/v4_"+name,run_opts={"device":str(device)})
        def infer(x):
            y=engine.separate_batch(x)
            if y.ndim!=3 or y.shape[-1]!=1: raise RuntimeError("Expected one enhanced source")
            return y.squeeze(-1)
        return infer
    if name == "deepfilter":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "deepfilternet==0.5.6"])
        # DF 0.5.6 uses the older torchaudio AudioMetaData import.
        import types, torchaudio
        try:
            from torchaudio.backend.common import AudioMetaData
        except ImportError:
            from torchaudio import AudioMetaData
            backend=types.ModuleType("torchaudio.backend"); backend.__path__=[]
            common=types.ModuleType("torchaudio.backend.common"); common.AudioMetaData=AudioMetaData
            backend.common=common; sys.modules["torchaudio.backend"]=backend; sys.modules["torchaudio.backend.common"]=common
        from df.enhance import init_df, enhance
        import torchaudio.functional as AF
        model, state, *_ = init_df()
        def infer(x):
            out = enhance(model, state, AF.resample(x.cpu(), 16000, 48000), pad=True)
            return AF.resample(out.cpu(), 48000, 16000).to(x.device)
        return infer
    raise ValueError(name)

def public_recipe(recipe,device):
    """Replay exactly the global validation-selected blend; no clean input."""
    parts=recipe.split(":")
    if len(parts)==2:
        name,alpha=parts
        alpha=float(alpha)
        if not 0<=alpha<=1: raise ValueError("Invalid blend")
        a=load_model(name,device)
        return lambda x: alpha*checked_output(a,x)+(1-alpha)*x
    if len(parts)==4 and parts[0]=="blend":
        _,a,b,alpha=parts; alpha=float(alpha)
        if not 0<=alpha<=1: raise ValueError("Invalid blend")
        fa,fb=load_model(a,device),load_model(b,device)
        return lambda x: alpha*checked_output(fa,x)+(1-alpha)*checked_output(fb,x)
    if len(parts)==4 and parts[0]=="mean3":
        fs=[load_model(name,device) for name in parts[1:]]
        return lambda x: sum(checked_output(fn,x) for fn in fs)/3
    raise ValueError("Invalid public recipe")

def checked_output(fn, noisy):
    y = fn(noisy)
    if y.shape != noisy.shape or not torch.isfinite(y).all():
        raise RuntimeError("Invalid enhanced waveform shape or values")
    return y.clamp(-1, 1)

class NoisyIdentity(nn.Module):
    def forward(self,x): return x


def export_recipe_expert(name,model):
    import hashlib,shutil
    family={"cmgan":"cmgan_transfer","metricgan":"metricgan","sepformer_dns":"sepformer_dns_transfer","sepformer_wham":"sepformer_wham_transfer"}[name]
    root=WORK/"recipe_experts"/name;root.mkdir(parents=True,exist_ok=False)
    checkpoint=root/"best_model.pt"
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},checkpoint)
    files={"best_model.pt":hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    if family!="cmgan_transfer":
        for filename in ("offline_hyperparams.yaml","offline_hyperparams.json"):
            shutil.copy2(WORK/filename,root/filename)
            files[filename]=hashlib.sha256((root/filename).read_bytes()).hexdigest()
        if json.loads((root/"offline_hyperparams.json").read_text())["family"]!=family: raise RuntimeError("Wrong archived public hparams")
    return {"family":family,"files":files}


def run():
    if "public_assets" not in globals(): raise RuntimeError("Use V4 controller to package shared evaluation helpers")
    WORK.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    root = locate_data_root()
    data = SpeechPairs(root, "validation")
    if len(data) != 164:
        raise RuntimeError("Expected 164 validation clips")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pesq_fn = ensure_pesq()
    names = ("metricgan", "deepfilter", "sepformer_dns", "sepformer_wham")
    if EXPERIMENT_CONFIG.get("v5_public"): names=names+("cmgan",)
    if EXPERIMENT_CONFIG.get("v571_recipe"): names=("cmgan","metricgan","sepformer_dns","sepformer_wham")
    predictions, errors, recipe_experts = {}, {}, {}
    noisy = torch.from_numpy(data.noisy)
    clean = torch.from_numpy(data.clean)
    challenge_count=len(noisy)
    _,dv,_,dn,_,_=public_assets(evaluation_only=True)
    external=torch.utils.data.ConcatDataset([SyntheticPairs(dv,dn,32,2026,False),SyntheticPairs(dv,dn,32,3026,False,stress=True)])
    pairs=[external[i] for i in range(64)]
    noisy=torch.cat([noisy,torch.stack([x for x,y in pairs])])
    clean=torch.cat([clean,torch.stack([y for x,y in pairs])])
    for name in names:
        try:
            fn = load_model(name, device)
            with torch.inference_mode():
                values = torch.cat([
                    checked_output(fn, x.to(device)).cpu()
                    for x in noisy.split(1)
                ])
            if EXPERIMENT_CONFIG.get("v571_recipe"): recipe_experts[name]=export_recipe_expert(name,fn)
            predictions[name]=values
            del fn
        except Exception as exc:
            errors[name] = repr(exc)
            print("MODEL_FAILED", name, repr(exc), flush=True)
    if not predictions:
        raise RuntimeError("No public model succeeded: " + str(errors))
    (WORK/"public_models_status.json").write_text(json.dumps({"succeeded":list(predictions),"errors":errors},indent=2),encoding="utf-8")
    # Fixed global blend grid; no per-file clean-informed oracle.
    import itertools
    def candidates():
        for name, pred in predictions.items():
            for alpha in (0.7,0.85,1.0):
                yield f"{name}:{alpha}",alpha*pred+(1-alpha)*noisy
        for a,b in itertools.combinations(predictions,2):
            for alpha in (0.25,0.5,0.75):
                yield f"blend:{a}:{b}:{alpha}",alpha*predictions[a]+(1-alpha)*predictions[b]
        if EXPERIMENT_CONFIG.get("v5_public"):
            for a,b,c in itertools.combinations(predictions,3):
                yield f"mean3:{a}:{b}:{c}",(predictions[a]+predictions[b]+predictions[c])/3
    scores = {}
    best_rows = None
    best = -float("inf")
    winner = None
    for name, pred in candidates():
        rows=[[i,score] for i,score in enumerate(score_waveforms(clean.numpy(),pred.numpy(),pesq_fn))]
        ds=[float(np.mean([row[1] for row in chunk])) for chunk in (rows[:challenge_count],rows[challenge_count:challenge_count+32],rows[challenge_count+32:])]
        robust=0.5*float(np.mean(ds))+0.5*min(ds)
        scores[name] = {"challenge":ds[0],"external_normal":ds[1],"external_stress":ds[2],"robust":robust}
        print("PESQ", name, scores[name], flush=True)
        objective=ds[0] if EXPERIMENT_CONFIG.get("selection_target")=="challenge" else robust
        if objective > best:
            best, winner, best_rows = objective, name, rows
    write_csv(WORK / "validation_per_file.csv", ["index","pesq_wb"], best_rows[:164])
    write_csv(WORK / "external_validation.csv", ["index","pesq_wb"], [[i,row[1]] for i,row in enumerate(best_rows[164:])])
    domains={k:scores[winner][k] for k in ("challenge","external_normal","external_stress")}
    external_mean=(domains["external_normal"]+domains["external_stress"])/2
    result = {"experiment":"V4_PUBLIC_ZERO_SHOT", "validation_files":164,
              "evaluation_protocol":"v4.2-fixed-dev-v2","domain_scores":domains,"external_validation_files":64,
              "external_validation_pesq_wb":external_mean,"joint_score":(domains["challenge"]+external_mean)/2,"robust_score":scores[winner]["robust"],
              "selection_target":EXPERIMENT_CONFIG.get("selection_target","robust"),"v5_public":bool(EXPERIMENT_CONFIG.get("v5_public")),
              "best_validation_pesq_wb":domains["challenge"], "winner":winner, "scores":scores,
              "errors":errors, "selection":"global validation blend, no per-file oracle",
              "v571_recipe":bool(EXPERIMENT_CONFIG.get("v571_recipe")),"recipe_experts":recipe_experts,
              "seed":42,"model_family":"public","v56_key":EXPERIMENT_CONFIG.get("v56_key"),
              "v56_source_sha":EXPERIMENT_CONFIG.get("v56_source_sha"),"v56_action":EXPERIMENT_CONFIG.get("v56_action"),
              "v56_requested_epochs":EXPERIMENT_CONFIG.get("epochs") if EXPERIMENT_CONFIG.get("v56_key") else None}
    (WORK / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL_RESULT", json.dumps(result), flush=True)

if __name__ == "__main__":
    try: run()
    finally:
        if "close_pesq_pool" in globals(): close_pesq_pool()
